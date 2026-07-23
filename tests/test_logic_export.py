"""Round-trip test for the EXS24 exporter.

Logic Pro itself can't run in CI, so instead we parse the written .exs back with
the same block layout ConvertWithMoss (the reference reader) uses and assert the
structure is self-consistent: block headers, instrument counts, zone<->sample
linkage, key/velocity ranges, and that every referenced WAV exists with a
matching data-chunk offset.

Run:  python -m pytest tests/test_logic_export.py -q
  or: python tests/test_logic_export.py
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logic_export as lx


def _parse_blocks(data: bytes):
    """Yield (type, index, name, content) for each 84-byte-header block."""
    i = 0
    blocks = []
    while i + 84 <= len(data):
        assert data[i] == 0, "expected big-endian flag (0)"
        assert data[i + 1] == 1 and data[i + 2] == 0, "version must be 1.0"
        btype = data[i + 3] & 0x0F
        size = struct.unpack(">I", data[i + 4:i + 8])[0]
        index = struct.unpack(">I", data[i + 8:i + 12])[0]
        magic = data[i + 16:i + 20]
        assert magic == b"SOBT", f"bad magic {magic!r}"
        name = data[i + 20:i + 84].split(b"\x00", 1)[0].decode("ascii")
        content = data[i + 84:i + 84 + size]
        assert len(content) == size, "truncated block content"
        blocks.append((btype, index, name, content))
        i += 84 + size
    assert i == len(data), "trailing bytes / misaligned blocks"
    return blocks


def _make_seg(rms, n=2000):
    class S:
        pass
    s = S()
    rng = np.random.default_rng(int(rms * 1000))
    s.audio = (rng.standard_normal(n).astype(np.float32) * 0.1)
    s.rms = rms
    return s


def test_exs_roundtrip():
    # key 0 -> two stacked samples (velocity split); key 5 -> one sample.
    notes = {
        0: [_make_seg(0.2), _make_seg(0.05)],   # deliberately out of rms order
        5: [_make_seg(0.3)],
    }
    labels = {0: "KICK", 5: "LO-T"}
    zones = lx.instrument_to_zonespecs(notes, key_label=lambda k: labels[k])

    # 3 zones total (2 on key 0 + 1 on key 5).
    assert len(zones) == 3

    with tempfile.TemporaryDirectory() as d:
        exs_path = lx.write_exs_kit("My Test Kit!!", zones, 44100, Path(d))
        assert exs_path.exists()
        assert exs_path.name == "My Test Kit.exs"   # sanitized

        blocks = _parse_blocks(exs_path.read_bytes())
        by_type = {}
        for b in blocks:
            by_type.setdefault(b[0], []).append(b)

        inst = by_type[lx._TYPE_INSTRUMENT]
        zone_blocks = by_type[lx._TYPE_ZONE]
        group_blocks = by_type[lx._TYPE_GROUP]
        sample_blocks = by_type[lx._TYPE_SAMPLE]
        param_blocks = by_type[lx._TYPE_PARAMS]

        assert len(inst) == 1 and len(group_blocks) == 1 and len(param_blocks) == 1
        assert len(zone_blocks) == 3 and len(sample_blocks) == 3

        # Instrument declares the right block counts.
        ic = inst[0][3]
        n_zones, n_groups, n_samples, n_params = struct.unpack(">IIII", ic[4:20])
        assert (n_zones, n_groups, n_samples, n_params) == (3, 1, 3, 1)

        # Zones: verify key mapping, single-key range, velocity split, linkage.
        kit_dir = exs_path.parent
        seen_notes = []
        for btype, index, name, content in zone_blocks:
            opts = content[0]
            assert opts & 1, "drum zones must be one-shot"
            assert opts & 8, "velocity range must be enabled"
            key = content[1]
            key_low = content[6]
            key_high = content[7]
            vel_low = content[9]
            vel_high = content[10]
            assert key == key_low == key_high, "drum zone spans one key"
            assert 1 <= vel_low <= vel_high <= 127
            group_index = struct.unpack(">I", content[88:92])[0]
            sample_index = struct.unpack(">I", content[92:96])[0]
            assert group_index == 0
            assert sample_index == index, "zone i must reference sample i"
            seen_notes.append(key)

        # key 0 -> MIDI 36 (two zones), key 5 -> MIDI 41 (one zone).
        assert sorted(seen_notes) == [36, 36, 41]

        # The two zones on note 36 must partition velocity with no gap/overlap.
        note36 = sorted(
            (struct.unpack(">I", c[92:96])[0], c[9], c[10])
            for t, i, n, c in zone_blocks if c[1] == 36
        )
        assert note36[0][1] == 1 and note36[-1][2] == 127
        assert note36[0][2] + 1 == note36[1][1], "velocity bands must be contiguous"

        # Samples: every referenced WAV exists and offsets/rates are right.
        for btype, index, name, content in sample_blocks:
            wave_data_start = struct.unpack(">I", content[0:4])[0]
            length = struct.unpack(">I", content[4:8])[0]
            sr = struct.unpack(">I", content[8:12])[0]
            bit_depth = struct.unpack(">I", content[12:16])[0]
            channels = struct.unpack(">I", content[16:20])[0]
            ftype = content[28:32]
            file_name = content[336:592].split(b"\x00", 1)[0].decode("ascii")
            assert sr == 44100 and bit_depth == 24 and channels == 1
            assert ftype == b"WAVE"
            wav = kit_dir / file_name
            assert wav.exists(), f"missing {file_name}"
            assert wave_data_start == lx._wav_data_offset(wav)
            assert length == 2000

        # Params block matches the fixed ConvertWithMoss default layout length.
        assert len(param_blocks[0][3]) == 4 + 100 + 1000 + 4 + 400


def test_exs_round_robin():
    # key 0 -> 3 samples, key 5 -> 2, key 7 -> 1. Round-robin = one group per
    # sample; a KEY's groups are chained via the EXS24 "previous group" link
    # (offset 80): chain head = -1, each later group -> the prior group's index.
    notes = {
        0: [_make_seg(0.2), _make_seg(0.05), _make_seg(0.4)],
        5: [_make_seg(0.3), _make_seg(0.1)],
        7: [_make_seg(0.15)],
    }
    labels = {0: "KICK", 5: "SNR", 7: "HH"}
    zones = lx.instrument_to_zonespecs(
        notes, key_label=lambda k: labels[k], round_robin=True)

    # One zone (and one group) per sample; every zone full-velocity.
    assert len(zones) == 6
    assert [z.group_index for z in zones] == [0, 1, 2, 3, 4, 5]
    assert all(z.vel_low == 1 and z.vel_high == 127 for z in zones)

    with tempfile.TemporaryDirectory() as d:
        exs_path = lx.write_exs_kit("RR Kit", zones, 44100, Path(d))
        blocks = _parse_blocks(exs_path.read_bytes())
        by_type = {}
        for b in blocks:
            by_type.setdefault(b[0], []).append(b)

        groups = by_type[lx._TYPE_GROUP]
        samples = by_type[lx._TYPE_SAMPLE]
        zone_blocks = by_type[lx._TYPE_ZONE]

        assert len(groups) == 6 and len(samples) == 6 and len(zone_blocks) == 6
        ic = by_type[lx._TYPE_INSTRUMENT][0][3]
        n_zones, n_groups, n_samples, _ = struct.unpack(">IIII", ic[4:20])
        assert (n_zones, n_groups, n_samples) == (6, 6, 6)

        # Read each group's round-robin previous-link (u32 at offset 80).
        rr_prev = {}
        for _t, gi, _n, c in groups:
            v = struct.unpack(">I", c[80:84])[0]
            rr_prev[gi] = -1 if v == 0xFFFFFFFF else v

        # Group each zone's group by MIDI note; each key must be a valid chain.
        by_note = {}
        for _t, _i, _n, c in zone_blocks:
            note = c[1]
            gidx = struct.unpack(">I", c[88:92])[0]
            by_note.setdefault(note, []).append(gidx)
        # key 0->MIDI 36 (3 samples), 5->41 (2), 7->43 (1)
        assert sorted(by_note) == [36, 41, 43]
        for note, gs in by_note.items():
            gs = sorted(gs)
            assert rr_prev[gs[0]] == -1, f"note {note} head not -1"
            for j in range(1, len(gs)):
                assert rr_prev[gs[j]] == gs[j - 1], f"note {note} broken chain"

        kit_dir = exs_path.parent
        for _t, _i, _n, c in samples:
            fname = c[336:592].split(b"\x00", 1)[0].decode("ascii")
            assert (kit_dir / fname).exists()


if __name__ == "__main__":
    test_exs_roundtrip()
    test_exs_round_robin()
    print("ok")
