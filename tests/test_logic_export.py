"""Round-trip test for the EXS24 exporter (modern little-endian "TBOS" format).

Logic Pro itself can't run in CI, so instead we parse the written .exs back with
the same block layout Logic's own factory kits use and assert the structure is
self-consistent: block headers, instrument counts, zone<->sample linkage,
key/velocity ranges, the populated (anti-gating) params block, and that every
referenced WAV exists with a matching data-chunk offset.

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
        assert data[i] == 1, "expected little-endian flag (1)"
        assert data[i + 1] == 1 and data[i + 2] == 0, "version must be 1.0"
        btype = data[i + 3] & 0x3F           # modern format ORs the type with 0x40
        size = struct.unpack("<I", data[i + 4:i + 8])[0]
        index = struct.unpack("<I", data[i + 8:i + 12])[0]
        magic = data[i + 16:i + 20]
        assert magic == b"TBOS", f"bad magic {magic!r}"
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
        output_blocks = by_type[lx._TYPE_OUTPUT]

        assert len(inst) == 1 and len(group_blocks) == 1 and len(param_blocks) == 1
        assert len(zone_blocks) == 3 and len(sample_blocks) == 3
        assert len(output_blocks) == 6           # the six factory output blocks

        # Instrument declares the right block counts (n_t8 at offset 32).
        ic = inst[0][3]
        n_zones, n_groups, n_samples, n_params = struct.unpack("<IIII", ic[4:20])
        assert (n_zones, n_groups, n_samples, n_params) == (3, 1, 3, 1)
        assert struct.unpack("<I", ic[32:36])[0] == 6
        assert struct.unpack("<I", ic[40:44])[0] == 0    # no trailing bplist

        # The single group must be polyphonic (offset 3 bit0) with max voices
        # (offset 2 = 0) — the per-group half of the anti-gating fix.
        gc = group_blocks[0][3]
        assert gc[2] == 0 and (gc[3] & 1), "group must be polyphonic / max voices"

        # Zones: verify key mapping, single-key range, velocity split, linkage.
        kit_dir = exs_path.parent
        seen_notes = []
        for btype, index, name, content in zone_blocks:
            assert content[0] & 1, "drum zones must be one-shot"
            key = content[1]
            key_low = content[6]
            key_high = content[7]
            vel_low = content[9]
            vel_high = content[10]
            assert key == key_low == key_high, "drum zone spans one key"
            assert 1 <= vel_high <= 127 and vel_low <= vel_high
            group_index = struct.unpack("<I", content[88:92])[0]
            sample_index = struct.unpack("<I", content[92:96])[0]
            assert group_index == 0
            assert sample_index == index, "zone i must reference sample i"
            seen_notes.append(key)

        # key 0 -> MIDI 36 (two zones), key 5 -> MIDI 41 (one zone).
        assert sorted(seen_notes) == [36, 36, 41]

        # The two zones on note 36 must partition velocity with no gap/overlap.
        note36 = sorted(
            (struct.unpack("<I", c[92:96])[0], c[9], c[10])
            for t, i, n, c in zone_blocks if c[1] == 36
        )
        assert note36[-1][2] == 127
        assert note36[0][2] + 1 == note36[1][1], "velocity bands must be contiguous"

        # Samples: every referenced WAV exists and offsets/rates are right.
        for btype, index, name, content in sample_blocks:
            data_start = struct.unpack("<I", content[0:4])[0]
            length = struct.unpack("<I", content[4:8])[0]
            sr = struct.unpack("<I", content[8:12])[0]
            bit_depth = struct.unpack("<I", content[12:16])[0]
            channels = struct.unpack("<I", content[16:20])[0]
            ftype = content[28:32]
            dir_path = content[80:336].split(b"\x00", 1)[0].decode("ascii")
            file_name = content[336:600].split(b"\x00", 1)[0].decode("ascii")
            assert sr == 44100 and bit_depth == 24 and channels == 1
            assert ftype == b"EVAW", "WAVE 4cc must be byte-reversed for LE"
            wav = kit_dir / file_name
            assert wav.exists(), f"missing {file_name}"
            assert Path(dir_path) == kit_dir, "sample must store its absolute dir"
            assert data_start == lx._wav_data_offset(wav)
            assert length == 2000

        # Params block is the populated factory layout (1108 bytes) carrying the
        # global voice count that stops cross-key gating — NOT an all-zero block.
        pc = param_blocks[0][3]
        assert len(pc) == 1108
        assert struct.unpack("<I", pc[112:116])[0] == 16, "global voices must be 16"


def test_exs_round_robin():
    # key 0 -> 3 samples, key 5 -> 2, key 7 -> 1. Modern round-robin = several
    # FULL-velocity zones OVERLAPPING on one key, all in the single shared group;
    # Logic auto-cycles overlapping zones. NO per-sample groups, NO group chain,
    # NO cycle flag — factory kits leave those unset, and setting them makes Logic
    # steal one voice across the chain (the gating / only-one-sample-plays bug).
    notes = {
        0: [_make_seg(0.2), _make_seg(0.05), _make_seg(0.4)],
        5: [_make_seg(0.3), _make_seg(0.1)],
        7: [_make_seg(0.15)],
    }
    labels = {0: "KICK", 5: "SNR", 7: "HH"}
    zones = lx.instrument_to_zonespecs(
        notes, key_label=lambda k: labels[k], round_robin=True)

    # One zone per sample; every zone full-velocity and in the shared group 0.
    assert len(zones) == 6
    assert all(z.group_index == 0 for z in zones)
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

        # Exactly ONE group for the whole kit; one sample + one zone per sample.
        assert len(groups) == 1 and len(samples) == 6 and len(zone_blocks) == 6
        ic = by_type[lx._TYPE_INSTRUMENT][0][3]
        n_zones, n_groups, n_samples, _ = struct.unpack("<IIII", ic[4:20])
        assert (n_zones, n_groups, n_samples) == (6, 1, 6)

        # The shared group must be a clean factory group: no chain (@80 = -1),
        # no cycle flag (@90 = 0), polyphonic (@3 bit0), max voices (@2 = 0).
        gc = groups[0][3]
        assert struct.unpack("<i", gc[80:84])[0] == -1, "group must not chain"
        assert gc[90] == 0, "group must not set the cycle flag"
        assert gc[2] == 0 and (gc[3] & 1), "group must be polyphonic / max voices"

        # Each key's samples overlap: several full-velocity zones on one note, all
        # in group 0, each pointing at a distinct sample.
        by_note = {}
        for _t, _i, _n, c in zone_blocks:
            assert struct.unpack("<I", c[88:92])[0] == 0, "all zones share group 0"
            assert c[9] <= 1 and c[10] == 127, "round-robin zones are full-velocity"
            by_note.setdefault(c[1], []).append(struct.unpack("<I", c[92:96])[0])
        # key 0->MIDI 36 (3 samples), 5->41 (2), 7->43 (1)
        assert sorted(by_note) == [36, 41, 43]
        assert sorted(len(v) for v in by_note.values()) == [1, 2, 3]
        for note, sids in by_note.items():
            assert len(set(sids)) == len(sids), f"note {note} zones reuse a sample"

        kit_dir = exs_path.parent
        for _t, _i, _n, c in samples:
            fname = c[336:600].split(b"\x00", 1)[0].decode("ascii")
            assert (kit_dir / fname).exists()


if __name__ == "__main__":
    test_exs_roundtrip()
    test_exs_round_robin()
    print("ok")
