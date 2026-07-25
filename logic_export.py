"""Export a sample/drum kit as a native Logic Pro EXS24 ("Sampler") instrument.

Writes a `<KitName>/` folder containing one WAV per sample plus a `<KitName>.exs`
that maps each sample to a MIDI key. Dropped into (or written directly to)
`~/Music/Audio Music Apps/Sampler Instruments/`, the kit shows up in Logic's
Sampler plugin instrument menu and plays natively.

The EXS24 binary layout is the modern little-endian ("TBOS") format, cloned
from Logic's own factory kits and from the user's Logic-authored reference
instrument ("Round Robin / Gated Hat Reference") for the round-robin and
hi-hat-choke group fields. See `logic-pro-export-feature` memory for the
field-by-field derivation.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

# Kick on C1 (MIDI 36), the Logic/GM drum-kit convention. App key index k maps
# to MIDI note BASE_MIDI_NOTE + k, so a full drum layout lands contiguously.
BASE_MIDI_NOTE = 36

# Standard user location Logic scans for Sampler instruments.
LOGIC_SAMPLER_INSTRUMENTS = (
    Path.home() / "Music" / "Audio Music Apps" / "Sampler Instruments"
)

_SAFE_RE = re.compile(r"[^A-Za-z0-9 _-]+")


def sanitize_name(name: str, fallback: str = "Kit") -> str:
    cleaned = _SAFE_RE.sub("", (name or "").strip()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or fallback


@dataclass
class ZoneSpec:
    """One sample mapped to one MIDI key, optionally within a velocity band."""
    audio: np.ndarray            # float32 mono, values in [-1, 1]
    name: str                    # base filename (no extension), unique per kit
    midi_note: int               # 0..127
    vel_low: int = 1             # 1..127
    vel_high: int = 127          # 1..127
    one_shot: bool = True        # drums: ignore note-off, play whole sample
    group_index: int = 0         # which EXS group this zone belongs to
    rr: bool = False             # round-robin alternate (opts 0x09, full vel)


# ---------------------------------------------------------------------------
# Building zones from the app's instrument model
# ---------------------------------------------------------------------------

def instrument_to_zonespecs(
    notes: dict[int, list],
    key_label: Callable[[int], str],
    base_note: int = BASE_MIDI_NOTE,
    one_shot: bool = True,
    round_robin: bool = False,
    process_audio: Optional[Callable[[object, np.ndarray], np.ndarray]] = None,
) -> list[ZoneSpec]:
    """Flatten `Instrument.notes` (key index -> [Segment]) into ZoneSpecs.

    Two layouts for keys with several stacked samples, both mirroring the
    user's Logic-authored reference instrument ("Round Robin / Gated Hat
    Reference", built and verified by ear in Logic's Sampler UI):

    - velocity split (default): a key's samples partition the velocity range —
      soft hits low, loud hits high — so the hit velocity picks one. One group
      per key. Matches the app's VEL LAYER mode.
    - round robin (`round_robin=True`): every sample is a FULL-velocity zone in
      ITS OWN group, and write_exs_kit chains a key's groups (@80 linked list)
      with the group "select by" mode set to round robin (@84 = 2) — exactly
      the encoding Logic's Sampler saves when you configure round robin in its
      UI. Matches the app's RANDOM mode.
    """
    valid: dict[int, list] = {}
    for key in sorted(notes):
        segs = [s for s in notes[key] if getattr(s, "audio", None) is not None
                and np.asarray(s.audio).size > 0]
        if segs:
            # Soft -> loud so velocity bands are stable and round-robin cycles in
            # a predictable order.
            valid[key] = sorted(segs, key=lambda s: getattr(s, "rms", 0.0))
    if not valid:
        return []

    # Per-segment audio, optionally run through `process_audio(seg, raw)` to bake
    # the app's per-sample effects (volume, filter, envelope, equal-loudness) into
    # the exported WAV. Cached by id(seg.audio) so a reused Segment.audio still
    # dedups to one WAV.
    _processed: dict[int, np.ndarray] = {}
    def aud(seg) -> np.ndarray:
        raw = np.asarray(seg.audio, dtype=np.float32).ravel()
        if process_audio is None:
            return raw
        key = id(seg.audio)
        if key not in _processed:
            out = np.asarray(process_audio(seg, raw), dtype=np.float32).ravel()
            _processed[key] = out
        return _processed[key]

    # Each key is its own group (group_index = the key's position in `valid`), so
    # keys never share a group. Logic's factory kits always use several groups;
    # a single kit-wide group leaves most keys silent in Logic.
    if not round_robin:
        zones: list[ZoneSpec] = []
        used: set[str] = set()
        for gidx, (key, segs) in enumerate(valid.items()):
            midi_note = int(np.clip(base_note + key, 0, 127))
            n = len(segs)
            label = sanitize_name(key_label(key), f"Key{key}")
            for i, seg in enumerate(segs):
                vlo = 1 if i == 0 else int(round(i * 127.0 / n)) + 1
                vhi = 127 if i == n - 1 else int(round((i + 1) * 127.0 / n))
                vlo = max(1, min(127, vlo))
                vhi = max(vlo, min(127, vhi))
                base = f"{label}" if n == 1 else f"{label}_{i + 1}"
                name = _unique(base, used)
                zones.append(ZoneSpec(
                    audio=aud(seg),
                    name=name, midi_note=midi_note,
                    vel_low=vlo, vel_high=vhi, one_shot=one_shot,
                    group_index=gidx,
                ))
        return zones

    # Round-robin, cloned from the reference: every sample is a FULL-velocity
    # (0-127) zone in ITS OWN group ("RR Kick 1/2/3" style). A key's groups get
    # consecutive indices so write_exs_kit can chain them (@80) and flag them
    # select-by-round-robin (@84 = 2).
    zones = []
    used = set()
    gi = 0
    for key, segs in valid.items():
        label = sanitize_name(key_label(key), f"Key{key}")
        n = len(segs)
        midi_note = int(np.clip(base_note + key, 0, 127))
        for i, seg in enumerate(segs):
            base = f"{label}" if n == 1 else f"{label}_{i + 1}"
            zones.append(ZoneSpec(
                audio=aud(seg), name=_unique(base, used), midi_note=midi_note,
                vel_low=0, vel_high=127, one_shot=one_shot,
                group_index=gi, rr=True,
            ))
            gi += 1
    return zones


def _unique(base: str, used: set[str]) -> str:
    name = base
    n = 2
    while name.lower() in used:
        name = f"{base}_{n}"
        n += 1
    used.add(name.lower())
    return name


# ---------------------------------------------------------------------------
# EXS24 binary writer — modern little-endian ("TBOS") format
#
# Templates come from Logic's own factory kits (see exs_templates.py); the
# round-robin and choke field semantics come from the user's Logic-authored,
# ear-verified reference instrument ("Round Robin / Gated Hat Reference"):
#
# - Velocity split: one group per key; zones partition the velocity range.
# - Round robin: one group PER SAMPLE; a key's groups are chained via the @80
#   previous-group link (head -1) AND flagged @84 = 2 ("select by: round
#   robin"). Both are required — the chain alone chokes (@84 missing was the
#   bug in every earlier attempt), and @90 must stay 0.
# - Hi-hat choke: @4 = exclusive class 1 + @2 = mono voices on the hat groups;
#   every other group is fully polyphonic (@2 = 0) with full release.
# ---------------------------------------------------------------------------

import exs_templates as _T

_MAGIC = b"TBOS"                 # little-endian block magic
_TYPE_INSTRUMENT = 0x00
_TYPE_ZONE = 0x01
_TYPE_GROUP = 0x02
_TYPE_SAMPLE = 0x03
_TYPE_PARAMS = 0x04
_TYPE_OUTPUT = 0x08              # the six cosmetic "output" blocks


def _le32(v: int) -> bytes:
    return struct.pack("<I", v & 0xFFFFFFFF)


def _ascii(s: str, size: int) -> bytes:
    raw = s.encode("ascii", "replace")[:size - 1]     # keep a NUL terminator
    return raw + b"\x00" * (size - len(raw))


def _put(buf: bytearray, off: int, data: bytes) -> None:
    buf[off:off + len(data)] = data


def _block(block_type: int, index: int, name: str, content: bytes) -> bytes:
    """84-byte block header + content, little-endian.

    Modern EXS marks the header type byte with 0x40; magic is "TBOS" (SOBT
    reversed), byte0 = 1 (little-endian flag)."""
    head = bytes((1, 1, 0, (block_type & 0x3F) | 0x40))
    head += _le32(len(content)) + _le32(index) + _le32(0) + _MAGIC
    head += _ascii(name, 64)
    return head + content


def _instrument_content(n_zones: int, n_groups: int, n_samples: int) -> bytes:
    b = bytearray(_T.INSTR_TMPL)
    _put(b, 4, _le32(n_zones))
    _put(b, 8, _le32(n_groups))
    _put(b, 12, _le32(n_samples))
    _put(b, 16, _le32(1))            # one params block
    _put(b, 28, _le32(0))            # no per-zone T7 articulation blocks
    _put(b, 32, _le32(0))            # no T8 output blocks (reference has none)
    _put(b, 40, _le32(0))            # no trailing bplist state block
    _put(b, 44, _le32(0))            # no T11 file-bookmark blocks (ref has 12)
    return bytes(b)


def _zone_content(z: ZoneSpec, sample_index: int, group_index: int,
                  length: int) -> bytes:
    b = bytearray(_T.ZONE_TMPL)     # a known-good full-range one-shot zone
    if z.rr:
        # Clone the reference's round-robin zones exactly: opts 0x09.
        b[0] = 0x09 if z.one_shot else 0x08
    else:
        # opts bit0 = one-shot (ignore note-off, play the whole sample). The
        # template has it set; honor z.one_shot for non-one-shot callers.
        b[0] = (b[0] | 0x01) if z.one_shot else (b[0] & ~0x01)
    root = int(np.clip(z.midi_note, 0, 127))
    b[1] = root                     # root key
    b[4] = 0                         # neutral fine-tune (template carries a per-sample offset)
    b[6] = root                     # key low
    b[7] = root                     # key high
    b[9] = int(np.clip(z.vel_low, 0, 127))
    b[10] = int(np.clip(z.vel_high, 1, 127))
    _put(b, 16, _le32(max(0, length)))   # sample end (frames)
    _put(b, 24, _le32(max(0, length)))
    _put(b, 88, _le32(group_index))
    _put(b, 92, _le32(sample_index))
    return bytes(b)


def _group_content(rr_prev: int = -1, select_rr: bool = False,
                   exclusive: int = 0, mono: bool = False) -> bytes:
    """One group, patched from the factory template. Field meanings come from
    the user's Logic-authored, ear-verified reference instrument
    ("Round Robin / Gated Hat Reference"):

    - ROUND-ROBIN = the @80 previous-group chain (head -1, each later group →
      the prior one) **plus @84 = 2** (the group's "select by" mode = round
      robin). @84 was the missing ingredient in every earlier attempt. @90
      stays 0 (an earlier "cycle flag = 1" guess was wrong and caused choking).
    - CHOKE = @4 = exclusive class (hats share class 1) with @2 = 1 (mono
      voices) — exactly how the reference gates its hats.
    - Everything else: polyphonic (@3 bit0), max voices (@2 = 0), @91 = 1 —
      straight from the template, matching the reference byte-for-byte."""
    b = bytearray(_T.GROUP_TMPL)
    b[2] = 1 if mono else 0
    b[4] = exclusive & 0xFF
    b[52] = 0 if mono else 4          # reference: 4 on RR/poly groups, 0 on mono
    _put(b, 80, _le32(rr_prev if rr_prev >= 0 else 0xFFFFFFFF))
    b[84] = 2 if select_rr else 0
    return bytes(b)


def _sample_content(length: int, sample_rate: int, bit_depth: int,
                    file_size: int, data_start: int, dir_path: str,
                    file_name: str) -> bytes:
    b = bytearray(_T.SAMPLE_TMPL)
    _put(b, 0, _le32(data_start))       # byte offset of PCM data in the WAV
    _put(b, 4, _le32(length))           # frames
    _put(b, 8, _le32(sample_rate))
    _put(b, 12, _le32(bit_depth))
    _put(b, 16, _le32(1))               # channels (mono)
    _put(b, 20, _le32(1))
    _put(b, 24, _le32(1))
    _put(b, 28, b"EVAW")                # "WAVE" 4cc, byte-reversed for LE
    _put(b, 32, _le32(file_size))
    _put(b, 80, _ascii(dir_path, 256))  # absolute directory holding the WAV
    _put(b, 336, _ascii(file_name, 264))
    return bytes(b)


def _wav_data_offset(path: Path) -> int:
    """Byte offset of the WAV 'data' chunk payload (44 for a canonical header)."""
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            return 44
        f.read(4)
        if f.read(4) != b"WAVE":
            return 44
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return 44
            cid = hdr[:4]
            size = struct.unpack("<I", hdr[4:])[0]
            if cid == b"data":
                return f.tell()
            f.seek(size + (size & 1), 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_exs_kit(
    kit_name: str,
    zones: list[ZoneSpec],
    sample_rate: int,
    dest_dir: Path,
    choke_notes: Optional[set] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """Write `<dest_dir>/<KitName>.exs` plus its WAVs, flat into `dest_dir`.

    No redundant per-kit subfolder: the .exs sits directly in `dest_dir` (the GUI
    points this at a single "Samplebot-3000" folder), and WAVs are named
    `<KitName> - <zone>.wav` so several kits can share the folder without clashing.

    `choke_notes`: MIDI notes whose groups join exclusive class 1 with mono
    voices — the hi-hat choke from the reference instrument (closed/pedal hats
    cut the open hat's ring; nothing else is ever gated).

    Returns the path to the written .exs. Raises ValueError if there are no zones.
    """
    if not zones:
        raise ValueError("nothing to export — the kit has no samples")

    safe_kit = sanitize_name(kit_name)
    kit_dir = Path(dest_dir).resolve()   # write flat here, no <KitName>/ subfolder
    kit_dir.mkdir(parents=True, exist_ok=True)
    dir_path = str(kit_dir)              # absolute path stored in each sample block

    bit_depth = 24
    subtype = "PCM_24"

    # 1) Map each zone to a WAV, deduped by audio-array identity. In both layouts
    #    every zone currently has distinct audio, so this is a 1:1 zone↔sample
    #    mapping; the dedup is kept only so a future caller that reuses one
    #    Segment.audio across zones still writes a single shared WAV.
    sample_of_zone: list[int] = []          # zone i -> sample index
    unique_zones: list[ZoneSpec] = []       # one representative ZoneSpec per WAV
    by_audio_id: dict[int, int] = {}
    for z in zones:
        aid = id(z.audio)
        si = by_audio_id.get(aid)
        if si is None:
            si = len(unique_zones)
            by_audio_id[aid] = si
            unique_zones.append(z)
        sample_of_zone.append(si)

    # 2) Write the unique WAVs and gather per-sample metadata.
    sample_meta = []  # (length, file_size, data_start, file_name)
    total = len(unique_zones)
    for si, z in enumerate(unique_zones):
        if progress:
            progress(f"writing {z.name}.wav", si / max(1, total) * 0.7)
        wav_path = kit_dir / f"{safe_kit} - {z.name}.wav"
        audio = np.clip(np.asarray(z.audio, dtype=np.float32).ravel(), -1.0, 1.0)
        sf.write(str(wav_path), audio, int(sample_rate), subtype=subtype)
        sample_meta.append((
            int(audio.shape[0]),
            int(wav_path.stat().st_size),
            _wav_data_offset(wav_path),
            wav_path.name,
        ))

    if progress:
        progress("building instrument", 0.8)

    # 3) Groups come straight from the zone group_index (one per key for
    #    velocity split, one per SAMPLE for round-robin). Per the reference
    #    instrument: a key that owns several round-robin groups gets them
    #    chained via @80 (head -1, each later group → the prior one) and every
    #    group in the chain flagged select-by-round-robin (@84 = 2). Groups on
    #    choke notes join exclusive class 1 with mono voices. Name each group
    #    after its first zone so Logic's group list is readable.
    n_groups = max((z.group_index for z in zones), default=0) + 1
    group_names: dict[int, str] = {}
    group_note: dict[int, int] = {}
    for z in zones:
        group_names.setdefault(z.group_index, z.name)
        group_note.setdefault(z.group_index, z.midi_note)
    note_groups: dict[int, list[int]] = {}
    for gi, note in group_note.items():
        note_groups.setdefault(note, []).append(gi)
    rr_prev = {gi: -1 for gi in range(n_groups)}
    select_rr = {gi: False for gi in range(n_groups)}
    for note, gis in note_groups.items():
        gis = sorted(gis)
        if len(gis) > 1:                      # round-robin chain for this key
            for j, gi in enumerate(gis):
                select_rr[gi] = True
                if j:
                    rr_prev[gi] = gis[j - 1]
    choke = choke_notes or set()

    # 4) Assemble the EXS blocks in file order: instrument, zones, groups,
    #    samples, params, then the six cosmetic output blocks.
    out = bytearray()
    out += _block(_TYPE_INSTRUMENT, 0, safe_kit,
                  _instrument_content(len(zones), n_groups, len(unique_zones)))
    for i, z in enumerate(zones):
        length = sample_meta[sample_of_zone[i]][0]
        out += _block(_TYPE_ZONE, i, z.name,
                      _zone_content(z, sample_index=sample_of_zone[i],
                                    group_index=z.group_index, length=length))
    for gi in range(n_groups):
        choked = group_note.get(gi) in choke
        out += _block(_TYPE_GROUP, gi, group_names.get(gi, safe_kit),
                      _group_content(rr_prev=rr_prev[gi],
                                     select_rr=select_rr[gi],
                                     exclusive=1 if choked else 0,
                                     mono=choked))
    for si, z in enumerate(unique_zones):
        length, file_size, data_start, file_name = sample_meta[si]
        out += _block(_TYPE_SAMPLE, si, z.name,
                      _sample_content(length, int(sample_rate), bit_depth,
                                      file_size, data_start, dir_path, file_name))
    out += _block(_TYPE_PARAMS, 0, "Default Param", bytes(_T.PARAMS_TMPL))
    # No T8/T11/bplist trailer blocks — the reference instrument has none of
    # ours (its T11 file-bookmarks are per-machine and regenerated by Logic).

    exs_path = kit_dir / f"{safe_kit}.exs"
    exs_path.write_bytes(bytes(out))
    if progress:
        progress("done", 1.0)
    return exs_path

