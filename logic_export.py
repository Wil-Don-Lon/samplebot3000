"""Export a sample/drum kit as a native Logic Pro EXS24 ("Sampler") instrument.

Writes a `<KitName>/` folder containing one WAV per sample plus a `<KitName>.exs`
that maps each sample to a MIDI key. Dropped into (or written directly to)
`~/Music/Audio Music Apps/Sampler Instruments/`, the kit shows up in Logic's
Sampler plugin instrument menu and plays natively.

The EXS24 binary layout here mirrors, byte for byte, the known-good writer in
git-moss/ConvertWithMoss (LGPLv3) — big-endian, block magic "SOBT", samples
referenced by filename in the same folder. See `logic-pro-export-feature` memory
for the field-by-field derivation.
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

    Two layouts for keys with several stacked samples:

    - velocity split (default): a key's samples partition the velocity range —
      soft hits low, loud hits high — so the hit velocity picks one. This is how
      Logic's own factory drum kits give per-hit variation, and it exports
      reliably. Matches the app's VEL LAYER mode.
    - round robin (`round_robin=True`): every sample is a FULL-velocity zone and
      the i-th sample of every key goes in group i, so a key's samples overlap
      across groups. This mirrors how Logic's *consolidated* factory kits arrange
      round-robin alternates. NOTE: Logic's separate-WAV ("modern"/electronic)
      instrument format — the one this writer emits — is NOT known to actually
      rotate these at play time (its factory kits use one sample per key), so
      round-robin export may not audibly cycle in Logic. Prefer velocity split
      for reliable variation. Kept because it still carries every sample.

    Both layouts split the kit across several clean groups (factory kits always
    use several — one kit-wide group left most keys silent in Logic; a group
    chain stole voices → gating). Velocity split uses one group per KEY; round
    robin uses one group per RR position.
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

    # Round-robin: every sample on a key is a FULL-velocity zone (so a key's
    # samples overlap), and the i-th sample of every key goes in group i. That
    # scatters a key's samples across DIFFERENT groups, which is the only way
    # Logic rotates overlapping same-key zones (same-group overlap layers). Groups
    # stay clean (no chain), so rotation is polyphonic — matches Deep Crunch etc.
    zones = []
    used = set()
    for key, segs in valid.items():
        label = sanitize_name(key_label(key), f"Key{key}")
        n = len(segs)
        midi_note = int(np.clip(base_note + key, 0, 127))
        for i, seg in enumerate(segs):
            base = f"{label}" if n == 1 else f"{label}_{i + 1}"
            zones.append(ZoneSpec(
                audio=aud(seg), name=_unique(base, used), midi_note=midi_note,
                vel_low=1, vel_high=127, one_shot=one_shot,
                group_index=i,       # RR position -> its own group (cross-group RR)
            ))
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
# Rewritten 2026-07-24 to match Logic's own factory kits byte-for-byte (see
# exs_templates.py for how the templates were extracted). The previous writer
# emitted the old big-endian "SOBT" layout with an all-zero PARAMS block. The
# modern format carries a populated PARAMS block (16 voices) plus per-group
# polyphonic options.
#
# Layout: one group per key (factory kits always use several groups; a single
# kit-wide group leaves most keys silent in Logic). Round-robin is NOT a group
# field — it rides on multiple full-velocity zones OVERLAPPING on one key, which
# Logic's Sampler auto-cycles. An earlier writer chained per-sample groups via
# offset-80/90; in the modern format those offsets create voice-stealing (gating
# + only one sample sounding), so groups are now the factory template verbatim.
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
    _put(b, 32, _le32(len(_T.T8_NAMES)))
    _put(b, 40, _le32(0))            # no trailing bplist state block
    return bytes(b)


def _zone_content(z: ZoneSpec, sample_index: int, group_index: int,
                  length: int) -> bytes:
    b = bytearray(_T.ZONE_TMPL)     # a known-good full-range one-shot zone
    # opts bit0 = one-shot (ignore note-off, play the whole sample). The template
    # has it set; honor z.one_shot so a non-one-shot caller isn't silently overridden.
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


def _group_content() -> bytes:
    """One key's group, verbatim from the factory template: polyphonic (offset 3
    bit0 = 1), max voices (offset 2 = 0), no round-robin chain (offset 80 = -1)
    and no cycle flag (offset 90 = 0). Logic's own kits set exactly this even for
    keys with several round-robin samples — round-robin comes from the overlapping
    zones, never from group fields. Chaining groups here (an earlier attempt) made
    Logic steal one voice across the chain: gating, plus only one sample ever
    sounding while the zone display still cycled."""
    return bytes(_T.GROUP_TMPL)


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
    progress: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """Write `<dest_dir>/<KitName>.exs` plus its WAVs, flat into `dest_dir`.

    No redundant per-kit subfolder: the .exs sits directly in `dest_dir` (the GUI
    points this at a single "Samplebot-3000" folder), and WAVs are named
    `<KitName> - <zone>.wav` so several kits can share the folder without clashing.
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

    # 3) Groups come straight from the zone group_index (per key for velocity
    #    split, per RR position for round-robin — see instrument_to_zonespecs).
    #    Every group is the factory template verbatim; round-robin rides on how
    #    zones are spread across groups, not on group fields. Name each group
    #    after its first zone so Logic's group list is readable.
    n_groups = max((z.group_index for z in zones), default=0) + 1
    group_names: dict[int, str] = {}
    for z in zones:
        group_names.setdefault(z.group_index, z.name)

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
        out += _block(_TYPE_GROUP, gi, group_names.get(gi, safe_kit),
                      _group_content())
    for si, z in enumerate(unique_zones):
        length, file_size, data_start, file_name = sample_meta[si]
        out += _block(_TYPE_SAMPLE, si, z.name,
                      _sample_content(length, int(sample_rate), bit_depth,
                                      file_size, data_start, dir_path, file_name))
    out += _block(_TYPE_PARAMS, 0, "Default Param", bytes(_T.PARAMS_TMPL))
    for oi, oname in enumerate(_T.T8_NAMES):
        out += _block(_TYPE_OUTPUT, oi, oname, _T.T8_CONTENT)

    exs_path = kit_dir / f"{safe_kit}.exs"
    exs_path.write_bytes(bytes(out))
    if progress:
        progress("done", 1.0)
    return exs_path


# ---------------------------------------------------------------------------
# Round-robin writer — the classic ("108-byte zone") EXS format
#
# The modern format above never rotates multiple samples on one key (its factory
# kits are one-sample-per-key). Logic's *classic* EXS format does: several
# full-velocity zones sharing a group round-robin. It's an older, simpler layout
# (40-byte instrument, 108-byte zones, 120-byte groups, 592-byte samples, a
# 304-byte params block, NO output/bplist blocks) and its block header uses a
# plain type byte (no 0x40). Templated verbatim from Logic's "Electronic Snare
# D1 6" (a 2-way full-velocity round-robin), patching per-instance fields.
# ---------------------------------------------------------------------------

def _block_rr(block_type: int, index: int, name: str, content: bytes) -> bytes:
    """Classic-format block header: like _block but the type byte is plain (the
    modern format ORs it with 0x40; the classic format does not)."""
    head = bytes((1, 1, 0, block_type & 0x3F))
    head += _le32(len(content)) + _le32(index) + _le32(0) + _MAGIC
    head += _ascii(name, 64)
    return head + content


def _zone_rr_content(midi_note: int, group_index: int, sample_index: int,
                     length: int, alt_index: int) -> bytes:
    b = bytearray(_T.ZONE_RR_TMPL)
    # opts: the factory round-robin alternates carry 0x00 on the first zone of a
    # key and 0x08 on the rest — replicate that so Logic treats them as a cycle.
    b[0] = 0x00 if alt_index == 0 else 0x08
    root = int(np.clip(midi_note, 0, 127))
    b[1] = root                                  # root key
    b[6] = root                                  # key low  (single key)
    b[7] = root                                  # key high
    b[9] = 0                                      # full velocity range
    b[10] = 127
    _put(b, 16, _le32(max(0, length)))           # sample end (frames)
    _put(b, 24, _le32(max(0, length)))
    _put(b, 88, _le32(group_index))
    _put(b, 92, _le32(sample_index))
    return bytes(b)


def _sample_rr_content(length: int, sample_rate: int, bit_depth: int,
                       file_size: int, data_start: int, dir_path: str,
                       file_name: str) -> bytes:
    b = bytearray(_T.SAMPLE_RR_TMPL)
    _put(b, 0, _le32(data_start))
    _put(b, 4, _le32(length))
    _put(b, 8, _le32(sample_rate))
    _put(b, 12, _le32(bit_depth))
    _put(b, 16, _le32(1))                        # channels (mono)
    _put(b, 20, _le32(1))
    _put(b, 24, _le32(1))
    _put(b, 28, b"EVAW")                         # "WAVE" 4cc, byte-reversed
    _put(b, 32, _le32(file_size))
    _put(b, 80, _ascii(dir_path, 256))
    _put(b, 336, _ascii(file_name, len(_T.SAMPLE_RR_TMPL) - 336))
    return bytes(b)


def write_exs_kit_rr(
    kit_name: str,
    notes: dict[int, list],
    key_label: Callable[[int], str],
    sample_rate: int,
    dest_dir: Path,
    base_note: int = BASE_MIDI_NOTE,
    process_audio: Optional[Callable[[object, np.ndarray], np.ndarray]] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Path:
    """Write a round-robin kit in the classic EXS format: each key's samples
    become full-velocity zones in that key's own group, so Logic cycles them.

    `process_audio(seg, raw)` bakes the app's per-sample effects into the WAVs
    (same as the modern writer). Returns the written .exs path.
    """
    # Collect, per key, its samples soft->loud (the round-robin order).
    valid: dict[int, list] = {}
    for key in sorted(notes):
        segs = [s for s in notes[key] if getattr(s, "audio", None) is not None
                and np.asarray(s.audio).size > 0]
        if segs:
            valid[key] = sorted(segs, key=lambda s: getattr(s, "rms", 0.0))
    if not valid:
        raise ValueError("nothing to export — the kit has no samples")

    _processed: dict[int, np.ndarray] = {}
    def aud(seg) -> np.ndarray:
        raw = np.asarray(seg.audio, dtype=np.float32).ravel()
        if process_audio is None:
            return raw
        k = id(seg.audio)
        if k not in _processed:
            _processed[k] = np.asarray(process_audio(seg, raw),
                                       dtype=np.float32).ravel()
        return _processed[k]

    safe_kit = sanitize_name(kit_name)
    kit_dir = Path(dest_dir).resolve()
    kit_dir.mkdir(parents=True, exist_ok=True)
    dir_path = str(kit_dir)
    bit_depth, subtype = 24, "PCM_24"

    # One group per key; each group holds that key's full-velocity alternates.
    # (zone info, group index, alt index within the group) plus per-key group name.
    zone_infos: list[tuple] = []   # (audio, midi_note, group_index, alt_index)
    group_names: list[str] = []
    used: set[str] = set()
    zone_names: list[str] = []
    for gidx, (key, segs) in enumerate(valid.items()):
        label = sanitize_name(key_label(key), f"Key{key}")
        group_names.append(label)
        midi_note = int(np.clip(base_note + key, 0, 127))
        n = len(segs)
        for i, seg in enumerate(segs):
            base = label if n == 1 else f"{label}_{i + 1}"
            zone_names.append(_unique(base, used))
            zone_infos.append((aud(seg), midi_note, gidx, i))

    # Dedup WAVs by audio identity, write them, gather per-sample metadata.
    sample_of_zone: list[int] = []
    unique_audio: list[np.ndarray] = []
    unique_name: list[str] = []
    by_id: dict[int, int] = {}
    for (audio, _mn, _g, _a), zname in zip(zone_infos, zone_names):
        aid = id(audio)
        si = by_id.get(aid)
        if si is None:
            si = len(unique_audio)
            by_id[aid] = si
            unique_audio.append(audio)
            unique_name.append(zname)
        sample_of_zone.append(si)

    sample_meta = []
    total = len(unique_audio)
    for si, audio in enumerate(unique_audio):
        if progress:
            progress(f"writing {unique_name[si]}.wav", si / max(1, total) * 0.7)
        wav_path = kit_dir / f"{safe_kit} - {unique_name[si]}.wav"
        a = np.clip(np.asarray(audio, dtype=np.float32).ravel(), -1.0, 1.0)
        sf.write(str(wav_path), a, int(sample_rate), subtype=subtype)
        sample_meta.append((int(a.shape[0]), int(wav_path.stat().st_size),
                            _wav_data_offset(wav_path), wav_path.name))

    if progress:
        progress("building instrument", 0.8)

    n_zones = len(zone_infos)
    n_groups = len(group_names)
    n_samples = len(unique_audio)

    out = bytearray()
    instr = bytearray(_T.INSTR_RR_TMPL)
    _put(instr, 4, _le32(n_zones))
    _put(instr, 8, _le32(n_groups))
    _put(instr, 12, _le32(n_samples))
    _put(instr, 16, _le32(1))
    out += _block_rr(_TYPE_INSTRUMENT, 0, safe_kit, bytes(instr))
    for i, (audio, midi_note, gidx, alt) in enumerate(zone_infos):
        length = sample_meta[sample_of_zone[i]][0]
        out += _block_rr(_TYPE_ZONE, i, zone_names[i],
                         _zone_rr_content(midi_note, gidx, sample_of_zone[i],
                                          length, alt))
    for gi, gname in enumerate(group_names):
        out += _block_rr(_TYPE_GROUP, gi, gname, bytes(_T.GROUP_RR_TMPL))
    for si in range(n_samples):
        length, file_size, data_start, file_name = sample_meta[si]
        out += _block_rr(_TYPE_SAMPLE, si, unique_name[si],
                         _sample_rr_content(length, int(sample_rate), bit_depth,
                                            file_size, data_start, dir_path,
                                            file_name))
    out += _block_rr(_TYPE_PARAMS, 0, "Default Param", bytes(_T.PARAMS_RR_TMPL))

    exs_path = kit_dir / f"{safe_kit}.exs"
    exs_path.write_bytes(bytes(out))
    if progress:
        progress("done", 1.0)
    return exs_path
