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
    gain_for: Optional[Callable[[object], float]] = None,
) -> list[ZoneSpec]:
    """Flatten `Instrument.notes` (key index -> [Segment]) into ZoneSpecs.

    Two layouts for keys with several stacked samples:

    - velocity split (default): soft hits on low velocities, loud on high —
      matches the app's VEL LAYER playback. One group; deterministic by velocity.
    - round robin (`round_robin=True`): every sample covers the full velocity
      range and lands in its OWN group; Logic cycles the groups so a different
      sample fires each hit — the Logic-side stand-in for the app's RANDOM mode.

    EXS24 cycles round-robin groups GLOBALLY, so a key with fewer samples than
    the deepest key would hit silent groups. We avoid that by padding each key's
    cycle: it reuses its own samples (same WAV, no duplication on disk) to fill
    up to the max depth, so every group has a zone for every key.
    """
    valid: dict[int, list] = {}
    for key in sorted(notes):
        segs = [s for s in notes[key] if getattr(s, "audio", None) is not None
                and np.asarray(s.audio).size > 0]
        if segs:
            # Soft -> loud so velocity bands (and the padding order) are stable.
            valid[key] = sorted(segs, key=lambda s: getattr(s, "rms", 0.0))
    if not valid:
        return []

    # Per-segment audio, optionally scaled by `gain_for` (e.g. equal-loudness).
    # Cached by id(seg.audio) so round-robin padding still dedups to one WAV
    # (same gained array object reused across a key's padded zones).
    _gained: dict[int, np.ndarray] = {}
    def aud(seg) -> np.ndarray:
        raw = np.asarray(seg.audio, dtype=np.float32).ravel()
        if gain_for is None:
            return raw
        key = id(seg.audio)
        if key not in _gained:
            g = float(gain_for(seg))
            _gained[key] = (raw * g).astype(np.float32) if g != 1.0 else raw
        return _gained[key]

    if not round_robin:
        zones: list[ZoneSpec] = []
        used: set[str] = set()
        for key, segs in valid.items():
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
                    group_index=0,
                ))
        return zones

    # Round-robin: depth = deepest key; group g holds every key's g-th sample
    # (cycling for shallower keys). Zones sharing a Segment.audio object dedup to
    # one WAV in write_exs_kit, so padding costs no extra files.
    depth = max(len(segs) for segs in valid.values())
    # Pre-name each key's real samples once; padded zones reuse those names/audio.
    used = set()
    per_key: dict[int, list[tuple[np.ndarray, str]]] = {}
    for key, segs in valid.items():
        label = sanitize_name(key_label(key), f"Key{key}")
        n = len(segs)
        entries = []
        for i, seg in enumerate(segs):
            base = f"{label}" if n == 1 else f"{label}_{i + 1}"
            entries.append((aud(seg), _unique(base, used)))
        per_key[key] = entries

    zones = []
    for g in range(depth):
        for key, entries in per_key.items():
            audio, name = entries[g % len(entries)]
            midi_note = int(np.clip(base_note + key, 0, 127))
            zones.append(ZoneSpec(
                audio=audio, name=name, midi_note=midi_note,
                vel_low=1, vel_high=127, one_shot=one_shot,
                group_index=g,
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
# EXS24 binary writer
# ---------------------------------------------------------------------------

_MAGIC = b"SOBT"                 # big-endian block magic
_TYPE_INSTRUMENT = 0x00
_TYPE_ZONE = 0x01
_TYPE_GROUP = 0x02
_TYPE_SAMPLE = 0x03
_TYPE_PARAMS = 0x04


def _u32(v: int) -> bytes:
    return struct.pack(">I", v & 0xFFFFFFFF)


def _u16(v: int) -> bytes:
    return struct.pack(">H", v & 0xFFFF)


def _s8(v: int) -> bytes:
    return struct.pack(">B", v & 0xFF)          # two's-complement in one byte


def _ascii(s: str, size: int) -> bytes:
    raw = s.encode("ascii", "replace")[:size]
    return raw + b"\x00" * (size - len(raw))


def _block(block_type: int, index: int, name: str, content: bytes) -> bytes:
    """84-byte block header + content (big-endian, per ConvertWithMoss)."""
    head = bytes((0, 1, 0, block_type & 0xFF))   # BE flag, version 1.0, type
    head += _u32(len(content)) + _u32(index) + _u32(0) + _MAGIC
    head += _ascii(name, 64)
    return head + content


def _instrument_content(n_zones: int, n_groups: int, n_samples: int,
                        n_params: int) -> bytes:
    return (_u32(0) + _u32(n_zones) + _u32(n_groups) + _u32(n_samples)
            + _u32(n_params) + _u32(0) * 5)


def _zone_content(z: ZoneSpec, sample_index: int, group_index: int,
                  length: int) -> bytes:
    pitch = True                                 # pitch tracking on (harmless: single key)
    opts = (1 if z.one_shot else 0) | (0 if pitch else 2) | 8   # bit3 velRangeOn
    b = bytearray()
    b += _s8(opts)
    b += _s8(int(np.clip(z.midi_note, 0, 127)))  # root key
    b += _s8(0)                                  # fine tuning
    b += _s8(0)                                  # pan
    b += _s8(0)                                  # volume adjust (dB)
    b += _s8(0)                                  # volume scale
    b += _s8(int(np.clip(z.midi_note, 0, 127)))  # key low
    b += _s8(int(np.clip(z.midi_note, 0, 127)))  # key high
    b += b"\x00"                                 # pad
    b += _s8(int(np.clip(z.vel_low, 1, 127)))
    b += _s8(int(np.clip(z.vel_high, 1, 127)))
    b += b"\x00"                                 # pad
    b += _u32(0)                                 # sample start
    b += _u32(max(0, length))                    # sample end (whole sample)
    b += _u32(0)                                 # loop start
    b += _u32(0)                                 # loop end
    b += _u32(0)                                 # loop crossfade
    b += _s8(0)                                  # loop tune
    b += _s8(0)                                  # loop options (loop off)
    b += _s8(0)                                  # loop direction
    b += b"\x00" * 42
    b += _s8(0)                                  # flex options
    b += _s8(0)                                  # flex speed
    b += _s8(0)                                  # tail tune
    b += _s8(0)                                  # coarse tuning
    b += b"\x00"                                 # pad
    b += _s8(0)                                  # output
    b += b"\x00" * 5
    b += _u32(group_index)                       # group index (-1 if none)
    b += _u32(sample_index)
    return bytes(b)


def _group_content(rr_seq: int = -1) -> bytes:
    """Group block. `rr_seq` >= 0 sets this group's round-robin sequence position
    (groups 0,1,2… cycle); -1 leaves round-robin off."""
    b = bytearray()
    b += _s8(0)          # volume
    b += _s8(0)          # pan
    b += _s8(0)          # polyphony (0 = max)
    b += _s8(0)          # options
    b += _s8(0)          # exclusive
    b += _s8(0)          # min velocity
    b += _s8(127)        # max velocity
    b += _s8(0)          # sample-select random offset
    b += b"\x00" * 8
    b += _u16(0)         # release-trigger time
    b += b"\x00" * 14
    b += _s8(128)        # velocity range crossfade (+128 bias)
    b += _s8(0)          # velocity crossfade type
    b += _s8(0)          # key-range crossfade type
    b += _s8(128)        # key range crossfade (+128 bias)
    b += b"\x00" * 2
    b += _s8(80)         # enable-by-tempo low
    b += _s8(140)        # enable-by-tempo high
    b += b"\x00"
    b += _s8(0)          # cutoff offset
    b += b"\x00"
    b += _s8(0)          # reso offset
    b += b"\x00" * 12
    b += _u32(0) * 4     # env1 attack/decay/sustain/release offsets
    b += b"\x00"
    b += _s8(0)          # release trigger
    b += _s8(0)          # output
    b += _s8(0)          # enable-by-note value
    b += b"\x00" * 4
    b += _u32(rr_seq if rr_seq >= 0 else 0xFFFFFFFF)  # round-robin group pos
    b += _s8(0)          # enable-by type
    b += _s8(0)          # enable-by control value
    b += _s8(0)          # control low
    b += _s8(127)        # control high
    b += _s8(0)          # start note
    b += _s8(127)        # end note
    b += _s8(0)          # midi channel
    b += _s8(0)          # articulation
    return bytes(b)


def _sample_content(length: int, sample_rate: int, bit_depth: int,
                    file_size: int, wave_data_start: int,
                    file_name: str) -> bytes:
    b = bytearray()
    b += _u32(wave_data_start)
    b += _u32(length)
    b += _u32(sample_rate)
    b += _u32(bit_depth)
    b += _u32(1)                 # channels
    b += _u32(1)                 # channels 2
    b += b"\x00" * 4
    b += _ascii("WAVE", 4)       # type (big-endian)
    b += _u32(file_size)
    b += _u32(0)                 # not compressed
    b += b"\x00" * 40
    b += _ascii("", 256)         # file path (empty -> same folder as .exs)
    b += _ascii(file_name, 256)
    return bytes(b)


def _params_content() -> bytes:
    # Exact byte layout ConvertWithMoss emits for the default/empty parameter
    # set: 100 "old" slots then 200 "new" slots, all zero -> Sampler defaults.
    return (_u32(100) + b"\x00" * (100 + 1000)
            + _u32(200) + b"\x00" * 400)


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
    """Write `<dest_dir>/<KitName>/<KitName>.exs` plus one WAV per zone.

    Returns the path to the written .exs file. Raises ValueError if there are no
    zones to export.
    """
    if not zones:
        raise ValueError("nothing to export — the kit has no samples")

    safe_kit = sanitize_name(kit_name)
    dest_dir = Path(dest_dir)
    kit_dir = dest_dir / safe_kit
    kit_dir.mkdir(parents=True, exist_ok=True)

    bit_depth = 24
    subtype = "PCM_24"

    # 1) Dedup samples by audio-array identity: round-robin padding reuses a
    #    key's own Segment.audio across groups, so those zones share one WAV.
    #    (Velocity-split zones all have distinct audio → 1:1 zone↔sample, as before.)
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
    sample_meta = []  # (length, file_size, wave_data_start, file_name)
    total = len(unique_zones)
    for si, z in enumerate(unique_zones):
        if progress:
            progress(f"writing {z.name}.wav", si / max(1, total) * 0.7)
        wav_path = kit_dir / f"{z.name}.wav"
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

    # 3) Groups: one per distinct group_index. >1 group ⇒ round-robin, so each
    #    group's sequence position is its index; a single group leaves RR off.
    n_groups = max((z.group_index for z in zones), default=0) + 1
    rr = n_groups > 1

    # 4) Assemble the EXS blocks: instrument, zones, groups, samples, params.
    out = bytearray()
    out += _block(_TYPE_INSTRUMENT, 0, safe_kit,
                  _instrument_content(len(zones), n_groups,
                                      len(unique_zones), 1))
    for i, z in enumerate(zones):
        length = sample_meta[sample_of_zone[i]][0]
        out += _block(_TYPE_ZONE, i, z.name,
                      _zone_content(z, sample_index=sample_of_zone[i],
                                    group_index=z.group_index, length=length))
    for gi in range(n_groups):
        out += _block(_TYPE_GROUP, gi, safe_kit,
                      _group_content(rr_seq=gi if rr else -1))
    for si, z in enumerate(unique_zones):
        length, file_size, data_start, file_name = sample_meta[si]
        out += _block(_TYPE_SAMPLE, si, z.name,
                      _sample_content(length, int(sample_rate), bit_depth,
                                      file_size, data_start, file_name))
    out += _block(_TYPE_PARAMS, 0, safe_kit, _params_content())

    exs_path = kit_dir / f"{safe_kit}.exs"
    exs_path.write_bytes(bytes(out))
    if progress:
        progress("done", 1.0)
    return exs_path
