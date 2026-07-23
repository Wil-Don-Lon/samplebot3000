"""Mine labeled drum one-shots from Logic's factory content (and saved kits).

Logic's Drum Kit Designer ships acoustic kits as EXS sampler instruments whose
audio is "consolidated" — every velocity layer / round-robin for a piece is
concatenated into one long CAF, and each EXS *zone* points at a [start, end]
frame slice of it. This reads the EXS (little-endian, `0x40`-prefixed variant),
slices each zone back out as an individual hit, labels it from the sample
filename (Kick / Snare / HiHat / Ride / Crash / Clap / Tom), and writes it as a
WAV into the classifier's `training data/<bin>/` folders.

Also ingests Samplebot's own saved kits as reinforcement data: after you
recategorize a kit, each sample's key = a human-verified label.

CLI:
  python logic_import.py --scan                 # report what's minable
  python logic_import.py --mine [--per 12] [--limit-kits N]
  python logic_import.py --reinforce            # from kits/*
"""

from __future__ import annotations

import argparse
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile as sf

# Logic factory content root (macOS).
LOGIC_ROOT = Path("/Library/Application Support/Logic")
DKD_EXS = (LOGIC_ROOT / "Sampler Instruments" / "03 Drums & Percussion"
           / "04 Drum Kit Designer")

TRAIN_ROOT = Path(__file__).resolve().parent / "training data"

# Provenance prefixes so mined/reinforcement files are identifiable & removable.
LOGIC_PREFIX = "LGC_"
REINFORCE_PREFIX = "RNF_"

# Ordered keyword -> training bin. First match wins, so specific terms
# ("floor tom", "hihat") must precede general ones ("tom"). Bins mirror
# classifier.TRAIN_BINS. Percussion (shaker/tambourine/cowbell/sticks) is
# intentionally unmapped -> skipped, to keep the bins clean.
KEYWORD_TO_BIN: list[tuple[str, str]] = [
    ("kick", "kick"),
    ("sidestick", "snare"), ("side stick", "snare"), ("rimshot", "snare"),
    ("snare", "snare"),
    ("hihat", "hats"), ("hi hat", "hats"), ("open hat", "hats"),
    ("closed hat", "hats"), (" hh ", "hats"), ("hh_", "hats"),
    ("ride", "ride"),
    ("splash", "crash"), ("china", "crash"), ("crash", "crash"),
    ("clap", "clap"),
    ("floor tom", "lotom"), ("floortom", "lotom"),
]
# Generic toms are ambiguous (hi/mid/lo); mapped by MIDI key at slice time only
# when --toms is on. See label_for().


def label_for(sample_name: str, key: int, allow_toms: bool) -> Optional[str]:
    """Training bin for a hit, from its (consolidated) sample filename + key."""
    s = f" {sample_name.lower()} "
    for kw, bin_name in KEYWORD_TO_BIN:
        if kw in s:
            return bin_name
    if "tom" in s and allow_toms:
        # Toms spread across the keyboard: low key = floor/low, high = high tom.
        if key <= 45:
            return "lotom"
        if key >= 50:
            return "hitom"
        return "midtom"
    return None


# ---------------------------------------------------------------------------
# Minimal EXS reader (little- or big-endian, tolerant of the 0x40 type prefix)
# ---------------------------------------------------------------------------

@dataclass
class _Zone:
    name: str
    key: int
    sample_start: int
    sample_end: int
    sample_index: int


@dataclass
class _Sample:
    name: str
    length: int
    sample_rate: int
    file_name: str
    file_path: str


@dataclass
class ExsInstrument:
    zones: list[_Zone]
    samples: list[_Sample]


def read_exs(path: Path) -> ExsInstrument:
    data = Path(path).read_bytes()
    zones: list[_Zone] = []
    samples: list[_Sample] = []
    i = 0
    n = len(data)
    while i + 84 <= n:
        little = data[i] == 1                 # 0 = big-endian, 1 = little
        fmt = "<I" if little else ">I"
        btype = data[i + 3] & 0x0F            # tolerate the 0x40 "new format" bit
        size = struct.unpack(fmt, data[i + 4:i + 8])[0]
        if size < 0 or i + 84 + size > n:
            break
        content = data[i + 84:i + 84 + size]
        name = data[i + 20:i + 84].split(b"\x00", 1)[0].decode("latin1", "replace")
        if btype == 0x01 and len(content) >= 96:
            zones.append(_Zone(
                name=name,
                key=content[1],
                sample_start=struct.unpack(fmt, content[12:16])[0],
                sample_end=struct.unpack(fmt, content[16:20])[0],
                sample_index=struct.unpack(fmt, content[92:96])[0],
            ))
        elif btype == 0x03 and len(content) >= 592:
            fp = content[80:336].split(b"\x00", 1)[0].decode("latin1", "replace")
            fn = content[336:592].split(b"\x00", 1)[0].decode("latin1", "replace")
            samples.append(_Sample(
                name=name,
                length=struct.unpack(fmt, content[4:8])[0],
                sample_rate=struct.unpack(fmt, content[8:12])[0] or 44100,
                file_name=fn,
                file_path=fp,
            ))
        i += 84 + size
    return ExsInstrument(zones=zones, samples=samples)


def _resolve_sample_file(exs_path: Path, sample: _Sample) -> Optional[Path]:
    """Find the CAF/WAV a sample block refers to, relative to the .exs folder."""
    fn = sample.file_name or Path(sample.file_path).name
    if not fn:
        return None
    here = exs_path.parent
    # 1) exact name near the exs; 2) the Consolidated mirror of the DKD tree.
    for cand in (here / fn, *here.rglob(fn)):
        if cand.is_file():
            return cand
    consolidated = (LOGIC_ROOT / "EXS Factory Samples"
                    / "Drum Kit Designer Consolidated")
    for cand in consolidated.rglob(fn):
        return cand
    return None


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

# GM hi-hat articulation keys -> leaf label (used when split_hats is on).
HAT_KEY_ARTICULATION = {42: "hats_closed", 44: "hats_closed", 46: "hats_open"}


@dataclass
class Hit:
    audio: np.ndarray        # float32 mono
    sample_rate: int
    label: str
    source: str              # for the output filename
    kit: str = ""            # source kit stem (group for leave-one-kit-out CV)
    key: int = -1            # zone MIDI key


def iter_kit_hits(exs_path: Path, allow_toms: bool,
                  per_piece: Optional[int],
                  split_hats: bool = False) -> Iterator[Hit]:
    """Yield labeled one-shots sliced from one Drum Kit Designer .exs.

    `split_hats`: when on, refine the generic "hats" label to
    "hats_closed"/"hats_open" from the zone's GM MIDI key (42/44 vs 46).
    """
    inst = read_exs(exs_path)
    if not inst.zones or not inst.samples:
        return
    # Cache each referenced audio file (mono) once; slice many zones from it.
    file_cache: dict[int, Optional[np.ndarray]] = {}
    kept: Counter = Counter()
    kit = exs_path.stem
    for z in inst.zones:
        if z.sample_index >= len(inst.samples):
            continue
        sample = inst.samples[z.sample_index]
        label = label_for(sample.file_name or sample.name, z.key, allow_toms)
        if label is None:
            continue
        if split_hats and label == "hats":
            label = HAT_KEY_ARTICULATION.get(z.key, "hats")
        piece_key = (label, sample.file_name)
        if per_piece is not None and kept[piece_key] >= per_piece:
            continue
        if z.sample_index not in file_cache:
            resolved = _resolve_sample_file(exs_path, sample)
            file_cache[z.sample_index] = _load_mono(resolved) if resolved else None
        audio = file_cache[z.sample_index]
        if audio is None:
            continue
        start = max(0, z.sample_start)
        end = z.sample_end if z.sample_end > start else audio.shape[0]
        end = min(end, audio.shape[0])
        clip = audio[start:end]
        if clip.shape[0] < int(0.03 * sample.sample_rate):   # skip < 30ms
            continue
        kept[piece_key] += 1
        yield Hit(audio=np.ascontiguousarray(clip, dtype=np.float32),
                  sample_rate=sample.sample_rate, label=label,
                  source=f"{kit}_{z.key}_{kept[piece_key]}",
                  kit=kit, key=z.key)


def _load_mono(path: Path) -> Optional[np.ndarray]:
    try:
        audio, _sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:80]


def mine_drum_kit_designer(per_piece: Optional[int] = 12,
                           limit_kits: Optional[int] = None,
                           allow_toms: bool = False,
                           dest_root: Path = TRAIN_ROOT,
                           dry_run: bool = False) -> Counter:
    """Slice DKD kits into labeled one-shots under `dest_root/<bin>/`."""
    exs_files = sorted(DKD_EXS.rglob("*.exs"))
    if limit_kits:
        exs_files = exs_files[:limit_kits]
    written: Counter = Counter()
    for ei, exs in enumerate(exs_files):
        for hit in iter_kit_hits(exs, allow_toms, per_piece):
            if not dry_run:
                out_dir = dest_root / hit.label
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{LOGIC_PREFIX}{_safe(hit.source)}.wav"
                sf.write(str(out_dir / fname), hit.audio,
                         hit.sample_rate, subtype="PCM_24")
            written[hit.label] += 1
        print(f"  [{ei + 1}/{len(exs_files)}] {exs.stem:40s} "
              f"total so far: {sum(written.values())}")
    return written


def reinforce_from_kits(kits_root: Optional[Path] = None,
                        dest_root: Path = TRAIN_ROOT,
                        dry_run: bool = False) -> Counter:
    """Ingest saved Samplebot kits as reinforcement data.

    Only drum-layout kits are used (their key -> drum-piece mapping is known);
    each sample is labeled by the class of the key it was placed on.
    """
    import kit_store
    from pipeline import DRUM_KEYMAP
    key_to_class = {k: c for k, c, _lab in DRUM_KEYMAP}

    kits_root = kits_root or kit_store.KITS_DIR
    written: Counter = Counter()
    for kit in kit_store.list_kits(kits_root):
        if not kit["is_classify"]:
            continue                          # only drum-layout kits have labels
        inst, _is_classify = kit_store.load_kit(kit["path"])
        for key, segs in inst.notes.items():
            label = key_to_class.get(key)
            if label is None or label == "fx":
                continue
            for j, seg in enumerate(segs):
                if not dry_run:
                    out_dir = dest_root / label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{REINFORCE_PREFIX}{_safe(kit['name'])}_{key}_{j}.wav"
                    sf.write(str(out_dir / fname),
                             np.asarray(seg.audio, dtype=np.float32),
                             inst.sample_rate, subtype="PCM_24")
                written[label] += 1
    return written


def _cmd_scan() -> None:
    exs_files = sorted(DKD_EXS.rglob("*.exs"))
    print(f"Drum Kit Designer .exs files: {len(exs_files)}")
    print("Sampling label yield from the first 5 kits (per_piece=8, no toms)…")
    counts = Counter()
    for exs in exs_files[:5]:
        for hit in iter_kit_hits(exs, allow_toms=False, per_piece=8):
            counts[hit.label] += 1
    for lab, n in counts.most_common():
        print(f"  {lab:8s} {n}")
    print(f"  TOTAL   {sum(counts.values())} (from 5 kits)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--reinforce", action="store_true")
    ap.add_argument("--per", type=int, default=12,
                    help="max hits per (piece,file) per kit (default 12)")
    ap.add_argument("--limit-kits", type=int, default=None)
    ap.add_argument("--toms", action="store_true",
                    help="also mine toms (bucketed hi/mid/lo by key)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.scan:
        _cmd_scan()
    elif args.mine:
        per = None if args.per <= 0 else args.per
        counts = mine_drum_kit_designer(
            per_piece=per, limit_kits=args.limit_kits,
            allow_toms=args.toms, dry_run=args.dry_run)
        print("\nMined:", dict(counts.most_common()), "total", sum(counts.values()))
    elif args.reinforce:
        counts = reinforce_from_kits(dry_run=args.dry_run)
        print("Reinforcement:", dict(counts.most_common()), "total", sum(counts.values()))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
