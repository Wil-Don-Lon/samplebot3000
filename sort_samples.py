#!/usr/bin/env python3
"""Name-based sample sorter for Samplebot-3000 training data.

Classifies drum-sample files by tokens in their filename and moves them into
the canonical bins, renaming each to a simple <type><n>.<ext> form
(kick1.wav, snare134.wav, hitom7.wav, ...). Files whose names carry no clear
type token are left in place for the supervised-clustering pass.

The sorter is incremental and idempotent: files already sitting in the correct
bin with a conformant name are left untouched (their numbers are reserved), and
only new or misfiled files get appended with the next free number per type.

Run with --apply to actually move/rename; default is a dry run that reports.
Each --apply run writes a manifest CSV (original path -> new path) for
provenance, so we never lose track of where a sample came from again.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "training data"
UNSORTED = ROOT / "unsorted"
RAW = ROOT / "_raw"  # full original backup; never sorted from / into

AUDIO_EXTS = {".wav", ".aiff", ".aif", ".mp3", ".snd"}

# type -> destination bin folder name
BIN = {
    "kick": "kick",
    "snare": "snare",
    "clap": "clap",
    "hat": "hats",
    "ride": "ride",
    "crash": "crash",
    "tom": "toms",
    "hitom": "hitom",
    "midtom": "midtom",
    "lotom": "lotom",
    "fx": "fx",
}

# bin folder -> the canonical type whose name files in it should carry. Used to
# detect already-conformant files (and reserve their numbers) on re-runs.
FOLDER_TYPE = {folder: t for t, folder in BIN.items()}

# Ordered (type, regex). First match on the lowercased stem wins, so higher
# priority tokens come first. snare beats clap so a "Clap_Snare" hit lands in
# snare. Patterns are deliberately boundary-light so tokens glued to digits or
# underscores (HR16Hat_C1, RX21Tom_Lo, DM5FX59) still match.
RULES: list[tuple[str, re.Pattern]] = [
    ("kick",  re.compile(r"kick|kik|bass\s*drum|bassdrum|bdrum|\bb\.?d\b|909bd|\bkk\b")),
    ("snare", re.compile(r"snare|snr|\bsn\d")),
    ("clap",  re.compile(r"clap|\bclp|cla\d|\bcla\b|\bcld|handclap")),
    ("hat",   re.compile(r"hi[\s_-]?hat|hihat|hat|o?hh|chh")),
    ("ride",  re.compile(r"ride")),
    ("crash", re.compile(r"crash|crsh")),
    ("tom",   re.compile(r"tom")),
    ("fx",    re.compile(r"\bfx|fx\d|\dfx|sfx|effect")),
]

# Pitch refinement for toms, checked high -> mid -> low; default generic tom.
# Abbreviations (htom/mtom/ltom) are anchored to a non-letter/start so we don't
# catch the m in "sym tom" or the like.
_TOM_HIGH = re.compile(r"high|hi[\s_-]?tom|tom[\s_-]?hi|hitom|tomhi|(?:^|[^a-z])htom")
_TOM_MID = re.compile(r"mid|mi[\s_-]?tom|tom[\s_-]?mi|midtom|tommid|(?:^|[^a-z])m[d]?tom")
_TOM_LOW = re.compile(r"low|lo[\s_-]?tom|tom[\s_-]?lo|lotom|tomlo|(?:^|[^a-z])ltom|floor|flr")

_CONFORM = {t: re.compile(rf"^{t}\d+$") for t in BIN}


def _refine_tom(s: str) -> str:
    # Tokenise to catch pitch suffixes that trail the tom number, e.g.
    # DM5Tom01_Hi -> {"hi"}, DM5Tom01_M -> {"m"} (M = mid in DM5 kits).
    toks = set(re.split(r"[^a-z0-9]+", s))
    if _TOM_HIGH.search(s) or toks & {"hi", "high"}:
        return "hitom"
    if _TOM_MID.search(s) or toks & {"mid", "mi", "md"}:
        return "midtom"
    if _TOM_LOW.search(s) or toks & {"lo", "low"}:
        return "lotom"
    # Single-letter pitch suffix, only meaningful inside a tom name.
    if "h" in toks:
        return "hitom"
    if "m" in toks:
        return "midtom"
    if "l" in toks:
        return "lotom"
    return "tom"


def classify(name: str) -> str | None:
    """Return the detected type for a filename stem, or None if unknown."""
    s = name.lower()
    for t, pat in RULES:
        if pat.search(s):
            if t == "hat" and "orch" in s:
                continue  # orchestra hit, not a hi-hat
            return _refine_tom(s) if t == "tom" else t
    return None


def is_conformant(f: Path) -> bool:
    """True if f already sits in its correct bin with a canonical name."""
    folder = f.parent.name
    t = FOLDER_TYPE.get(folder)
    return t is not None and bool(_CONFORM[t].match(f.stem))


def scan() -> tuple[dict[str, int], list[tuple[Path, str]]]:
    """Return (max_used_number_per_type, [(src, type)] needing placement).

    Conformant files in the right bin are skipped but their numbers reserved.
    Non-conformant bin files and classifiable unsorted files are queued for
    (re)placement.
    """
    max_used: dict[str, int] = {t: 0 for t in BIN}
    todo: list[tuple[Path, str]] = []

    # Existing bins.
    for folder, t in FOLDER_TYPE.items():
        d = ROOT / folder
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*")):
            if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
                continue
            m = _CONFORM[t].match(f.stem)
            if m:
                max_used[t] = max(max_used[t], int(f.stem[len(t):]))
            else:
                dt = classify(f.stem) or t
                todo.append((f, dt))

    # Unsorted -> classify by name.
    for f in sorted(UNSORTED.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            t = classify(f.stem)
            if t is not None:
                todo.append((f, t))

    return max_used, todo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move/rename")
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()

    max_used, todo = scan()

    counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    for src, t in todo:
        counts[t] = counts.get(t, 0) + 1
        examples.setdefault(t, [])
        if len(examples[t]) < args.examples:
            examples[t].append(src.name)

    total_audio = sum(
        1 for f in UNSORTED.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    )

    print("=== SORT REPORT (new/changed placements) ===")
    for t in sorted(counts):
        print(f"  +{counts[t]:5d} -> {BIN[t]:8s}  (existing max {t}{max_used[t]})")
    print(f"\n  unsorted audio files : {total_audio}")
    print(f"  newly placeable      : {len(todo)}")
    print(f"  left in unsorted     : {total_audio - len(todo)}")
    print()
    for t in sorted(examples):
        print(f"  [{t}] e.g. " + ", ".join(examples[t]))
    print()

    if not args.apply:
        print("DRY RUN — nothing moved. Re-run with --apply to execute.")
        return 0

    for t in BIN:
        (ROOT / BIN[t]).mkdir(parents=True, exist_ok=True)

    counter = dict(max_used)
    rows = []
    for src, t in sorted(todo, key=lambda x: (x[1], str(x[0]))):
        counter[t] += 1
        ext = src.suffix.lower()
        dest = ROOT / BIN[t] / f"{t}{counter[t]}{ext}"
        while dest.exists() and dest != src:
            counter[t] += 1
            dest = ROOT / BIN[t] / f"{t}{counter[t]}{ext}"
        rows.append((str(src.relative_to(ROOT)), str(dest.relative_to(ROOT)), t))
        src.replace(dest)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = ROOT.parent / f"sort_manifest_{ts}.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["original", "new", "type"])
        w.writerows(rows)
    print(f"APPLIED — moved/renamed {len(rows)} files.")
    print(f"Manifest: {manifest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
