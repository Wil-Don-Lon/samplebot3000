#!/usr/bin/env python3
"""Normalize sample filenames to match the bin folder they live in.

After manual reorganization (moving a sample into a different bin without
renaming), a file may carry a name from its old type, e.g. snare86.wav sitting
in clap/. This renames every such "foreign" file to <foldertype><n>.<ext>,
honouring its current folder (the manual placement is treated as correct).

Files already conformant to their folder keep their name and number; foreign
files get the next free number per folder, so nothing else is renumbered.

Run with --apply to execute; default is a dry run. Writes a manifest CSV.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime

from sort_samples import ROOT, AUDIO_EXTS, FOLDER_TYPE


def plan() -> list[tuple]:
    """Return [(src, dest, type)] of foreign files to rename."""
    moves = []
    for folder, t in FOLDER_TYPE.items():
        d = ROOT / folder
        if not d.is_dir():
            continue
        pat = re.compile(rf"^{t}\d+$")
        files = [f for f in sorted(d.iterdir())
                 if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
        used = max((int(f.stem[len(t):]) for f in files if pat.match(f.stem)),
                   default=0)
        for f in files:
            if pat.match(f.stem):
                continue
            used += 1
            dest = d / f"{t}{used}{f.suffix.lower()}"
            while dest.exists() and dest != f:
                used += 1
                dest = d / f"{t}{used}{f.suffix.lower()}"
            moves.append((f, dest, t))
    return moves


def repack_plan() -> list[tuple]:
    """Return [(src, dest, type)] to renumber each folder contiguously 1..N.

    Files are ordered by their current number so relative order is preserved;
    only gaps are closed. Assumes files are already conformant (run normalize
    first). Entries where src == dest are omitted.
    """
    moves = []
    for folder, t in FOLDER_TYPE.items():
        d = ROOT / folder
        if not d.is_dir():
            continue
        pat = re.compile(rf"^{t}(\d+)$")
        files = []
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                m = pat.match(f.stem)
                if m:
                    files.append((int(m.group(1)), f))
        files.sort(key=lambda x: x[0])
        for i, (_, f) in enumerate(files, start=1):
            dest = d / f"{t}{i}{f.suffix.lower()}"
            if dest != f:
                moves.append((f, dest, t))
    return moves


def apply_repack(moves: list[tuple]) -> list[tuple]:
    """Two-phase rename (via temp names) so renumbering can't collide."""
    rows = []
    staged = []
    for n, (src, dest, t) in enumerate(moves):
        tmp = src.with_name(f"__repack_{n}__{dest.name}")
        src.replace(tmp)
        staged.append((tmp, dest, t, src))
    for tmp, dest, t, orig in staged:
        tmp.replace(dest)
        rows.append((str(orig.relative_to(ROOT)), str(dest.relative_to(ROOT)), t))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--repack", action="store_true",
                    help="renumber each folder contiguously 1..N")
    args = ap.parse_args()

    if args.repack:
        moves = repack_plan()
        print("=== REPACK REPORT ===")
        if not moves:
            print("  Nothing to do — every folder already numbered 1..N.")
            return 0
        by_folder: dict[str, int] = {}
        for src, _, _ in moves:
            by_folder[src.parent.name] = by_folder.get(src.parent.name, 0) + 1
        for folder in sorted(by_folder):
            print(f"  {folder}: {by_folder[folder]} file(s) renumbered")
        print()
        if not args.apply:
            for src, dest, _ in moves[:12]:
                print(f"  {src.parent.name}/{src.name}  ->  {dest.name}")
            if len(moves) > 12:
                print(f"  ... and {len(moves) - 12} more")
            print("\nDRY RUN — nothing renamed. Re-run with --apply to execute.")
            return 0
        rows = apply_repack(moves)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest = ROOT.parent / f"repack_manifest_{ts}.csv"
        with manifest.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["original", "new", "type"])
            w.writerows(rows)
        print(f"APPLIED — renumbered {len(rows)} files.")
        print(f"Manifest: {manifest.name}")
        return 0

    moves = plan()
    by_folder: dict[str, int] = {}
    for src, _, _ in moves:
        by_folder[src.parent.name] = by_folder.get(src.parent.name, 0) + 1

    print("=== NORMALIZE REPORT ===")
    if not moves:
        print("  Nothing to do — every file already matches its folder.")
        return 0
    for folder in sorted(by_folder):
        print(f"  {folder}: {by_folder[folder]} file(s) to rename")
    print()
    for src, dest, _ in moves[:20]:
        print(f"  {src.parent.name}/{src.name}  ->  {dest.name}")
    if len(moves) > 20:
        print(f"  ... and {len(moves) - 20} more")
    print()

    if not args.apply:
        print("DRY RUN — nothing renamed. Re-run with --apply to execute.")
        return 0

    rows = []
    for src, dest, t in moves:
        rows.append((str(src.relative_to(ROOT)), str(dest.relative_to(ROOT)), t))
        src.replace(dest)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = ROOT.parent / f"normalize_manifest_{ts}.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["original", "new", "type"])
        w.writerows(rows)
    print(f"APPLIED — renamed {len(rows)} files.")
    print(f"Manifest: {manifest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
