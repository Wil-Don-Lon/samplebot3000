"""Unified, provenance-tagged drum-sample loader for the hybrid classifier.

Yields one `Record` per labeled one-shot with everything the training harness
needs for an HONEST grouped evaluation:

    Record(slice, hand, leaf, family, domain, group)

- `slice`  : the fixed-window serving slice (`features_from_oneshot`) — the exact
             audio the live classifier sees, so there is no train/serve skew and
             it can be CLAP-embedded downstream.
- `hand`   : the 57-dim hand feature vector for that slice.
- `leaf`   : fine label the app places on keys (kick … hats_open).
- `family` : coarse family for the cascade (see TAXONOMY).
- `domain` : "electronic" | "acoustic".
- `group`  : leave-one-*-out unit — drum machine (electronic) or kit (acoustic) —
             so near-identical sibling hits never straddle a CV fold.

Sources reuse existing loaders/labelers: `_raw` machines via
`sort_samples.classify`, Logic Drum Kit Designer via `logic_import.iter_kit_hits`,
and IDMT via `datasets_loader.load_idmt`. Freesound is unlabeled on disk here, so
its loader is a graceful no-op until an annotation file is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from pipeline import (
    load_audio, features_from_oneshot, compute_features, TARGET_SR,
)
from sort_samples import classify as name_classify
from model_analysis import LABEL_MAP, AUDIO_EXTS, RAW
import logic_import
from datasets_loader import load_idmt

# Coarse family -> its fine leaf labels. Single-leaf families are leaves at the
# coarse level (no sub-classifier). FX is never a trained class (open-set reject).
TAXONOMY: dict[str, list[str]] = {
    "kick":   ["kick"],
    "snare":  ["snare"],
    "clap":   ["clap"],
    "tom":    ["hitom", "midtom", "lotom"],
    "cymbal": ["crash", "ride"],
    "hat":    ["hats", "hats_closed", "hats_open"],
}
LEAF_TO_FAMILY: dict[str, str] = {
    leaf: fam for fam, leaves in TAXONOMY.items() for leaf in leaves
}


@dataclass
class Record:
    slice: np.ndarray        # fixed-window serving slice (float32 mono @ TARGET_SR)
    hand: np.ndarray         # 57-dim hand features for the slice
    leaf: str
    family: str
    domain: str
    group: str


def _record(slice_audio: np.ndarray, hand: np.ndarray, leaf: str,
            domain: str, group: str) -> Optional[Record]:
    fam = LEAF_TO_FAMILY.get(leaf)
    if fam is None:
        return None
    return Record(slice=slice_audio, hand=hand, leaf=leaf, family=fam,
                  domain=domain, group=group)


# ---------------------------------------------------------------------------
# Electronic: the 201 drum-machine folders under _raw (group = machine)
# ---------------------------------------------------------------------------

def load_electronic() -> Iterator[Record]:
    if not RAW.exists():
        return
    machines = [d for d in sorted(RAW.iterdir()) if d.is_dir()]
    for machine in machines:
        for f in sorted(machine.rglob("*")):
            if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
                continue
            leaf = LABEL_MAP.get(name_classify(f.stem) or "")
            if leaf is None:            # generic tom / unlabelable / fx dropped
                continue
            try:
                audio = load_audio(str(f))
            except Exception:
                continue
            result = features_from_oneshot(audio)   # same slice serving sees
            if result is None:
                continue
            rec = _record(result[0], result[1], leaf, "electronic", machine.name)
            if rec is not None:
                yield rec


# ---------------------------------------------------------------------------
# Acoustic: Logic Drum Kit Designer (group = kit) + IDMT (group = idmt:kit)
# ---------------------------------------------------------------------------

def load_logic(per_piece: Optional[int] = 6, limit_kits: Optional[int] = None,
               split_hats: bool = False, allow_toms: bool = True) -> Iterator[Record]:
    exs_files = sorted(logic_import.DKD_EXS.rglob("*.exs"))
    if limit_kits:
        exs_files = exs_files[:limit_kits]
    for exs in exs_files:
        for hit in logic_import.iter_kit_hits(
                exs, allow_toms=allow_toms, per_piece=per_piece,
                split_hats=split_hats):
            result = features_from_oneshot(hit.audio)
            if result is None:
                continue
            rec = _record(result[0], result[1], hit.label, "acoustic",
                          hit.kit or exs.stem)
            if rec is not None:
                yield rec


def load_idmt_records() -> Iterator[Record]:
    try:
        rows = load_idmt()
    except FileNotFoundError:
        return
    for audio, label, kit in rows:
        # IDMT slices already come through the fixed classify window, so featurize
        # the slice directly (no second front-trim).
        rec = _record(audio, compute_features(audio), label, "acoustic",
                      f"idmt:{kit}")
        if rec is not None:
            yield rec


def load_freesound() -> Iterator[Record]:
    """Placeholder: the on-disk Freesound one-shots have no labels here. Returns
    nothing until an annotation file is provided."""
    return
    yield  # pragma: no cover  (marks this a generator)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_records(domain: str, per_piece: Optional[int] = 6,
                 limit_kits: Optional[int] = None,
                 include_reinforcement: bool = False) -> list[Record]:
    """All records for a domain: 'electronic' | 'acoustic'."""
    records: list[Record] = []
    if domain == "electronic":
        records.extend(load_electronic())
    elif domain == "acoustic":
        records.extend(load_logic(per_piece=per_piece, limit_kits=limit_kits))
        records.extend(load_idmt_records())
        records.extend(load_freesound())
    else:
        raise ValueError(f"unknown domain {domain!r} (electronic|acoustic)")
    if include_reinforcement:
        records.extend(load_reinforcement(domain))
    return records


def load_reinforcement(domain: str) -> Iterator[Record]:
    """Saved drum-layout kits as human-verified reinforcement (group per kit)."""
    import kit_store
    from pipeline import DRUM_KEYMAP
    key_to_class = {k: c for k, c, _lab in DRUM_KEYMAP}
    for kit in kit_store.list_kits():
        if not kit["is_classify"]:
            continue
        inst, _ = kit_store.load_kit(kit["path"])
        for key, segs in inst.notes.items():
            leaf = key_to_class.get(key)
            if leaf is None or leaf == "fx":
                continue
            for seg in segs:
                result = features_from_oneshot(np.asarray(seg.audio, np.float32))
                if result is None:
                    continue
                rec = _record(result[0], result[1], leaf, domain,
                              f"kit:{kit['name']}")
                if rec is not None:
                    yield rec


def summary(records: list[Record]) -> dict:
    from collections import Counter
    return {
        "n": len(records),
        "by_leaf": dict(sorted(Counter(r.leaf for r in records).items())),
        "by_family": dict(sorted(Counter(r.family for r in records).items())),
        "groups": len({r.group for r in records}),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["electronic", "acoustic"])
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--limit-kits", type=int, default=None)
    args = ap.parse_args()
    recs = load_records(args.domain, per_piece=args.per, limit_kits=args.limit_kits)
    import json
    print(json.dumps(summary(recs), indent=2))
