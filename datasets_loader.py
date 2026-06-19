#!/usr/bin/env python3
"""Loaders for external drum datasets (brief §4).

Each loader yields uniform rows so the eval harness can hold out by kit:

    Row = (slice_audio: np.ndarray, label: str, source_kit: str)

`label` is mapped onto the app's GM-style taxonomy; `source_kit` lets
leave-one-kit-out (LOMO) hold out a whole kit, exposing real cross-kit
generalization rather than leaking near-identical sibling hits across splits.

Datasets are multi-hit recordings, so we onset-slice each file through the SAME
live pipeline (detect_onsets → extract_segments with the fixed classify window)
the app uses, keeping train/serve consistency from §1.

Currently implemented: IDMT-SMT-Drums (acoustic + synth, kick/snare/hat).
Staged for later: StemGMD (toms), ENST (acoustic, license-pending).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from pipeline import (
    load_audio, detect_onsets, extract_segments, CLASSIFY_FEATURE_LEN_S,
)

DATASETS = Path(__file__).resolve().parent / "datasets"

# ---- IDMT-SMT-Drums ----
IDMT_AUDIO = DATASETS / "idmt_smt_drums" / "audio"
IDMT_LABEL = {"KD": "kick", "SD": "snare", "HH": "hats"}


def _idmt_parse(stem: str) -> tuple[str | None, str | None]:
    """'WaveDrum02_51#KD#train' -> (label, source_kit). MIX/unknown -> (None,None)."""
    parts = stem.split("#")
    if len(parts) < 2:
        return None, None
    label = IDMT_LABEL.get(parts[1])
    if label is None:
        return None, None
    kit = parts[0].split("_")[0]  # WaveDrum02_51 -> WaveDrum02
    return label, kit


def load_idmt() -> Iterator[tuple[np.ndarray, str, str]]:
    """Onset-slice each single-instrument IDMT file into labeled one-shots."""
    if not IDMT_AUDIO.exists():
        raise FileNotFoundError(f"IDMT not found at {IDMT_AUDIO}")
    for f in sorted(IDMT_AUDIO.glob("*.wav")):
        label, kit = _idmt_parse(f.stem)
        if label is None:  # skip #MIX and anything unparseable
            continue
        try:
            audio = load_audio(str(f))
        except Exception:
            continue
        onsets = detect_onsets(audio)
        if len(onsets) == 0:
            continue
        # Same fixed classify window as serving, so features are comparable.
        segs = extract_segments(
            audio, onsets, feature_length_s=CLASSIFY_FEATURE_LEN_S)
        for s in segs:
            yield s.audio, label, kit


LOADERS = {"idmt": load_idmt}


if __name__ == "__main__":
    from collections import Counter
    rows = list(load_idmt())
    print(f"IDMT: {len(rows)} slices")
    print("  by label:", dict(Counter(r[1] for r in rows)))
    print("  by kit  :", dict(Counter(r[2] for r in rows)))
