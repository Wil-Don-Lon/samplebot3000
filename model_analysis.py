#!/usr/bin/env python3
"""Honest model analysis for the drum classifier.

Two questions:
  1. Is our cross-val accuracy real, or leaked? We answer by comparing a random
     StratifiedKFold against a leave-one-machine-out GroupKFold. Near-identical
     siblings from the same drum machine (velocity layers, round-robins) landing
     in both train and test inflate random CV. GroupKFold = true "unseen gear".
  2. Train/serve skew: training features come from whole trimmed one-shots;
     inference features come from <=0.5s onset-aligned (often truncated)
     segments. We report one-shot duration stats to size that gap.

Features and labels are derived exactly as the training set was: same
compute_features, same name-based classify(). Group = the drum-machine folder
under _raw/drum samples/.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import load_audio, trim_silence, compute_features, TARGET_SR
from sort_samples import classify as name_classify
from classifier import make_model, MODEL_NAMES

RAW = Path(__file__).resolve().parent / "training data" / "_raw" / "drum samples"
CACHE = Path(__file__).resolve().parent / "analysis_cache.npz"

# Map name-classifier output -> training bin label. Generic "tom" is dropped
# (training excluded it); everything else mirrors the bins.
LABEL_MAP = {
    "kick": "kick", "snare": "snare", "clap": "clap", "hat": "hats",
    "ride": "ride", "crash": "crash", "hitom": "hitom", "midtom": "midtom",
    "lotom": "lotom", "fx": "fx",
}
AUDIO_EXTS = {".wav", ".aiff", ".aif", ".mp3", ".snd"}


def extract() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y, groups, durs = [], [], [], []
    machines = [d for d in sorted(RAW.iterdir()) if d.is_dir()]
    for mi, machine in enumerate(machines):
        group = machine.name
        n = 0
        for f in sorted(machine.rglob("*")):
            if not (f.is_file() and f.suffix.lower() in AUDIO_EXTS):
                continue
            label = LABEL_MAP.get(name_classify(f.stem) or "")
            if label is None:
                continue
            try:
                audio = load_audio(str(f))
            except Exception:
                continue
            audio = trim_silence(audio)
            if audio.size < 256:
                continue
            X.append(compute_features(audio))
            y.append(label)
            groups.append(group)
            durs.append(audio.size / float(TARGET_SR))
            n += 1
        print(f"  [{mi+1:3d}/{len(machines)}] {group[:34]:34s} {n:4d}")
    return (np.asarray(X, np.float32), np.asarray(y),
            np.asarray(groups), np.asarray(durs, np.float32))


def get_data():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["y"], d["groups"], d["durs"]
    print("Extracting from _raw (grouped by machine)…")
    X, y, groups, durs = extract()
    np.savez_compressed(CACHE, X=X, y=y,
                        groups=np.array(groups, dtype=object),
                        durs=durs)
    print(f"\nCached {len(y)} samples -> {CACHE.name}")
    return X, y, groups, durs


def analyze() -> None:
    from sklearn.model_selection import (
        StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict)
    from sklearn.metrics import classification_report, confusion_matrix

    X, y, groups, durs = get_data()
    classes = sorted(set(y))
    print("\n=== DATASET ===")
    print(f"  samples : {len(y)}")
    print(f"  machines: {len(set(groups))}")
    print(f"  classes : {dict(sorted(Counter(y).items()))}")

    # ---- train/serve skew: one-shot duration vs the 0.5s inference cap ----
    pcts = np.percentile(durs, [50, 75, 90, 99])
    over = float((durs > 0.5).mean()) * 100
    print("\n=== TRAIN/SERVE SKEW (one-shot trimmed duration) ===")
    print(f"  median {pcts[0]:.2f}s · p75 {pcts[1]:.2f}s · p90 {pcts[2]:.2f}s · p99 {pcts[3]:.2f}s")
    print(f"  {over:.0f}% of training one-shots are LONGER than the 0.5s segment cap used at inference")

    # ---- leakage: random CV vs leave-one-machine-out CV ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gkf = GroupKFold(n_splits=5)
    print("\n=== ACCURACY: random split vs unseen-machine split ===")
    print(f"  {'model':4s}  {'random(SKF)':>12s}  {'unseen(GKF)':>12s}  {'gap':>7s}")
    rows = []
    for name in MODEL_NAMES:
        try:
            rnd = cross_val_score(make_model(name), X, y, cv=skf, scoring="accuracy").mean()
            grp = cross_val_score(make_model(name), X, y, cv=gkf, groups=groups,
                                  scoring="accuracy").mean()
        except Exception as e:
            print(f"  {name:4s}  FAILED: {type(e).__name__}: {e}")
            continue
        rows.append((name, rnd, grp))
        print(f"  {name:4s}  {rnd:12.3f}  {grp:12.3f}  {rnd-grp:+7.3f}")

    if not rows:
        return
    best = max(rows, key=lambda r: r[2])[0]
    print(f"\nBest on unseen gear: {best}")

    # ---- per-class truth on the unseen-machine split ----
    print(f"\n=== {best}: per-class report on UNSEEN machines (GroupKFold OOF) ===")
    pred = cross_val_predict(make_model(best), X, y, cv=gkf, groups=groups)
    print(classification_report(y, pred, zero_division=0))
    cm = confusion_matrix(y, pred, labels=classes)
    print("Confusion (rows=true, cols=pred):")
    print("        " + " ".join(f"{c[:5]:>5s}" for c in classes))
    for c, r in zip(classes, cm):
        print(f"  {c[:6]:6s} " + " ".join(f"{v:5d}" for v in r))


if __name__ == "__main__":
    if "--extract" in sys.argv:
        get_data()
    else:
        analyze()
