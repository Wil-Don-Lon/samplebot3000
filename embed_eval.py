#!/usr/bin/env python3
"""Frozen-embedding evaluation harness for Samplebot-3000 (brief §2).

Mirrors model_analysis.py's HONEST protocol — leave-one-machine-out (GroupKFold)
macro-F1, grouped by source drum machine — but replaces the hand-crafted 57-dim
vector with a frozen pretrained audio embedding (default: CLAP). The sklearn
heads are unchanged (classifier.make_model), so any delta vs the §1 baseline is
purely the representation.

Each sample is embedded from the SAME de-skewed 0.5s slice the live classifier
sees (pipeline.features_from_oneshot), so we don't reintroduce train/serve skew.

Embeddings are slow, so they're cached to embed_cache_<backend>.npz.

Usage:
  python embed_eval.py --backend clap --extract   # build/refresh the cache
  python embed_eval.py --backend clap             # LOMO macro-F1 + per-class
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import load_audio, features_from_oneshot, pitch_temporal_features
from sort_samples import classify as name_classify
from model_analysis import RAW, LABEL_MAP, AUDIO_EXTS
from classifier import make_model
from embedders import get_embedder

HEADS = ["svm", "gb"]  # brief's named heads; strong + probabilistic


def cache_path(backend: str) -> Path:
    return Path(__file__).resolve().parent / f"embed_cache_{backend}.npz"


def extract(backend: str, batch_size: int = 32):
    """Embed every grouped one-shot from _raw, grouped by machine for LOMO."""
    emb = get_embedder(backend)
    machines = [d for d in sorted(RAW.iterdir()) if d.is_dir()]

    slices, scalars, y, groups = [], [], [], []
    for mi, machine in enumerate(machines):
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
            result = features_from_oneshot(audio)  # same slice the model sees
            if result is None:
                continue
            slices.append(result[0])
            scalars.append(pitch_temporal_features(result[0]))  # §3 pitch/temporal
            y.append(label)
            groups.append(machine.name)
            n += 1
        print(f"  [{mi+1:3d}/{len(machines)}] {machine.name[:34]:34s} {n:4d}")

    # Batch-embed all collected slices (the slow part).
    print(f"\nEmbedding {len(slices)} slices via {backend} (batch {batch_size})…")
    X = np.empty((len(slices), emb.dim), dtype=np.float32)
    for i in range(0, len(slices), batch_size):
        X[i:i + batch_size] = emb.embed_batch(slices[i:i + batch_size], 44100)
        if (i // batch_size) % 10 == 0:
            print(f"  {i + min(batch_size, len(slices) - i)}/{len(slices)}")
    S = np.stack(scalars).astype(np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    np.savez_compressed(cache_path(backend), X=X, S=S, y=y,
                        groups=np.array(groups, dtype=object))
    print(f"Cached {len(y)} embeddings × {X.shape[1]} dims (+{S.shape[1]} scalars) "
          f"-> {cache_path(backend).name}")
    return X, S, y, groups


def get_data(backend: str):
    cp = cache_path(backend)
    if cp.exists():
        d = np.load(cp, allow_pickle=True)
        if "S" in d:
            return d["X"], d["S"], d["y"], d["groups"]
        print("Cache lacks pitch scalars — re-extracting…")
    else:
        print(f"No embedding cache for {backend}. Extracting…")
    return extract(backend)


def _select(X, S, mode: str):
    """Pick the feature matrix for an eval mode: emb | scalars | fused."""
    if mode == "emb":
        return X
    if mode == "scalars":
        return S
    if mode == "fused":
        return np.hstack([X, S])
    raise ValueError(f"unknown features mode {mode!r}")


def analyze(backend: str, mode: str = "fused"):
    from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix

    X, S, y, groups = get_data(backend)
    Xm = _select(X, S, mode)
    classes = sorted(set(y))
    print(f"\n=== {backend.upper()} · features={mode} ({Xm.shape[1]}-dim) ===")
    print(f"  samples : {len(y)}")
    print(f"  machines: {len(set(groups))}")
    print(f"  classes : {dict(sorted(Counter(y).items()))}")

    gkf = GroupKFold(n_splits=5)
    print("\n=== LOMO macro-F1 (unseen-machine GroupKFold) ===")
    print(f"  baseline 57-dim MFCC gb: macro-F1 0.69 / acc 0.81")
    print(f"  CLAP emb-only      gb: macro-F1 0.74 / acc 0.86")
    best, best_f1 = None, -1.0
    for head in HEADS:
        f1 = cross_val_score(make_model(head), Xm, y, cv=gkf, groups=groups,
                             scoring="f1_macro").mean()
        acc = cross_val_score(make_model(head), Xm, y, cv=gkf, groups=groups,
                              scoring="accuracy").mean()
        print(f"  {head:4s}  macro-F1 {f1:.3f}  acc {acc:.3f}")
        if f1 > best_f1:
            best, best_f1 = head, f1

    print(f"\nBest head: {best} (macro-F1 {best_f1:.3f})")
    print(f"\n=== {best}: per-class on UNSEEN machines (GroupKFold OOF) ===")
    pred = cross_val_predict(make_model(best), Xm, y, cv=gkf, groups=groups)
    print(classification_report(y, pred, zero_division=0))
    cm = confusion_matrix(y, pred, labels=classes)
    print("Confusion (rows=true, cols=pred):")
    print("        " + " ".join(f"{c[:5]:>5s}" for c in classes))
    for c, r in zip(classes, cm):
        print(f"  {c[:6]:6s} " + " ".join(f"{v:5d}" for v in r))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="clap")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--features", default="fused",
                    choices=["emb", "scalars", "fused"])
    args = ap.parse_args()
    if args.extract:
        extract(args.backend)
    else:
        analyze(args.backend, args.features)
    return 0


if __name__ == "__main__":
    sys.exit(main())
