#!/usr/bin/env python3
"""Acoustic-data experiments for Samplebot-3000 (brief §4).

Uses the IDMT-SMT-Drums acoustic/synth kits (kick/snare/hat) to answer two
honest questions about our current best model (fused CLAP + pitch, gb):

  transfer : train on the drum-machine data, test on held-out IDMT acoustic kits.
             Measures the synth->acoustic domain gap directly. (No leakage: IDMT
             is 100% test.)
  augment  : de-dup IDMT (CLAP cosine) and add it to training, LOMO by source_kit
             on the kick/snare/hat subset. Does acoustic data improve unseen-kit
             generalization vs drum-machine-only?

Embeddings are cached to embed_cache_idmt.npz.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from datasets_loader import load_idmt
from embed_eval import get_data as get_machine_data
from embedders import get_embedder
from pipeline import pitch_temporal_features
from classifier import make_model

KSH = ["kick", "snare", "hats"]  # classes IDMT provides
IDMT_CACHE = "embed_cache_idmt.npz"


def extract_idmt(batch_size: int = 64):
    from pathlib import Path
    emb = get_embedder("clap")
    slices, y, kits = [], [], []
    for audio, label, kit in load_idmt():
        slices.append(audio)
        y.append(label)
        kits.append(kit)
    print(f"Embedding {len(slices)} IDMT slices…")
    X = np.empty((len(slices), emb.dim), dtype=np.float32)
    for i in range(0, len(slices), batch_size):
        X[i:i + batch_size] = emb.embed_batch(slices[i:i + batch_size], 44100)
        if (i // batch_size) % 10 == 0:
            print(f"  {min(i + batch_size, len(slices))}/{len(slices)}")
    S = np.stack([pitch_temporal_features(a) for a in slices]).astype(np.float32)
    np.savez_compressed(
        Path(__file__).resolve().parent / IDMT_CACHE,
        X=X, S=S, y=np.asarray(y), kits=np.array(kits, dtype=object))
    print(f"Cached {len(y)} IDMT embeddings -> {IDMT_CACHE}")
    return X, S, np.asarray(y), np.asarray(kits)


def get_idmt():
    from pathlib import Path
    cp = Path(__file__).resolve().parent / IDMT_CACHE
    if cp.exists():
        d = np.load(cp, allow_pickle=True)
        return d["X"], d["S"], d["y"], d["kits"]
    return extract_idmt()


def _dedup(X, threshold=0.9995, labels=None, kits=None):
    """Greedy near-duplicate removal by CLAP cosine (X is L2-normed).

    Only compares within the same (kit,label) bucket so we drop repeated near-
    identical sibling hits, not genuinely distinct sounds. Returns a keep-mask.
    """
    keep = np.ones(len(X), dtype=bool)
    buckets: dict = {}
    for i in range(len(X)):
        key = (kits[i], labels[i]) if labels is not None else 0
        kept = buckets.setdefault(key, [])
        if any(float(X[i] @ X[j]) > threshold for j in kept):
            keep[i] = False
        else:
            kept.append(i)
    return keep


def transfer():
    from sklearn.metrics import classification_report, confusion_matrix

    Xm, Sm, ym, _ = get_machine_data("clap")
    Xi, Si, yi, kits = get_idmt()
    classes = sorted(set(ym))

    # Train fused gb on ALL drum-machine classes; test on IDMT (k/s/h truth).
    clf = make_model("gb")
    clf.fit(np.hstack([Xm, Sm]), ym)
    pred = clf.predict(np.hstack([Xi, Si]))

    print("=== SYNTH->ACOUSTIC TRANSFER (train drum-machine, test IDMT) ===")
    print(f"  IDMT slices: {len(yi)}  {dict(Counter(yi))}")
    print(classification_report(yi, pred, labels=KSH, zero_division=0))
    cm = confusion_matrix(yi, pred, labels=classes)
    print("Where IDMT k/s/h slices land (rows=true k/s/h, cols=all pred):")
    print("        " + " ".join(f"{c[:5]:>5s}" for c in classes))
    for c, r in zip(classes, cm):
        if c in KSH:
            print(f"  {c[:6]:6s} " + " ".join(f"{v:5d}" for v in r))


def augment():
    from sklearn.model_selection import GroupKFold, cross_val_score

    Xm, Sm, ym, gm = get_machine_data("clap")
    Xi, Si, yi, gi = get_idmt()

    # Restrict drum-machine data to k/s/h for a fair like-for-like comparison.
    mksh = np.isin(ym, KSH)
    Xm, Sm, ym, gm = Xm[mksh], Sm[mksh], ym[mksh], gm[mksh]

    keep = _dedup(Xi, labels=yi, kits=gi)
    print(f"IDMT de-dup: kept {keep.sum()}/{len(keep)} slices "
          f"(dropped {len(keep)-keep.sum()} near-duplicates)")
    Xi, Si, yi, gi = Xi[keep], Si[keep], yi[keep], gi[keep]
    gi = np.array([f"idmt:{k}" for k in gi], dtype=object)  # distinct kit groups

    def lomo(X, S, y, g):
        return cross_val_score(make_model("gb"), np.hstack([X, S]), y,
                               cv=GroupKFold(5), groups=g,
                               scoring="f1_macro").mean()

    base = lomo(Xm, Sm, ym, gm)
    Xc = np.vstack([Xm, Xi]); Sc = np.vstack([Sm, Si])
    yc = np.concatenate([ym, yi]); gc = np.concatenate([gm, gi])
    comb = lomo(Xc, Sc, yc, gc)
    print("\n=== DOES ACOUSTIC DATA HELP? (LOMO macro-F1, k/s/h) ===")
    print(f"  drum-machine only      : {base:.3f}")
    print(f"  drum-machine + IDMT    : {comb:.3f}   (Δ {comb-base:+.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "transfer", "augment"])
    args = ap.parse_args()
    {"extract": extract_idmt, "transfer": transfer, "augment": augment}[args.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
