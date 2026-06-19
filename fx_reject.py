#!/usr/bin/env python3
"""FX as open-set rejection for Samplebot-3000 (brief §5).

FX is not a coherent class — it's "none of the above". So instead of training a
10th class, we train on the 9 REAL drum types and reject a slice to FX when the
model isn't confident it's any real type. Two rejection paths:

  calibrated : fused CLAP+pitch features -> CalibratedClassifierCV (isotonic), so
               max class probability is meaningful; reject if max_prob < tau.
  zeroshot   : CLAP audio embedding vs per-class text prompts (cosine); reject if
               max_similarity < tau. No training; the similarity IS the confidence.

Everything is honest leave-one-machine-out: FX slices are NEVER in training, and
real-class folds hold out whole machines. We sweep tau and report, at the tau
that maximizes 10-class macro-F1:
  - FX precision/recall (did weird sounds get rejected, separately reported), and
  - that real drums are not over-rejected (their macro-F1 holds up).

Baseline to beat: fx-as-a-class gave FX F1 0.48 in embed_eval (fused).

Usage:
  python fx_reject.py --method calibrated
  python fx_reject.py --method zeroshot
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from embed_eval import get_data
from classifier import make_model

REAL = ["kick", "snare", "clap", "hats", "ride", "crash", "hitom", "midtom", "lotom"]

# Short natural-language prompts per real class for the CLAP zero-shot path.
PROMPTS = {
    "kick": "a kick drum", "snare": "a snare drum", "clap": "a hand clap",
    "hats": "a hi-hat cymbal", "ride": "a ride cymbal", "crash": "a crash cymbal",
    "hitom": "a high tom drum", "midtom": "a mid tom drum",
    "lotom": "a low floor tom drum",
}


def _report(true, pred, taus, maxconf, classes):
    """Sweep tau, pick the one maximizing 10-class macro-F1, and print details."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    best = None
    for tau in taus:
        p = pred.copy()
        p[maxconf < tau] = "fx"
        f1 = f1_score(true, p, labels=classes, average="macro", zero_division=0)
        if best is None or f1 > best[1]:
            best = (tau, f1, p.copy())
    tau, f1, p = best

    fx_p = precision_score(true == "fx", p == "fx", zero_division=0)
    fx_r = recall_score(true == "fx", p == "fx", zero_division=0)
    real = true != "fx"
    real_f1 = f1_score(true[real], p[real],
                       labels=REAL, average="macro", zero_division=0)
    print(f"\nBest tau = {tau:.3f}")
    print(f"  10-class macro-F1 : {f1:.3f}   (fx-as-class baseline: 0.75 overall, fx F1 0.48)")
    print(f"  FX precision      : {fx_p:.3f}")
    print(f"  FX recall         : {fx_r:.3f}")
    print(f"  FX F1             : {2*fx_p*fx_r/max(fx_p+fx_r,1e-9):.3f}")
    print(f"  real-class macro-F1 (not over-rejected?): {real_f1:.3f}")
    return tau


def run_calibrated():
    from sklearn.model_selection import GroupKFold
    from sklearn.calibration import CalibratedClassifierCV

    X, S, y, groups = get_data("clap")
    Xf = np.hstack([X, S])  # fused CLAP + pitch/temporal (our best representation)
    classes = sorted(set(y))

    gkf = GroupKFold(n_splits=5)
    true = np.empty(len(y), dtype=object)
    pred = np.empty(len(y), dtype=object)
    maxconf = np.zeros(len(y))

    for tr, te in gkf.split(Xf, y, groups):
        real_tr = tr[y[tr] != "fx"]  # train on real classes only
        clf = CalibratedClassifierCV(make_model("gb"), method="isotonic", cv=3)
        clf.fit(Xf[real_tr], y[real_tr])
        proba = clf.predict_proba(Xf[te])
        cls = clf.classes_
        amax = proba.argmax(axis=1)
        true[te] = y[te]
        pred[te] = cls[amax]
        maxconf[te] = proba.max(axis=1)

    print("=== FX rejection · CALIBRATED (fused features, isotonic) ===")
    _report(true, pred, np.linspace(0.2, 0.8, 25), maxconf, classes)


def run_zeroshot():
    from embedders import get_embedder

    X, S, y, groups = get_data("clap")  # X = L2-normed CLAP audio embeddings
    classes = sorted(set(y))

    emb = get_embedder("clap")
    text = emb.embed_text([PROMPTS[c] for c in REAL])  # (9, 512) L2-normed
    sims = X @ text.T                                   # cosine per real class
    amax = sims.argmax(axis=1)
    pred = np.array([REAL[i] for i in amax], dtype=object)
    maxconf = sims.max(axis=1)
    true = np.asarray(y, dtype=object)

    print("=== FX rejection · CLAP ZERO-SHOT (text-prompt cosine, no training) ===")
    print(f"  cosine range: min {maxconf.min():.3f} / median {np.median(maxconf):.3f} / max {maxconf.max():.3f}")
    _report(true, pred, np.linspace(0.15, 0.6, 25), maxconf, classes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="calibrated",
                    choices=["calibrated", "zeroshot"])
    args = ap.parse_args()
    if args.method == "calibrated":
        run_calibrated()
    else:
        run_zeroshot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
