#!/usr/bin/env python3
"""Bake-off + trainer for the custom hybrid drum classifier (one per domain).

Pipeline:
  1. Cache features per domain: hand-57 + fused CLAP (512 ⧺ 5 pitch/temporal),
     with leaf/family labels and leave-one-group-out groups.
  2. Bake-off: for every cascade node and every flat specialist, pick the
     (estimator × representation) that maximizes grouped-CV macro-F1.
  3. Assemble BOTH architectures (cascade, flat) from those choices.
  4. Compare honestly under leave-one-group-out GroupKFold (out-of-fold macro-F1
     + per-class precision), ship the winner.
  5. Refit the winner on all domain data -> models/hybrid_<domain>.joblib.

Honest grouping: electronic holds out whole drum machines, acoustic holds out
whole kits — so near-identical sibling hits never leak across folds.

Usage:
  python train_hybrid.py --domain electronic            # full
  python train_hybrid.py --domain acoustic --quick      # fast (hand-only)
  python train_hybrid.py --domain electronic --rebuild-cache
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import drum_data
from drum_data import TAXONOMY, LEAF_TO_FAMILY
from classifier import make_model
from hybrid_classifier import Node, CascadeClassifier, FlatSpecialistClassifier

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"

ALL_HEADS = ["gb", "svm", "rf", "knn", "mlp"]
QUICK_HEADS = ["gb", "svm"]


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

def cache_path(domain: str) -> Path:
    return HERE / f"hybrid_cache_{domain}.npz"


def build_cache(domain: str, per_piece: int, limit_kits, want_fused: bool) -> dict:
    print(f"Loading {domain} records…")
    recs = drum_data.load_records(domain, per_piece=per_piece, limit_kits=limit_kits)
    print("  " + str(drum_data.summary(recs)))
    hand = np.stack([r.hand for r in recs]).astype(np.float32)
    y_leaf = np.array([r.leaf for r in recs])
    y_family = np.array([r.family for r in recs])
    groups = np.array([r.group for r in recs])

    fused = None
    if want_fused:
        from embedders import get_embedder, embedder_available
        if embedder_available():
            from pipeline import pitch_temporal_features, TARGET_SR
            emb = get_embedder("clap")
            slices = [r.slice for r in recs]
            print(f"CLAP-embedding {len(slices)} slices…")
            E = np.empty((len(slices), emb.dim), dtype=np.float32)
            bs = 64
            for i in range(0, len(slices), bs):
                E[i:i + bs] = emb.embed_batch(slices[i:i + bs], TARGET_SR)
                if (i // bs) % 10 == 0:
                    print(f"  {min(i + bs, len(slices))}/{len(slices)}")
            P = np.stack([pitch_temporal_features(s) for s in slices]).astype(np.float32)
            fused = np.hstack([E, P]).astype(np.float32)
        else:
            print("  torch/transformers absent — skipping fused features.")

    out = {"hand": hand, "y_leaf": y_leaf, "y_family": y_family, "groups": groups}
    if fused is not None:
        out["fused"] = fused
    np.savez_compressed(cache_path(domain), **out)
    print(f"Cached {len(y_leaf)} samples -> {cache_path(domain).name}")
    return out


def load_cache(domain: str, rebuild: bool, per_piece: int, limit_kits,
               want_fused: bool) -> dict:
    cp = cache_path(domain)
    if cp.exists() and not rebuild:
        d = np.load(cp, allow_pickle=True)
        return {k: d[k] for k in d.files}
    return build_cache(domain, per_piece, limit_kits, want_fused)


# ---------------------------------------------------------------------------
# Bake-off helpers
# ---------------------------------------------------------------------------

def _feature_types(cache: dict, want_fused: bool) -> list[str]:
    fts = ["hand"]
    if want_fused and "fused" in cache:
        fts.append("fused")
    return fts


def _grouped_f1(head: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                n_splits: int) -> float:
    """Leave-one-group-out macro-F1 for one head on one feature matrix."""
    from sklearn.model_selection import GroupKFold, cross_val_score
    n_groups = len(set(groups))
    if n_groups < 2 or len(y) < 2 * n_splits or len(set(y)) < 2:
        return float("nan")
    k = min(n_splits, n_groups)
    try:
        return cross_val_score(make_model(head), X, y, groups=groups,
                               cv=GroupKFold(k), scoring="f1_macro",
                               error_score="raise").mean()
    except Exception:
        return float("nan")


def bakeoff(cache: dict, idx: np.ndarray, y: np.ndarray, heads: list[str],
            fts: list[str], n_splits: int, label: str) -> tuple[str, str, float]:
    """Pick the (head, feature_type) with the best grouped macro-F1 for a node."""
    groups = cache["groups"][idx]
    best = ("gb", "hand", float("-inf"))
    for ft in fts:
        X = cache[ft][idx]
        for head in heads:
            f1 = _grouped_f1(head, X, y, groups, n_splits)
            if not np.isnan(f1) and f1 > best[2]:
                best = (head, ft, f1)
    print(f"    {label:16s} -> {best[0]}/{best[1]}  (macroF1 {best[2]:.3f})")
    return best if best[2] > float("-inf") else ("gb", fts[0], float("nan"))


def _fit(head: str, ft: str, cache: dict, idx: np.ndarray, y: np.ndarray):
    clf = make_model(head)
    clf.fit(cache[ft][idx], y)
    return clf


# ---------------------------------------------------------------------------
# Architecture assembly (choices selected once; refit per fold for eval)
# ---------------------------------------------------------------------------

def select_choices(cache: dict, heads: list[str], fts: list[str],
                   n_splits: int) -> dict:
    """One-time bake-off of every node/specialist over the full data."""
    y_leaf, y_family = cache["y_leaf"], cache["y_family"]
    all_idx = np.arange(len(y_leaf))
    leaves = sorted(set(y_leaf))
    families = sorted(set(y_family))

    print("  cascade bake-off:")
    cascade = {"family": bakeoff(cache, all_idx, y_family, heads, fts, n_splits, "family")}
    cascade["fine"] = {}
    for fam in families:
        fam_leaves = [l for l in TAXONOMY.get(fam, []) if l in leaves]
        if len(fam_leaves) < 2:
            continue
        idx = all_idx[np.isin(y_leaf, fam_leaves)]
        cascade["fine"][fam] = bakeoff(cache, idx, y_leaf[idx], heads, fts,
                                       n_splits, f"fine:{fam}")

    print("  flat specialist bake-off:")
    flat = {}
    for leaf in leaves:
        y_bin = np.where(y_leaf == leaf, leaf, "_rest")
        flat[leaf] = bakeoff(cache, all_idx, y_bin, heads, fts, n_splits, f"ovr:{leaf}")
    return {"cascade": cascade, "flat": flat, "leaves": leaves, "families": families}


def fit_cascade(cache: dict, idx: np.ndarray, choices: dict) -> CascadeClassifier:
    y_leaf, y_family = cache["y_leaf"], cache["y_family"]
    leaves_present = sorted(set(y_leaf[idx]))
    fh, ff, _ = choices["cascade"]["family"]
    family_node = Node(_fit(fh, ff, cache, idx, y_family[idx]), ff)
    fine_nodes = {}
    for fam, (h, ft, _) in choices["cascade"]["fine"].items():
        fam_leaves = [l for l in TAXONOMY.get(fam, []) if l in leaves_present]
        sub = idx[np.isin(y_leaf[idx], fam_leaves)]
        if len(set(y_leaf[sub])) < 2:
            continue
        fine_nodes[fam] = Node(_fit(h, ft, cache, sub, y_leaf[sub]), ft)
    return CascadeClassifier(family_node, fine_nodes, leaves_present)


def fit_flat(cache: dict, idx: np.ndarray, choices: dict) -> FlatSpecialistClassifier:
    from sklearn.calibration import CalibratedClassifierCV
    y_leaf = cache["y_leaf"]
    leaves_present = sorted(set(y_leaf[idx]))
    specialists = {}
    for leaf in leaves_present:
        h, ft, _ = choices["flat"][leaf]
        y_bin = np.where(y_leaf[idx] == leaf, leaf, "_rest")
        try:
            base = CalibratedClassifierCV(make_model(h), method="isotonic", cv=3)
            base.fit(cache[ft][idx], y_bin)
            est = base
        except Exception:
            est = _fit(h, ft, cache, idx, y_bin)
        specialists[leaf] = Node(est, ft, positive_label=leaf)
    return FlatSpecialistClassifier(specialists, leaves_present)


def _feat(cache: dict, idx: np.ndarray, fts: list[str]) -> dict:
    return {ft: cache[ft][idx] for ft in fts if ft in cache}


def eval_architecture(kind: str, cache: dict, choices: dict, fts: list[str],
                      n_splits: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions for an assembled architecture (leave-group-out)."""
    from sklearn.model_selection import GroupKFold
    y_leaf, groups = cache["y_leaf"], cache["groups"]
    all_idx = np.arange(len(y_leaf))
    oof = np.empty(len(y_leaf), dtype=object)
    k = min(n_splits, len(set(groups)))
    for tr, te in GroupKFold(k).split(all_idx, y_leaf, groups):
        model = (fit_cascade if kind == "cascade" else fit_flat)(cache, tr, choices)
        oof[te] = model.predict(_feat(cache, te, fts))
    return y_leaf, oof


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(domain: str, quick: bool, rebuild: bool, per_piece: int, limit_kits) -> None:
    from sklearn.metrics import f1_score, classification_report
    heads = QUICK_HEADS if quick else ALL_HEADS
    want_fused = not quick
    n_splits = 3 if quick else 5

    cache = load_cache(domain, rebuild, per_piece, limit_kits, want_fused)
    fts = _feature_types(cache, want_fused)
    print(f"\n=== {domain.upper()} · heads={heads} · features={fts} ===")
    print(f"  {len(cache['y_leaf'])} samples · {len(set(cache['groups']))} groups · "
          f"leaves {dict(sorted(Counter(cache['y_leaf']).items()))}")

    print("\n[1/3] Selecting best estimator+features per node…")
    choices = select_choices(cache, heads, fts, n_splits)

    print("\n[2/3] Honest leave-one-group-out comparison (cascade vs flat)…")
    results = {}
    for kind in ("cascade", "flat"):
        y_true, oof = eval_architecture(kind, cache, choices, fts, n_splits)
        macro = f1_score(y_true, oof, average="macro", zero_division=0)
        results[kind] = (macro, y_true, oof)
        print(f"  {kind:8s} LOGO macro-F1 = {macro:.3f}")
    winner = max(results, key=lambda k: results[k][0])
    print(f"\n  WINNER: {winner} (macro-F1 {results[winner][0]:.3f})")
    print("\n  per-class (winner, out-of-fold):")
    print(classification_report(results[winner][1], results[winner][2],
                                zero_division=0))

    print("[3/3] Refitting winner on all data + saving bundle…")
    all_idx = np.arange(len(cache["y_leaf"]))
    model = (fit_cascade if winner == "cascade" else fit_flat)(cache, all_idx, choices)
    bundle = {
        # "pipeline" is the key the GUI + pipeline read as the classifier model,
        # so a hybrid bundle is a drop-in for the old sklearn bundles.
        "pipeline": model,
        "feature_type": "hybrid",
        "labels": sorted(set(cache["y_leaf"])),
        "domain": domain,
        "architecture": winner,
        "cv_macro_f1": float(results[winner][0]),
        "choices": choices,
        "feature_types": sorted(model.feature_types_needed()),
    }
    import joblib
    MODEL_DIR.mkdir(exist_ok=True)
    out = MODEL_DIR / f"hybrid_{domain}.joblib"
    joblib.dump(bundle, out)
    print(f"Saved -> models/{out.name}  (arch={winner}, "
          f"feature_types={bundle['feature_types']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["electronic", "acoustic"])
    ap.add_argument("--quick", action="store_true", help="hand-only, fewer heads/folds")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--per", type=int, default=6, help="acoustic per-piece cap")
    ap.add_argument("--limit-kits", type=int, default=None)
    args = ap.parse_args()
    run(args.domain, args.quick, args.rebuild_cache, args.per, args.limit_kits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
