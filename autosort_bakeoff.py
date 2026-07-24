"""Bake-off: which feature set best predicts the drum FAMILY (for the autosorter)?

Compares, by GroupKFold (leave-kit/machine-out) macro-F1:
  - lexicon      : CLAP text-lexicon fingerprint (lexicon.py)      ~50 dims
  - hand         : the 57-dim acoustic vector
  - clap512      : raw CLAP audio embedding                        512 dims
  - fused_lex    : lexicon + hand + pitch/temporal   (the hypothesis)
  - fused_clap   : CLAP512 + pitch/temporal          (existing fused_clap)
plus a zero-shot argmax baseline (the current CLAP-sort's decision, family-mapped).

Target = GM FAMILY (kick, side_stick, snare, clap, tom, hat, crash, ride) via
pipeline.LEAF_TO_GM_FAMILY. Reuses the CLAP embedder + classifier.make_model. The
CLAP embedding of the corpus is cached to autosort_cache_<domain>.npz (slow once).

Run:  python autosort_bakeoff.py --domain acoustic
"""
from __future__ import annotations

import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import lexicon as LEX
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix

HEADS = ["gb", "svm"]
BATCH = 16   # CLAP's HTSAT pads to a long internal window; large batches OOM


def build_or_load_cache(domain: str, per_piece: int = 6, limit_kits=None):
    path = f"autosort_cache_{domain}.npz"
    if os.path.exists(path) and limit_kits is None:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    import drum_data
    from embedders import get_embedder
    print(f"[cache] loading {domain} records…", flush=True)
    records = drum_data.load_records(domain, per_piece=per_piece, limit_kits=limit_kits)
    slices = [r.slice for r in records]
    hand = np.stack([r.hand for r in records]).astype(np.float32)
    y_leaf = np.array([r.leaf for r in records])
    groups = np.array([r.group for r in records])
    print(f"[cache] {len(slices)} slices, {len(set(groups.tolist()))} groups; embedding CLAP…", flush=True)
    emb = get_embedder("clap")
    t0 = time.time()
    E = []
    for i in range(0, len(slices), BATCH):
        E.append(emb.embed_batch(slices[i:i + BATCH], P.TARGET_SR))
        if (i // BATCH) % 10 == 0:
            print(f"  embedded {i}/{len(slices)}  ({time.time()-t0:.0f}s)", flush=True)
    clap512 = np.vstack(E).astype(np.float32)
    pitch5 = np.stack([P.pitch_temporal_features(s) for s in slices]).astype(np.float32)
    lex = LEX.lexicon_fingerprint(clap512)
    if limit_kits is None:
        np.savez(path, hand=hand, clap512=clap512, lexicon=lex, pitch5=pitch5,
                 y_leaf=y_leaf, groups=groups)
        print(f"[cache] wrote {path} in {time.time()-t0:.0f}s", flush=True)
    return dict(hand=hand, clap512=clap512, lexicon=lex, pitch5=pitch5,
                y_leaf=y_leaf, groups=groups)


def to_family(y_leaf):
    return np.array([P.LEAF_TO_GM_FAMILY.get(str(l), "fx") for l in y_leaf])


def grouped_macro_f1(head, X, y, groups, n_splits=5):
    from classifier import make_model
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups.tolist()))))
    pred = cross_val_predict(make_model(head), X, y, groups=groups, cv=gkf)
    return f1_score(y, pred, average="macro"), pred


def zero_shot_family(clap512):
    """Reproduce the current CLAP-sort argmax (8 drum prompts) -> GM family, on the
    matrix, for a comparable baseline (no acoustic overrides — pure CLAP argmax)."""
    drums, Tmat = P._drum_text_vectors()
    sims = clap512.astype(np.float64) @ Tmat.T
    idx = sims.argmax(axis=1)
    leaf = np.array([drums[i] for i in idx])           # kick/snare/hats/tom/…
    return np.array([P.LEAF_TO_GM_FAMILY.get(str(l), "fx") for l in leaf])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="acoustic")
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="limit_kits (fast test)")
    args = ap.parse_args()

    c = build_or_load_cache(args.domain, args.per, limit_kits=args.limit)
    y = to_family(c["y_leaf"])
    groups = c["groups"]
    import collections
    print("\nfamily counts:", dict(collections.Counter(y.tolist())))

    feats = {
        "lexicon":    c["lexicon"],
        "hand":       c["hand"],
        "clap512":    c["clap512"],
        "fused_lex":  np.hstack([c["lexicon"], c["hand"], c["pitch5"]]),
        "fused_clap": np.hstack([c["clap512"], c["pitch5"]]),
    }

    # zero-shot baseline (accuracy + macro-F1 over the whole set, no CV needed)
    zs = zero_shot_family(c["clap512"])
    print(f"\nzero-shot CLAP argmax  macroF1 {f1_score(y, zs, average='macro'):.3f}  "
          f"acc {np.mean(zs==y):.3f}")

    print(f"\n{'feature set':12s} {'dims':>5s}  " + "  ".join(f"{h:>6s}" for h in HEADS))
    results = {}
    for name, X in feats.items():
        row = []
        for head in HEADS:
            f1, pred = grouped_macro_f1(head, X, y, groups)
            row.append(f1); results[(name, head)] = (f1, pred)
        print(f"{name:12s} {X.shape[1]:5d}  " + "  ".join(f"{f:6.3f}" for f in row))

    # per-class + confusion for the winner
    (bn, bh), (bf, bpred) = max(results.items(), key=lambda kv: kv[1][0])
    print(f"\nWINNER: {bn} + {bh}  (macroF1 {bf:.3f})")
    labels = sorted(set(y.tolist()))
    print(classification_report(y, bpred, labels=labels, digits=3, zero_division=0))
    print("confusion (rows=true):", labels)
    print(confusion_matrix(y, bpred, labels=labels))


if __name__ == "__main__":
    main()
