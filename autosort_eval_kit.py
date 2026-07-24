"""Real-target eval: the autosorter on the user's hand-labeled kit.

(1) Train the winning sorter (fused_lex = lexicon+hand+pitch, SVM) on the whole
    labeled corpus (autosort_cache_<domain>.npz), then predict the GM FAMILY of
    every one-shot in datasets/test data/ (a fully held-out kit). Report accuracy
    + confusion vs the filename ground truth.
(2) ARI diagnostic (user-requested): does the CLAP-lexicon space reproduce the
    HDBSCAN-on-audio clustering? adjusted_rand_score(HDBSCAN(hand), HDBSCAN(lexicon))
    on the kit, plus how well each clustering matches the true drum families.

Run (after autosort_bakeoff has built the cache):  python autosort_eval_kit.py
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import lexicon as LEX
import drum_data
from embedders import get_embedder
from classifier import make_model
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, adjusted_rand_score)
from sklearn.cluster import HDBSCAN


def _fam(leaves):
    return np.array([P.LEAF_TO_GM_FAMILY.get(str(l), "fx") for l in leaves])


def _zscore(X):
    X = np.asarray(X, dtype=np.float64)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def main(domain="acoustic"):
    cache = f"autosort_cache_{domain}.npz"
    if not os.path.exists(cache):
        print(f"missing {cache} — run autosort_bakeoff.py --domain {domain} first")
        return
    c = np.load(cache, allow_pickle=True)
    Xc = np.hstack([c["lexicon"], c["hand"], c["pitch5"]])
    yc = _fam(c["y_leaf"])

    # --- the user's kit (held out) ---
    print("slicing the user's labeled kit…", flush=True)
    recs = drum_data.load_test_kit()
    slices = [r.slice for r in recs]
    hand = np.stack([r.hand for r in recs]).astype(np.float32)
    y_leaf = np.array([r.leaf for r in recs])
    y_true = _fam(y_leaf)
    emb = get_embedder("clap")
    E = np.vstack([emb.embed_batch(slices[i:i+16], P.TARGET_SR)
                   for i in range(0, len(slices), 16)]).astype(np.float32)
    lex = LEX.lexicon_fingerprint(E)
    p5 = np.stack([P.pitch_temporal_features(s) for s in slices]).astype(np.float32)
    Xk = np.hstack([lex, hand, p5])
    print(f"kit: {len(recs)} one-shots, families {sorted(set(y_true.tolist()))}")

    # --- (1) train on corpus, predict the kit ---
    clf = make_model("svm").fit(Xc, yc)
    pred = clf.predict(Xk)
    labels = sorted(set(y_true.tolist()) | set(pred.tolist()))
    print(f"\n=== SORTER on the user's kit ===  accuracy {accuracy_score(y_true, pred):.3f}")
    print(classification_report(y_true, pred, labels=labels, digits=3, zero_division=0))
    print("confusion (rows=true):", labels)
    print(confusion_matrix(y_true, pred, labels=labels))

    # --- (2) ARI diagnostic ---
    def hdb(X):
        return HDBSCAN(min_cluster_size=3, min_samples=1,
                       cluster_selection_method="leaf").fit_predict(_zscore(X))
    la, ll = hdb(hand), hdb(lex)
    print("\n=== ARI diagnostic (kit) ===")
    print(f"  HDBSCAN(audio hand-57)  vs HDBSCAN(lexicon)  ARI = {adjusted_rand_score(la, ll):.3f}")
    print(f"  HDBSCAN(audio)  vs true families            ARI = {adjusted_rand_score(y_true, la):.3f}")
    print(f"  HDBSCAN(lexicon) vs true families           ARI = {adjusted_rand_score(y_true, ll):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "acoustic")
