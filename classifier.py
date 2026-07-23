#!/usr/bin/env python3
"""Supervised drum-sample classifier for Samplebot-3000.

Trains on the sorted training-data bins using the same 57-dim feature vector as
the unsupervised pipeline (pipeline.compute_features), so train-time and live
inference use an identical representation.

Mirrors the unsupervised "pick an algorithm" UX: five selectable models —
  svm  : SVC, RBF kernel        (recommended default)
  rf   : RandomForest
  knn  : k-Nearest Neighbours
  gb   : HistGradientBoosting
  mlp  : multi-layer perceptron (small neural net)

Each is wrapped in a StandardScaler pipeline; class_weight='balanced' is applied
where the estimator supports it, to cope with the imbalanced bins.

Feature extraction is the slow part, so it's cached to feature_cache.npz.

Usage:
  python classifier.py --extract            # build/refresh the feature cache
  python classifier.py --train --model svm  # train + evaluate one model
  python classifier.py --compare            # cross-val accuracy for all models
  python classifier.py --predict path.wav --model svm
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline import (
    load_audio, features_from_oneshot, pitch_temporal_features, TARGET_SR,
)

ROOT = Path(__file__).resolve().parent / "training data"
CACHE = Path(__file__).resolve().parent / "feature_cache.npz"
FUSED_CACHE = Path(__file__).resolve().parent / "fused_feature_cache.npz"
MODEL_DIR = Path(__file__).resolve().parent / "models"

# Bins used as labels. Generic `toms` is intentionally excluded: most of its
# files lost their pitch label, so it would blur the hi/mid/lo boundaries.
TRAIN_BINS = [
    "kick", "snare", "clap", "hats", "ride", "crash",
    "hitom", "midtom", "lotom", "fx",
]
AUDIO_EXTS = {".wav", ".aiff", ".aif", ".mp3", ".snd"}

MODEL_NAMES = ["svm", "rf", "knn", "gb", "mlp"]

# Embedding-based models: features are CLAP audio embedding ++ pitch/temporal
# scalars (brief §2+§3), not the 57-dim hand-crafted vector. Their bundles carry
# feature_type="fused_clap" so the pipeline re-embeds at inference.
ALL_MODEL_NAMES = ["clap"] + MODEL_NAMES  # CLAP first = preferred default

# Human-readable names for the GUI picker.
MODEL_LABELS = {
    "ensemble": "Ensemble (vote)",
    "clap": "CLAP+Pitch (GB)",
    "svm": "SVM (RBF)",
    "rf": "Random Forest",
    "knn": "k-NN",
    "gb": "Gradient Boost",
    "mlp": "Neural Net (MLP)",
}


def _trained_base_models() -> list[str]:
    """Base model names (not the ensemble) that have a saved .joblib."""
    if not MODEL_DIR.exists():
        return []
    return [n for n in ALL_MODEL_NAMES if (MODEL_DIR / f"{n}.joblib").exists()]


HYBRID_DOMAINS = ["acoustic", "electronic"]


def available_hybrids() -> list[str]:
    """Domains ('acoustic'/'electronic') with a trained hybrid bundle on disk.

    The app uses ONLY the hybrid classifier; the GUI offers these as a domain
    toggle. Load one with load_bundle(f'hybrid_{domain}')."""
    if not MODEL_DIR.exists():
        return []
    return [d for d in HYBRID_DOMAINS
            if (MODEL_DIR / f"hybrid_{d}.joblib").exists()]


def available_models() -> list[str]:
    """Selectable model names. Offers 'ensemble' when ≥2 base models exist so it
    has something to vote across."""
    base = _trained_base_models()
    return (["ensemble"] + base) if len(base) >= 2 else base


def load_bundle(name: str):
    """Load a model bundle: {'pipeline': ..., 'labels': [...], 'feature_type': ...}.

    The 'ensemble' pseudo-model is assembled on the fly from every trained base
    model; its 'pipeline' is the list of (pipeline, feature_type) members to vote.
    """
    import joblib
    if name == "ensemble":
        members, labels = [], set()
        for n in _trained_base_models():
            b = joblib.load(MODEL_DIR / f"{n}.joblib")
            members.append((b["pipeline"], b.get("feature_type", "hand")))
            labels.update(b.get("labels", []))
        return {"pipeline": members, "labels": sorted(labels),
                "feature_type": "ensemble"}
    return joblib.load(MODEL_DIR / f"{name}.joblib")


# -------- feature extraction --------

def file_features(path: Path) -> np.ndarray | None:
    """57-dim features for one one-shot sample.

    Routes through pipeline.features_from_oneshot so training featurizes each
    one-shot exactly as the live pipeline featurizes an onset-aligned slice
    (front-trim → 0.5s cap → tail-trim → peak-normalize → compute_features).
    This is what keeps train-time and serve-time features identical.
    """
    try:
        audio = load_audio(str(path))
    except Exception:
        return None
    if audio.size == 0:
        return None
    result = features_from_oneshot(audio)
    return None if result is None else result[1]


def extract_all() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, paths = [], [], []
    skipped = 0
    for label in TRAIN_BINS:
        d = ROOT / label
        files = sorted(f for f in d.glob("*")
                       if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        ok = 0
        for f in files:
            feats = file_features(f)
            if feats is None:
                skipped += 1
                continue
            X.append(feats)
            y.append(label)
            paths.append(str(f.relative_to(ROOT)))
            ok += 1
        print(f"  {label:7s} {ok:5d} samples")
    if skipped:
        print(f"  (skipped {skipped} unreadable/too-short files)")
    return np.asarray(X, dtype=np.float32), np.asarray(y), paths


def load_cache() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not CACHE.exists():
        print("No feature cache. Run:  python classifier.py --extract")
        sys.exit(1)
    data = np.load(CACHE, allow_pickle=True)
    return data["X"], data["y"], list(data["paths"])


# -------- models --------

def make_model(name: str):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    name = name.lower()
    if name == "svm":
        from sklearn.svm import SVC
        clf = SVC(kernel="rbf", C=10.0, gamma="scale",
                  class_weight="balanced", probability=True, random_state=42)
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     n_jobs=-1, random_state=42)
    elif name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
    elif name == "gb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(class_weight="balanced",
                                             random_state=42)
    elif name == "mlp":
        from sklearn.neural_network import MLPClassifier
        # early_stopping is left off: in sklearn 1.8 its internal validation
        # scorer calls np.isnan on string class labels and errors. Training-loss
        # convergence (tol/n_iter_no_change) is plenty on this dataset size.
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000,
                            early_stopping=False, random_state=42)
    else:
        raise ValueError(f"unknown model {name!r} (choose from {MODEL_NAMES})")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# -------- commands --------

def cmd_extract() -> None:
    print("Extracting features…")
    X, y, paths = extract_all()
    np.savez_compressed(CACHE, X=X, y=y, paths=np.array(paths, dtype=object))
    print(f"\nCached {len(y)} samples × {X.shape[1]} features -> {CACHE.name}")


def cmd_train(model: str) -> None:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    X, y, _ = load_cache()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = make_model(model)
    print(f"Training '{model}' on {len(ytr)} samples…")
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    print(f"\n=== {model} — held-out test report ===")
    print(classification_report(yte, pred, zero_division=0))

    labels = sorted(set(y))
    cm = confusion_matrix(yte, pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{l[:5]:>5s}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"  {l[:6]:6s} " + " ".join(f"{v:5d}" for v in row))

    cv = cross_val_score(make_model(model), X, y, cv=5, scoring="accuracy")
    print(f"\n5-fold CV accuracy: {cv.mean():.3f} ± {cv.std():.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    out = MODEL_DIR / f"{model}.joblib"
    joblib.dump({"pipeline": pipe, "labels": labels}, out)
    print(f"Saved -> models/{out.name}")


def cmd_compare() -> None:
    from sklearn.model_selection import cross_val_score

    X, y, _ = load_cache()
    print(f"5-fold CV accuracy on {len(y)} samples:\n")
    results = []
    for name in MODEL_NAMES:
        try:
            cv = cross_val_score(make_model(name), X, y, cv=5, scoring="accuracy")
        except Exception as e:
            print(f"  {name:4s}  FAILED: {type(e).__name__}: {e}")
            continue
        results.append((name, cv.mean(), cv.std()))
        print(f"  {name:4s}  {cv.mean():.3f} ± {cv.std():.3f}")
    if results:
        best = max(results, key=lambda r: r[1])
        print(f"\nBest: {best[0]} ({best[1]:.3f})")


def extract_fused_all() -> tuple[np.ndarray, np.ndarray]:
    """Fused CLAP-embedding ++ pitch/temporal features over the training bins.

    Each one-shot is reduced to its fixed-window slice (features_from_oneshot),
    then CLAP-embedded (batched) and concatenated with the 5 pitch/temporal
    scalars — the same representation embed_eval found best (macro-F1 0.75).
    """
    from embedders import get_embedder
    emb = get_embedder("clap")
    slices, y = [], []
    for label in TRAIN_BINS:
        d = ROOT / label
        files = sorted(f for f in d.glob("*")
                       if f.is_file() and f.suffix.lower() in AUDIO_EXTS)
        ok = 0
        for f in files:
            try:
                audio = load_audio(str(f))
            except Exception:
                continue
            r = features_from_oneshot(audio)
            if r is None:
                continue
            slices.append(r[0])
            y.append(label)
            ok += 1
        print(f"  {label:7s} {ok:5d} samples")
    print(f"Embedding {len(slices)} slices via CLAP (batched)…")
    bs = 64
    E = np.empty((len(slices), emb.dim), dtype=np.float32)
    for i in range(0, len(slices), bs):
        E[i:i + bs] = emb.embed_batch(slices[i:i + bs], TARGET_SR)
    P = np.stack([pitch_temporal_features(a) for a in slices]).astype(np.float32)
    return np.hstack([E, P]).astype(np.float32), np.asarray(y)


def cmd_train_clap() -> None:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import joblib

    if FUSED_CACHE.exists():
        d = np.load(FUSED_CACHE, allow_pickle=True)
        X, y = d["X"], d["y"]
    else:
        print("Building fused feature cache…")
        X, y = extract_fused_all()
        np.savez_compressed(FUSED_CACHE, X=X, y=y)
        print(f"Cached {len(y)} × {X.shape[1]} -> {FUSED_CACHE.name}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = make_model("gb")
    print(f"Training CLAP+Pitch (gb) on {len(ytr)} samples…")
    pipe.fit(Xtr, ytr)
    print("\n=== clap — held-out test report ===")
    print(classification_report(yte, pipe.predict(Xte), zero_division=0))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "labels": sorted(set(y)), "feature_type": "fused_clap"},
        MODEL_DIR / "clap.joblib")
    print("Saved -> models/clap.joblib")


def _file_feature_matrix(path: Path, feature_type: str) -> np.ndarray | None:
    """One-row feature matrix for a single file, matching how the model trained.

    "hand" → the 57-dim vector; "fused_clap" → CLAP embedding ++ pitch/temporal
    scalars over the fixed-window slice (same representation as extract_fused_all).
    """
    if feature_type == "fused_clap":
        from embedders import get_embedder
        try:
            audio = load_audio(str(path))
        except Exception:
            return None
        r = features_from_oneshot(audio)
        if r is None:
            return None
        sl = r[0]
        E = get_embedder("clap").embed_batch([sl], TARGET_SR)
        P = np.stack([pitch_temporal_features(sl)]).astype(np.float32)
        return np.hstack([E, P]).astype(np.float64)
    feats = file_features(path)
    return None if feats is None else feats.reshape(1, -1).astype(np.float64)


def cmd_predict(path: str, model: str) -> None:
    import joblib

    mp = MODEL_DIR / f"{model}.joblib"
    if not mp.exists():
        print(f"No trained {model} model. Run: python classifier.py --train --model {model}")
        sys.exit(1)
    bundle = joblib.load(mp)
    feature_type = bundle.get("feature_type", "hand")

    if feature_type == "ensemble":
        votes = []
        for member, ftype in bundle["pipeline"]:
            X = _file_feature_matrix(Path(path), ftype)
            if X is None:
                print("Could not read/analyse that file.")
                sys.exit(1)
            votes.append(str(member.predict(X)[0]))
        pred = Counter(votes).most_common(1)[0][0]
        print(f"Prediction: {pred}")
        print("Votes: " + ", ".join(votes))
        return

    pipe = bundle["pipeline"]
    X = _file_feature_matrix(Path(path), feature_type)
    if X is None:
        print("Could not read/analyse that file.")
        sys.exit(1)
    pred = pipe.predict(X)[0]
    print(f"Prediction: {pred}")
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        order = np.argsort(proba)[::-1][:3]
        print("Top 3:")
        for i in order:
            print(f"  {pipe.classes_[i]:7s} {proba[i]:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--predict", metavar="FILE")
    ap.add_argument("--model", default="svm", choices=ALL_MODEL_NAMES)
    args = ap.parse_args()

    if args.extract:
        cmd_extract()
    elif args.compare:
        cmd_compare()
    elif args.predict:
        cmd_predict(args.predict, args.model)
    elif args.train:
        cmd_train_clap() if args.model == "clap" else cmd_train(args.model)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
