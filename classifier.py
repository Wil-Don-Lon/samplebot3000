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
from pathlib import Path

import numpy as np

from pipeline import load_audio, trim_silence, compute_features

ROOT = Path(__file__).resolve().parent / "training data"
CACHE = Path(__file__).resolve().parent / "feature_cache.npz"
MODEL_DIR = Path(__file__).resolve().parent / "models"

# Bins used as labels. Generic `toms` is intentionally excluded: most of its
# files lost their pitch label, so it would blur the hi/mid/lo boundaries.
TRAIN_BINS = [
    "kick", "snare", "clap", "hats", "ride", "crash",
    "hitom", "midtom", "lotom", "fx",
]
AUDIO_EXTS = {".wav", ".aiff", ".aif", ".mp3", ".snd"}

MODEL_NAMES = ["svm", "rf", "knn", "gb", "mlp"]

# Human-readable names for the GUI picker.
MODEL_LABELS = {
    "svm": "SVM (RBF)",
    "rf": "Random Forest",
    "knn": "k-NN",
    "gb": "Gradient Boost",
    "mlp": "Neural Net (MLP)",
}


def available_models() -> list[str]:
    """Trained model names (in preference order) that have a saved .joblib."""
    if not MODEL_DIR.exists():
        return []
    return [n for n in MODEL_NAMES if (MODEL_DIR / f"{n}.joblib").exists()]


def load_bundle(name: str):
    """Load a trained model bundle: {'pipeline': ..., 'labels': [...]}."""
    import joblib
    return joblib.load(MODEL_DIR / f"{name}.joblib")


# -------- feature extraction --------

def file_features(path: Path) -> np.ndarray | None:
    """57-dim features for one one-shot sample (loaded, trimmed, analysed)."""
    try:
        audio = load_audio(str(path))
    except Exception:
        return None
    if audio.size == 0:
        return None
    audio = trim_silence(audio)
    if audio.size < 256:
        return None
    return compute_features(audio)


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


def cmd_predict(path: str, model: str) -> None:
    import joblib

    mp = MODEL_DIR / f"{model}.joblib"
    if not mp.exists():
        print(f"No trained {model} model. Run: python classifier.py --train --model {model}")
        sys.exit(1)
    bundle = joblib.load(mp)
    pipe = bundle["pipeline"]
    feats = file_features(Path(path))
    if feats is None:
        print("Could not read/analyse that file.")
        sys.exit(1)
    X = feats.reshape(1, -1)
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
    ap.add_argument("--model", default="svm", choices=MODEL_NAMES)
    args = ap.parse_args()

    if args.extract:
        cmd_extract()
    elif args.compare:
        cmd_compare()
    elif args.predict:
        cmd_predict(args.predict, args.model)
    elif args.train:
        cmd_train(args.model)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
