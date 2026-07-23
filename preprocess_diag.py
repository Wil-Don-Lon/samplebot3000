#!/usr/bin/env python3
"""Diagnostic report on the current detection / segmentation / preprocessing path.

REPORT ONLY — changes nothing. It quantifies where the front end may be hurting
the classifier, to scope the (separate) preprocessing overhaul:

  - onset detection vs sensitivity + inter-onset gaps (needs a multi-hit --audio)
  - source one-shot duration distribution and how much the 0.5s classify cap
    truncates (train/serve window mismatch)
  - loudness spread (peak vs RMS) — is peak-normalization enough, or is perceived
    level still all over the place?
  - leading silence before the first transient (onset-alignment quality)
  - channel/sample-rate consistency of the raw corpus
  - optional near-duplicate rate via a CLAP embed cache, if one exists

Usage:
  python preprocess_diag.py                       # corpus stats
  python preprocess_diag.py --audio loop.wav      # + onset-detection behavior
  python preprocess_diag.py --sample 400          # files sampled per corpus
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from pipeline import (
    load_audio, detect_onsets, trim_silence, TARGET_SR, MIN_SEGMENT_SAMPLES,
    CLASSIFY_FEATURE_LEN_S,
)
from model_analysis import RAW, AUDIO_EXTS

HERE = Path(__file__).resolve().parent


def _pct(a, ps=(50, 75, 90, 99)):
    a = np.asarray(a, float)
    return {p: float(np.percentile(a, p)) for p in ps} if a.size else {}


def _leading_silence_s(audio: np.ndarray, thresh_db: float = -38.0) -> float:
    """Seconds before the signal first crosses the trim threshold."""
    if audio.size == 0:
        return 0.0
    thr = 10.0 ** (thresh_db / 20.0) * float(np.max(np.abs(audio)) or 1.0)
    above = np.where(np.abs(audio) >= thr)[0]
    return float(above[0]) / TARGET_SR if above.size else 0.0


def _corpus_files(sample: int) -> list[Path]:
    files: list[Path] = []
    if RAW.exists():
        for machine in sorted(RAW.iterdir()):
            if machine.is_dir():
                files += [f for f in machine.rglob("*")
                          if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
    random.seed(0)
    random.shuffle(files)
    return files[:sample]


def corpus_report(sample: int) -> None:
    files = _corpus_files(sample)
    print(f"\n=== CORPUS PREPROCESSING STATS  (n={len(files)} sampled raw files) ===")
    if not files:
        print("  (no _raw corpus found)")
        return
    durs, peaks, rmss, leads, srs, chans, too_short = [], [], [], [], [], [], 0
    cap = CLASSIFY_FEATURE_LEN_S
    for f in files:
        try:
            import soundfile as sf
            info = sf.info(str(f))
            srs.append(info.samplerate)
            chans.append(info.channels)
        except Exception:
            pass
        try:
            audio = load_audio(str(f))
        except Exception:
            continue
        trimmed = trim_silence(audio)
        if trimmed.size < MIN_SEGMENT_SAMPLES:
            too_short += 1
            continue
        durs.append(trimmed.size / TARGET_SR)
        peaks.append(float(np.max(np.abs(audio))))
        rmss.append(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))))
        leads.append(_leading_silence_s(audio))

    d = np.asarray(durs)
    print(f"  channels : {dict(_count(chans))}   sample rates: {dict(_count(srs))}")
    print(f"  too-short after trim: {too_short}")
    print(f"  duration (trimmed, s): {_fmt(_pct(durs))}")
    print(f"    > {cap}s cap (truncated at classify time): "
          f"{100*float((d > cap).mean()):.0f}%")
    print(f"  peak level    : {_fmt(_pct(peaks))}")
    print(f"  RMS level     : {_fmt(_pct(rmss))}  "
          f"(spread here ⇒ peak-norm leaves loudness uneven)")
    print(f"  leading silence before 1st transient (s): {_fmt(_pct(leads))}")


def onset_report(audio_path: str) -> None:
    print(f"\n=== ONSET DETECTION vs SENSITIVITY  ({Path(audio_path).name}) ===")
    audio = load_audio(audio_path)
    dur = audio.size / TARGET_SR
    print(f"  file: {dur:.1f}s")
    for sens in (0.2, 0.35, 0.5, 0.65, 0.8):
        onsets = detect_onsets(audio, sensitivity=sens)
        gaps = np.diff(onsets) if len(onsets) > 1 else np.array([])
        gtxt = (f"min gap {gaps.min():.3f}s / median {np.median(gaps):.3f}s"
                if gaps.size else "n/a")
        print(f"  sens {sens:.2f}: {len(onsets):4d} onsets   {gtxt}")


def near_dup_report() -> None:
    """Reuse an existing CLAP embed cache to estimate near-duplicate rate."""
    from acoustic_eval import _dedup
    for cache in ("embed_cache_clap.npz", "hybrid_cache_acoustic.npz"):
        cp = HERE / cache
        if not cp.exists():
            continue
        d = np.load(cp, allow_pickle=True)
        key = "X" if "X" in d.files else ("fused" if "fused" in d.files else None)
        if key is None:
            continue
        X = d[key].astype(np.float32)
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-8, None)
        labels = d["y"] if "y" in d.files else (d["y_leaf"] if "y_leaf" in d.files else None)
        keep = _dedup(X, threshold=0.999, labels=labels)
        print(f"\n=== NEAR-DUPLICATE RATE  ({cache}) ===")
        print(f"  {len(keep)} samples · {int((~keep).sum())} near-dupes "
              f"({100*float((~keep).mean()):.1f}%) at cosine>0.999")
        return
    print("\n=== NEAR-DUPLICATE RATE ===\n  (no CLAP embed cache found — skipped)")


def _count(xs):
    from collections import Counter
    return dict(sorted(Counter(xs).items()))


def _fmt(pct: dict) -> str:
    return "  ".join(f"p{k} {v:.2f}" for k, v in pct.items()) if pct else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="a multi-hit file for onset stats")
    ap.add_argument("--sample", type=int, default=400)
    args = ap.parse_args()
    print("PREPROCESSING DIAGNOSTIC — report only, nothing is modified.")
    corpus_report(args.sample)
    if args.audio:
        onset_report(args.audio)
    near_dup_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
