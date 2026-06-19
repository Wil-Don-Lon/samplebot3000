"""Regression test: train-time and serve-time features must be identical.

Brief §1 acceptance — there is exactly ONE 'audio slice -> feature vector'
implementation, and a known sample produces identical features through the
training entry point (classifier.file_features / features_from_oneshot) and the
serving entry point (run_pipeline -> extract_segments). If a future edit makes
one path diverge from the other, this test fails.

Run:  python -m pytest tests/test_train_serve_skew.py -q
  or: python tests/test_train_serve_skew.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (  # noqa: E402
    TARGET_SR, SEGMENT_LENGTH_S, CLASSIFY_FEATURE_LEN_S,
    extract_segments, features_from_oneshot, segment_features,
)


def _synthetic_oneshot() -> np.ndarray:
    """A tom-like one-shot: sharp transient at sample 0, ~1.0s decay.

    Deliberately LONGER than CLASSIFY_FEATURE_LEN_S (0.5s) and still audible
    past it, so the fixed classify window actually caps the sample — that's what
    lets the slider-invariance test below mean something.
    """
    n = int(1.0 * TARGET_SR)
    t = np.arange(n) / TARGET_SR
    audio = np.sin(2 * np.pi * 180.0 * t) * np.exp(-t * 5.0)
    audio[0] = 1.0  # unambiguous transient at the front
    return audio.astype(np.float32)


def test_serving_classify_equals_training_features():
    """Classification serving features == training features for the same onset."""
    audio = _synthetic_oneshot()

    # Serving entry, classification mode: fixed feature window from onset t=0.
    segs = extract_segments(
        audio, np.array([0.0]), feature_length_s=CLASSIFY_FEATURE_LEN_S)
    assert len(segs) == 1, "expected exactly one segment from one onset"
    f_serve = segs[0].features

    # Training entry (pipeline core used by classifier.file_features).
    result = features_from_oneshot(audio)
    assert result is not None
    f_train = result[1]

    assert f_serve.shape == f_train.shape == (57,)
    assert np.array_equal(f_serve, f_train), (
        "train/serve feature skew: classification serving and "
        "features_from_oneshot produced different vectors for the same slice"
    )


def test_classify_features_invariant_to_sample_len():
    """The Sample Len slider must NOT change classification features.

    This is the property that fully closes train/serve skew: a fixed model sees
    the same feature window no matter where the user drags Sample Len. The played
    audio length DOES change; the feature vector does not.
    """
    audio = _synthetic_oneshot()
    onset = np.array([0.0])

    short = extract_segments(
        audio, onset, segment_length_s=0.3, feature_length_s=CLASSIFY_FEATURE_LEN_S)
    long = extract_segments(
        audio, onset, segment_length_s=2.0, feature_length_s=CLASSIFY_FEATURE_LEN_S)
    assert len(short) == len(long) == 1

    # Playback slices differ in length with the slider...
    assert short[0].audio.size != long[0].audio.size
    # ...but the classifier features are identical.
    assert np.array_equal(short[0].features, long[0].features), (
        "classification features changed with Sample Len — the fixed feature "
        "window is not actually decoupled from the playback length"
    )


def test_file_features_matches_pipeline():
    """The actual classifier training entry (reads a file) matches the core."""
    import soundfile as sf
    from classifier import file_features
    from pipeline import load_audio

    audio = _synthetic_oneshot()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "oneshot.wav")
        sf.write(path, audio, TARGET_SR, subtype="FLOAT")
        f_file = file_features(path)
        f_core = features_from_oneshot(load_audio(path))[1]

    assert f_file is not None
    # Same code path; allow float epsilon from the wav write/read round-trip.
    assert np.allclose(f_file, f_core, rtol=1e-5, atol=1e-6)


def test_segment_features_is_the_shared_impl():
    """features_from_oneshot delegates to segment_features (single impl)."""
    audio = _synthetic_oneshot()
    cap = int(CLASSIFY_FEATURE_LEN_S * TARGET_SR)
    direct = segment_features(audio[:cap])
    viashot = features_from_oneshot(audio)
    assert direct is not None and viashot is not None
    assert np.array_equal(direct[1], viashot[1])


if __name__ == "__main__":
    test_serving_classify_equals_training_features()
    test_classify_features_invariant_to_sample_len()
    test_file_features_matches_pipeline()
    test_segment_features_is_the_shared_impl()
    print("OK — classify features match training and are slider-invariant.")
