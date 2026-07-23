"""Equal-loudness (A-weighted) perceived-loudness normalization.

The ear is far less sensitive to lows than mids/highs (the Fletcher-Munson /
ISO-226 equal-loudness contours), so two samples peak-normalized to the same
level are NOT perceived equally loud — a bright hat sounds harsher/louder than a
kick. We estimate each sample's PERCEIVED loudness by weighting its spectrum with
the A-weighting curve, then compute per-sample gains that pull everything to a
common perceived loudness.

Samples are already peak-normalized, so we can't push a kick louder without
clipping — instead we tame the bright/harsh ones DOWN to match (which makes the
low end sit prominent), then apply a single global make-up so the loudest-peaking
sample just touches the ceiling. Net: balanced perceived loudness, no clipping,
no timbre change (it's pure per-sample gain, not EQ).
"""
from __future__ import annotations

import numpy as np

_A_FLOOR_DB = -24.0     # never attenuate an individual sample more than this
_REF_PERCENTILE = 10.0  # target loudness = this percentile of the kit (the quiet end)
# Full A-weighting is too aggressive (a crash lands ~-15 dB). Blend toward unity:
# 1.0 = full equal-loudness, 0.0 = no effect. ~0.4 keeps it subtle but audible.
_STRENGTH = 0.4


def a_weighting_linear(freqs: np.ndarray) -> np.ndarray:
    """A-weighting (IEC 61672) as a linear magnitude per frequency."""
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-6)
    f2 = f * f
    ra = (12194.0 ** 2 * f2 * f2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    a_db = 20.0 * np.log10(ra) + 2.00
    return 10.0 ** (a_db / 20.0)


def perceptual_loudness(audio: np.ndarray, sr: int) -> float:
    """A-weighted spectral loudness of one sample (linear, RMS-like)."""
    a = np.asarray(audio, dtype=np.float64).ravel()
    n = a.size
    if n < 32:
        return 1e-9
    X = np.abs(np.fft.rfft(a * np.hanning(n)))
    w = a_weighting_linear(np.fft.rfftfreq(n, 1.0 / sr))
    return float(np.sqrt(np.sum((X * w) ** 2) / n)) + 1e-12


def equal_loudness_gains(audios: list[np.ndarray], sr: int,
                         strength: float = _STRENGTH,
                         ceiling: float = 0.99) -> np.ndarray:
    """Per-sample gains equalizing perceived loudness across the kit.

    Returns one gain per input sample. Guarantees no sample peaks above `ceiling`
    (so nothing clips). Gains ≤ 1 balance the kit toward its quiet (low-frequency)
    end; a shared make-up then lifts everything so the loudest-peaking sample
    reaches the ceiling. `strength` blends toward unity (1 = full A-weighting,
    0 = off) so the effect can be kept subtle.
    """
    if not audios:
        return np.ones(0)
    L = np.array([perceptual_loudness(a, sr) for a in audios])
    L_ref = np.percentile(L, _REF_PERCENTILE)
    floor = 10.0 ** (_A_FLOOR_DB / 20.0)
    g = np.clip(L_ref / L, floor, 1.0)                       # tame the loud-perceived
    g = g ** float(np.clip(strength, 0.0, 1.0))              # soften toward unity

    peaks = np.array([float(np.abs(a).max()) * gi for a, gi in zip(audios, g)])
    max_peak = float(peaks.max()) if peaks.size else 0.0
    if max_peak > 1e-9:
        g = g * (ceiling / max_peak)                         # global make-up to ceiling
    # Safety: clamp any residual over-ceiling sample.
    for i, a in enumerate(audios):
        p = float(np.abs(a).max())
        if p * g[i] > ceiling:
            g[i] = ceiling / (p + 1e-12)
    return g
