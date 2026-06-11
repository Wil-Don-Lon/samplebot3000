"""Audio analysis pipeline: file path -> Instrument.

Three clustering modes:
- "manual"  : KMeans with a user-specified k
- "auto"    : AgglomerativeClustering with a distance threshold
- "hdbscan" : HDBSCAN with min_cluster_size; noise points get reassigned
              to their nearest cluster's centroid so nothing is dropped.

Feature vector (57 dims): captures timbre (MFCC mean/std), timbral evolution
(MFCC delta), pitch content (chroma), spectral shape (centroid mean/std,
rolloff, bandwidth, flatness), and noisiness (zero-crossing rate).

Segments are silence-trimmed on both ends, deduplicated, and per-cluster
volume-normalized before reaching the keyboard.

Octaves overlap on C: octave N covers cluster indices [12N, 12N+12] so the
top C of octave N is the same cluster as the low C of octave N+1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import librosa
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN


# -------- constants --------

TARGET_SR = 44100

# Piano keys for one octave's display (17 visible, 12 unique with overlap).
NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    "C(oct)", "C#(oct)", "D(oct)", "D#(oct)", "E(oct)",
]
NOTE_NAMES_12 = NOTE_NAMES[:12]   # 12 unique pitch classes

SEGMENT_LENGTH_S = 0.5
MIN_SEGMENT_S = 0.05
MIN_ONSET_GAP_S = 0.10            # reject onsets less than 100ms apart
TRIM_THRESHOLD_DB_DEFAULT = -38.0 # default silence threshold for trim (dBFS)
TRIM_SMOOTH_SAMPLES = 64          # ~1.5ms moving-average window for trim

# Similarity-based dedupe: two segments within DEDUP_TIME_WINDOW (set dynamically
# to the longest trimmed duration) and closer than DEDUP_DISTANCE in 57-dim
# z-scored feature space are treated as duplicates.
DEDUP_DISTANCE = 1.5

MAX_KEYS = 48                     # 4 overlapping octaves' worth of unique slots

MIN_SEGMENT_SAMPLES = int(MIN_SEGMENT_S * TARGET_SR)

# Feature vector layout (57 dims):
#   [0..12]   MFCC means        (13)  timbre
#   [13..25]  MFCC stds         (13)  timbre variation
#   [26..38]  MFCC delta means  (13)  timbral evolution / dynamics
#   [39..50]  chroma means      (12)  pitch class energy
#   [51]      centroid mean           brightness
#   [52]      centroid std            brightness variation
#   [53]      rolloff mean            high-frequency content
#   [54]      bandwidth mean          frequency spread
#   [55]      flatness mean           tonal vs noise-like
#   [56]      zero-crossing rate      noisiness
CENTROID_MEAN_INDEX = 51


ProgressCallback = Callable[[str, float], None]


# -------- data classes --------


@dataclass
class Segment:
    audio: np.ndarray
    onset_time: float
    features: np.ndarray
    rms: float
    cluster: int = -1
    dominant_freq: float = 0.0   # FFT peak frequency, used for key ordering


@dataclass
class Instrument:
    sample_rate: int = TARGET_SR
    notes: dict[int, list[Segment]] = field(default_factory=dict)

    def representative(self, note_index: int) -> Optional[Segment]:
        segs = self.notes.get(note_index)
        if not segs:
            return None
        sorted_by_rms = sorted(segs, key=lambda s: s.rms)
        return sorted_by_rms[len(sorted_by_rms) // 2]

    def loaded_notes(self) -> set[int]:
        return {k for k, v in self.notes.items() if v}

    def max_octave(self) -> int:
        """Highest octave index with content beyond the previous octave's overlap.

        Octave N covers cluster indices [12N, 12N+16] (17 visible keys). The
        high 5 keys of octave N overlap with the low 5 keys of octave N+1.
        Octave N+1 is "useful" only if there are clusters past 12N+16.
        """
        loaded = self.loaded_notes()
        if not loaded:
            return 0
        m = max(loaded)
        # Octave N reveals new content (not in octaves 0..N-1) iff m >= 12N + 5
        return max(0, (m - 5) // 12)


# -------- pipeline steps --------


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32, copy=False)


def detect_onsets(
    audio: np.ndarray,
    sensitivity: float = 0.5,
    min_gap_s: float = MIN_ONSET_GAP_S,
) -> np.ndarray:
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    delta = 0.12 - 0.10 * sensitivity
    onset_samples = librosa.onset.onset_detect(
        y=audio, sr=TARGET_SR, delta=delta, backtrack=True, units="samples",
    )
    if len(onset_samples) == 0:
        return np.array([], dtype=np.float64)
    onset_times = onset_samples.astype(np.int64) / TARGET_SR

    # Enforce a minimum gap between consecutive onsets so a single hit doesn't
    # produce two adjacent segments. librosa's internal `wait` doesn't always
    # do this strongly enough.
    if min_gap_s > 0 and len(onset_times) > 1:
        filtered = [float(onset_times[0])]
        for t in onset_times[1:]:
            if float(t) - filtered[-1] >= min_gap_s:
                filtered.append(float(t))
        onset_times = np.array(filtered, dtype=np.float64)
    return onset_times


def trim_silence(
    audio: np.ndarray,
    threshold_db: float = TRIM_THRESHOLD_DB_DEFAULT,
) -> np.ndarray:
    """Strip leading and trailing silence using an absolute dBFS threshold.

    threshold_db is dBFS — the absolute audio level below which we consider
    a region "silent" for trimming purposes. -40 dB is roughly a quiet room
    tone; -30 dB will trim more aggressively into low-level content; -50 dB
    only catches near-silence. The user controls this from the GUI.

    Smoothing the abs(audio) envelope before thresholding prevents a single
    bright sample in an otherwise quiet region from anchoring the trim
    point.
    """
    if audio.size == 0:
        return audio
    threshold_linear = 10.0 ** (float(threshold_db) / 20.0)

    abs_audio = np.abs(audio).astype(np.float64)
    if abs_audio.size > TRIM_SMOOTH_SAMPLES:
        pad = TRIM_SMOOTH_SAMPLES // 2
        padded = np.pad(abs_audio, pad, mode="edge")
        kernel = np.ones(TRIM_SMOOTH_SAMPLES, dtype=np.float64) / TRIM_SMOOTH_SAMPLES
        smoothed = np.convolve(padded, kernel, mode="valid")[:abs_audio.size]
    else:
        smoothed = abs_audio

    above = np.where(smoothed >= threshold_linear)[0]
    if len(above) == 0:
        return audio[:0]
    start = int(above[0])
    end = int(above[-1]) + 1
    return audio[start:end]


def dominant_frequency(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    """Perceived pitch via lowest-strong-peak in the spectrum.

    Naive FFT argmax locks onto the loudest single bin, which for harmonic
    sounds is often a harmonic (2nd or 3rd partial) rather than the
    fundamental, and for noisy sounds is essentially arbitrary because the
    spectrum is flat-ish with noise-driven peaks. Taking the lowest
    frequency that exceeds 50% of the max amplitude gives a more
    perceptually grounded answer: returns the fundamental for tonal
    material (since the fundamental is usually within 6 dB of the loudest
    harmonic), and the lower edge of the dominant band for noisy material.
    """
    if audio.size < 64:
        return 0.0
    n_fft = 8192
    n = min(len(audio), n_fft)
    windowed = audio[:n].astype(np.float64) * np.hanning(n)
    if n < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[:n] = windowed
        windowed = padded
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # Ignore sub-30 Hz: DC drift and room rumble shouldn't drive pitch sorting.
    spec[freqs < 30.0] = 0.0
    peak = float(spec.max())
    if peak < 1e-10:
        return 0.0
    threshold = peak * 0.5
    strong = np.where(spec >= threshold)[0]
    if len(strong) == 0:
        return float(freqs[int(np.argmax(spec))])
    return float(freqs[int(strong[0])])


def extract_segments(
    audio: np.ndarray,
    onset_times: np.ndarray,
    segment_length_s: float = SEGMENT_LENGTH_S,
    trim_threshold_db: float = TRIM_THRESHOLD_DB_DEFAULT,
    noise_gate_db: Optional[float] = None,
) -> list[Segment]:
    """Cut, gate, trim. One segment per surviving onset.

    For each onset t_i:
      1. Form chunk [t_i, t_i + L].
      2. If a later onset t_{i+1} falls inside that range, truncate the
         chunk to end at t_{i+1}. This prevents a long sample length from
         accidentally swallowing the next hit.
      3. Drop the chunk if its peak amplitude is below noise_gate_db (the
         "drop entire sample if too quiet" check).
      4. Trim leading and trailing silence using trim_threshold_db.
      5. Drop the chunk if it's shorter than MIN_SEGMENT_S after trim.
      6. Exact-bytes dedupe as a safety net before clustering-time
         feature dedupe.
    """
    segment_samples = int(segment_length_s * TARGET_SR)
    segments: list[Segment] = []
    seen_hashes: set[int] = set()

    if noise_gate_db is not None:
        gate_linear = 10.0 ** (float(noise_gate_db) / 20.0)
    else:
        gate_linear = 0.0

    for i, t in enumerate(onset_times):
        start = int(t * TARGET_SR)
        end = start + segment_samples

        # Step 2: truncate at the next onset if one falls inside this chunk.
        # Using the onset list directly avoids re-detecting transients —
        # they're the same algorithm and parameters.
        if i + 1 < len(onset_times):
            next_start = int(onset_times[i + 1] * TARGET_SR)
            if next_start < end:
                end = next_start

        chunk = audio[start:end]
        if chunk.size < MIN_SEGMENT_SAMPLES:
            continue

        # Step 3: peak noise gate. Measured on the (possibly truncated)
        # chunk before silence trimming; a loud transient anywhere in the
        # chunk keeps the sample alive.
        peak = float(np.abs(chunk).max())
        if peak < gate_linear:
            continue

        # Step 4: trim leading and trailing silence using the absolute
        # threshold parameter.
        trimmed = trim_silence(chunk, threshold_db=trim_threshold_db)
        if trimmed.size < MIN_SEGMENT_SAMPLES:
            continue
        trimmed = trimmed.astype(np.float32, copy=False)

        # Exact-bytes guard against perfect duplicates.
        h = hash(trimmed.tobytes())
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        rms = float(np.sqrt(np.mean(trimmed.astype(np.float64) ** 2)))
        feats = compute_features(trimmed)
        dom_freq = dominant_frequency(trimmed)
        segments.append(
            Segment(
                audio=trimmed,
                onset_time=float(t),
                features=feats,
                rms=rms,
                dominant_freq=dom_freq,
            )
        )

    return segments


def compute_features(audio: np.ndarray) -> np.ndarray:
    """57-dim feature vector. See module docstring for layout."""
    sr = TARGET_SR

    # Peak-normalize for feature computation so loudness doesn't influence
    # spectral/timbral feature scaling. The original audio is unchanged.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-6:
        audio_norm = (audio / peak * 0.95).astype(np.float32, copy=False)
    else:
        audio_norm = audio

    # MFCC: 13 means + 13 stds + 13 delta means
    mfcc = librosa.feature.mfcc(y=audio_norm, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds = mfcc.std(axis=1)
    if mfcc.shape[1] >= 9:
        mfcc_delta = librosa.feature.delta(mfcc)
    else:
        mfcc_delta = np.zeros_like(mfcc)
    mfcc_delta_means = mfcc_delta.mean(axis=1)

    # Chroma (12 dims): pitch class energy distribution
    chroma = librosa.feature.chroma_stft(y=audio_norm, sr=sr)
    chroma_means = chroma.mean(axis=1)

    # Spectral & temporal scalars
    centroid = librosa.feature.spectral_centroid(y=audio_norm, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio_norm, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_norm, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=audio_norm)
    zcr = librosa.feature.zero_crossing_rate(y=audio_norm)

    scalars = np.array([
        float(centroid.mean()),
        float(centroid.std()),
        float(rolloff.mean()),
        float(bandwidth.mean()),
        float(flatness.mean()),
        float(zcr.mean()),
    ], dtype=np.float32)

    vec = np.concatenate([
        mfcc_means.astype(np.float32, copy=False),
        mfcc_stds.astype(np.float32, copy=False),
        mfcc_delta_means.astype(np.float32, copy=False),
        chroma_means.astype(np.float32, copy=False),
        scalars,
    ])
    # 13 + 13 + 13 + 12 + 6 = 57
    return vec.astype(np.float32, copy=False)


def _normalize_features(segments: list[Segment]) -> np.ndarray:
    X = np.stack([s.features for s in segments]).astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std


def _dedup_similar_segments(
    segments: list[Segment],
    distance_threshold: float = DEDUP_DISTANCE,
) -> list[Segment]:
    """Drop the shorter of any consecutive pair of segments with very similar
    features.

    Window for "consecutive" is the longest trimmed segment's duration, so a
    pair gets compared only if their onsets are within one full sample length
    of each other. Within that window, if their z-score-normalized feature
    vectors are within distance_threshold, the shorter audio is dropped.

    This is the step that catches double-triggered transients: same physical
    hit, two onsets, two segments that look near-identical in feature space.
    Truly different consecutive hits sit much further apart in feature space
    (typical inter-cluster distance is 5–10) so they both survive.
    """
    if len(segments) < 2:
        return segments

    Xn = _normalize_features(segments)
    order = sorted(range(len(segments)), key=lambda i: segments[i].onset_time)
    # Window = longest trimmed sample duration so the comparison reach scales
    # with the actual material.
    sr = TARGET_SR
    longest_s = max(s.audio.size / sr for s in segments)
    keep = [True] * len(segments)

    for k in range(len(order)):
        idx_i = order[k]
        if not keep[idx_i]:
            continue
        for k2 in range(k + 1, len(order)):
            idx_j = order[k2]
            if not keep[idx_j]:
                continue
            dt = segments[idx_j].onset_time - segments[idx_i].onset_time
            if dt > longest_s:
                break
            dist = float(np.linalg.norm(Xn[idx_i] - Xn[idx_j]))
            if dist < distance_threshold:
                # Drop the shorter of the two.
                if segments[idx_j].audio.size < segments[idx_i].audio.size:
                    keep[idx_j] = False
                else:
                    keep[idx_i] = False
                    break   # idx_i is gone — stop comparing its pair

    return [s for i, s in enumerate(segments) if keep[i]]


def cluster_segments(segments: list[Segment], n_clusters: int) -> list[Segment]:
    if not segments:
        return segments
    n_clusters = max(1, min(n_clusters, len(segments)))
    Xn = _normalize_features(segments)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(Xn)
    for seg, lab in zip(segments, labels):
        seg.cluster = int(lab)
    return segments


def cluster_segments_auto(segments: list[Segment], threshold: float) -> list[Segment]:
    if not segments:
        return segments
    if len(segments) == 1:
        segments[0].cluster = 0
        return segments
    Xn = _normalize_features(segments)
    ac = AgglomerativeClustering(
        n_clusters=None, distance_threshold=float(threshold), linkage="ward"
    )
    labels = ac.fit_predict(Xn)
    for seg, lab in zip(segments, labels):
        seg.cluster = int(lab)
    return segments


def cluster_segments_hdbscan(segments: list[Segment], min_cluster_size: int) -> list[Segment]:
    """Density-based clustering.

    Configuration choices that make HDBSCAN actually sensitive on noisy
    57-dim audio features:

    - cluster_selection_method='leaf' (overrides HDBSCAN's 'eom' default).
      EOM (excess of mass) prefers fewer, larger clusters; leaf takes the
      leaves of the cluster tree, which gives finer-grained, more
      homogeneous clusters.
    - min_samples=1 keeps density requirements permissive.
    - cluster_selection_epsilon=0: no force-merge of close clusters.

    Noise points get their OWN clusters (via a separate KMeans pass on
    just the noise subset), not reassigned to existing centroids.
    Reassigning was the previous behavior and it was bad: "nearest" in
    57-dim space does not mean "similar," so noise points contaminated
    the coherent clusters HDBSCAN found. Now noise stays quarantined in
    its own buckets.

    If HDBSCAN labels more than 70% of points as noise we treat that as
    "couldn't find structure" and fall back to plain KMeans rather than
    poison the few real clusters.
    """
    if not segments:
        return segments
    if len(segments) < 2:
        segments[0].cluster = 0
        return segments

    Xn = _normalize_features(segments)
    n = len(segments)
    # Cap mcs at half the dataset — asking for clusters larger than that
    # guarantees almost everything ends up as noise.
    mcs = max(2, min(int(min_cluster_size), max(2, n // 2)))

    h = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=1,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.0,
    )
    labels = h.fit_predict(Xn).astype(np.int64)

    unique = {int(l) for l in labels} - {-1}
    n_noise = int(np.sum(labels == -1))
    noise_frac = n_noise / n if n > 0 else 1.0

    # Couldn't find structure → fall back to KMeans.
    if not unique or noise_frac > 0.7:
        fallback_k = max(2, min(n // 5, 16))
        km = KMeans(n_clusters=fallback_k, n_init=10, random_state=42)
        labels = km.fit_predict(Xn)
        for seg, lab in zip(segments, labels):
            seg.cluster = int(lab)
        return segments

    # Cluster noise points among themselves with KMeans, assigning fresh
    # cluster IDs starting after HDBSCAN's highest. Noise stays separated
    # from the real clusters; similar noise points stick together.
    if n_noise > 0:
        noise_idx = np.where(labels == -1)[0]
        next_id = max(unique) + 1
        if n_noise <= 2:
            for i, idx in enumerate(noise_idx):
                labels[idx] = next_id + i
        else:
            noise_k = max(2, min(n_noise // 4, 16))
            km = KMeans(n_clusters=noise_k, n_init=5, random_state=42)
            noise_labels = km.fit_predict(Xn[noise_idx])
            for i, idx in enumerate(noise_idx):
                labels[idx] = next_id + int(noise_labels[i])

    for seg, lab in zip(segments, labels):
        seg.cluster = int(lab)
    return segments


def _normalize_cluster_loudness(by_cluster: dict[int, list[Segment]]) -> None:
    """Per-cluster gain so every key plays at consistent loudness.

    Scales each cluster so its loudest segment hits TARGET_RMS, capped at
    PEAK_LIMIT to prevent clipping. Within-cluster relative loudness is
    preserved (velocity layering still works).
    """
    TARGET_RMS = 0.20      # roughly -14 dBFS
    PEAK_LIMIT = 0.95
    GAIN_CAP = 30.0        # don't blow up basically-silent clusters

    for segs in by_cluster.values():
        if not segs:
            continue
        max_rms = max(s.rms for s in segs)
        if max_rms < 1e-6:
            continue
        gain = TARGET_RMS / max_rms
        # Peak-limit check using actual peak (transients can spike well above RMS)
        cluster_peak = max(float(np.max(np.abs(s.audio))) for s in segs)
        if cluster_peak > 1e-6 and cluster_peak * gain > PEAK_LIMIT:
            gain = PEAK_LIMIT / cluster_peak
        gain = min(gain, GAIN_CAP)
        for s in segs:
            s.audio = (s.audio * gain).astype(np.float32, copy=False)
            s.rms = float(np.sqrt(np.mean(s.audio.astype(np.float64) ** 2)))


def assign_keys_to_clusters(
    segments: list[Segment],
    sort_by_freq: bool = True,
    n_keys: int = MAX_KEYS,
) -> Instrument:
    """Group segments by cluster id, order clusters, assign to keyboard keys.

    Pure — no mutation. Called once during pipeline run, and again by the GUI
    when the user toggles the pitch-sort checkbox.
    """
    instrument = Instrument()
    if not segments:
        return instrument

    by_cluster: dict[int, list[Segment]] = {}
    for s in segments:
        by_cluster.setdefault(s.cluster, []).append(s)

    for cid in by_cluster:
        by_cluster[cid].sort(key=lambda s: s.rms)

    if sort_by_freq:
        def order_key(cid: int) -> float:
            vals = [s.dominant_freq for s in by_cluster[cid]]
            return float(np.median(vals)) if vals else 0.0
    else:
        def order_key(cid: int) -> float:
            return float(cid)

    sorted_cluster_ids = sorted(by_cluster.keys(), key=order_key)
    for key_idx, cid in enumerate(sorted_cluster_ids[:n_keys]):
        instrument.notes[key_idx] = by_cluster[cid]
    return instrument


def build_instrument(
    segments: list[Segment],
    n_keys: int = MAX_KEYS,
    sort_by_freq: bool = True,
) -> Instrument:
    """One-time build: normalize cluster loudness (mutates audio), then assign keys."""
    if not segments:
        return Instrument()

    by_cluster: dict[int, list[Segment]] = {}
    for s in segments:
        by_cluster.setdefault(s.cluster, []).append(s)
    _normalize_cluster_loudness(by_cluster)

    return assign_keys_to_clusters(segments, sort_by_freq=sort_by_freq, n_keys=n_keys)


def run_pipeline(
    audio_path: str,
    mode: str = "manual",
    n_clusters: int = 8,
    threshold: float = 5.0,
    min_cluster_size: int = 3,
    sensitivity: float = 0.5,
    segment_length_s: float = SEGMENT_LENGTH_S,
    trim_threshold_db: float = TRIM_THRESHOLD_DB_DEFAULT,
    noise_gate_db: Optional[float] = None,
    n_keys: int = MAX_KEYS,
    progress_callback: Optional[ProgressCallback] = None,
) -> Instrument:
    def report(msg: str, frac: float) -> None:
        if progress_callback is not None:
            progress_callback(msg, frac)

    report("Loading audio…", 0.05)
    audio = load_audio(audio_path)

    report("Detecting onsets…", 0.25)
    onsets = detect_onsets(audio, sensitivity=sensitivity)
    if len(onsets) == 0:
        report("No onsets detected — try raising sensitivity.", 1.0)
        return Instrument()

    report(f"Extracting {len(onsets)} segments…", 0.40)
    segments = extract_segments(
        audio, onsets,
        segment_length_s=segment_length_s,
        trim_threshold_db=trim_threshold_db,
        noise_gate_db=noise_gate_db,
    )
    if not segments:
        report("No usable segments — all were silent, too short, or below the gate.", 1.0)
        return Instrument()

    n_before_dedup = len(segments)
    segments = _dedup_similar_segments(segments)
    n_dedup = n_before_dedup - len(segments)
    if n_dedup > 0:
        report(f"Deduped {n_dedup} similar segments…", 0.55)

    n_kept = len(segments)
    n_dropped = len(onsets) - n_kept

    if mode == "hdbscan":
        report(f"HDBSCAN on {n_kept} segments (min size {min_cluster_size})…", 0.70)
        cluster_segments_hdbscan(segments, min_cluster_size)
    elif mode == "auto":
        report(f"Agglomerative on {n_kept} segments (threshold {threshold:.1f})…", 0.70)
        cluster_segments_auto(segments, threshold)
    else:
        n_eff = max(1, min(n_clusters, n_kept))
        report(f"KMeans on {n_kept} segments (k={n_eff})…", 0.70)
        cluster_segments(segments, n_eff)

    n_clusters_found = len({s.cluster for s in segments})
    report("Building instrument…", 0.90)
    instrument = build_instrument(segments, n_keys=n_keys)
    n_loaded = len(instrument.loaded_notes())
    n_octaves = instrument.max_octave() + 1 if n_loaded else 0

    parts = [f"{n_kept} segments"]
    if n_dropped > 0:
        bits = []
        if n_dropped - n_dedup > 0:
            bits.append(f"{n_dropped - n_dedup} silent/gated")
        if n_dedup > 0:
            bits.append(f"{n_dedup} dup")
        parts.append("(" + ", ".join(bits) + " dropped)")
    parts.append(f"→ {n_clusters_found} clusters → {n_loaded} keys")
    parts.append(f"across {n_octaves} octave{'s' if n_octaves != 1 else ''}.")
    report(" ".join(parts), 1.0)
    return instrument
