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

SEGMENT_LENGTH_S = 0.5            # default PLAYBACK capture length (user-adjustable)

# Fixed window the CLASSIFIER features are always computed over, decoupled from
# the user's "Sample Len" slider. A trained model is a fixed artifact; if the
# classify feature window tracked the slider, moving it would feed the model
# out-of-distribution slices (train/serve skew). Training (features_from_oneshot)
# and classification serving both use THIS value, so they can never drift. Must
# match the window the saved models/*.joblib were trained on. Clustering mode is
# self-consistent per run, so it keeps using the slider, not this.
CLASSIFY_FEATURE_LEN_S = 0.5
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
    label: str = ""              # predicted class (classification mode)
    playback: object = None      # per-sample PlaybackSettings (set by the GUI)
    # The fixed-window slice the classifier features were computed on (classify
    # mode only). Embedding-based classifiers re-embed THIS, not `audio` (the
    # variable-length playback slice), to stay train/serve-consistent.
    classify_audio: Optional[np.ndarray] = None


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
    # Wider, more perceptible range: low sensitivity = high delta (only strong
    # transients), high sensitivity = near-zero delta (catches subtle hits).
    delta = 0.22 - 0.21 * sensitivity        # 0.22 .. 0.01
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


def low_band_f0(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    fmin: float = 40.0,
    fmax: float = 400.0,
) -> float:
    """Resonant fundamental in the drum body's low band (Hz), or 0.0.

    Unlike `dominant_frequency` (which finds the lowest strong peak anywhere),
    this peak-picks the magnitude spectrum strictly within [fmin, fmax] — the
    band where a tom/kick's membrane fundamental lives. This is the feature that
    separates high/mid/low toms, which differ by pitch, not timbre, and which
    MFCCs and semantic embeddings (CLAP) cannot resolve.

    The first ~5 ms of broadband attack is skipped so the transient doesn't
    swamp the sustained resonance.
    """
    if audio.size < 256:
        return 0.0
    skip = min(int(0.005 * sr), audio.size // 4)  # drop the attack click
    body = audio[skip:]
    n_fft = 8192
    n = min(body.size, n_fft)
    windowed = body[:n].astype(np.float64) * np.hanning(n)
    if n < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[:n] = windowed
        windowed = padded
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return 0.0
    band_spec = spec[band]
    if float(band_spec.max()) < 1e-10:
        return 0.0
    return float(freqs[band][int(np.argmax(band_spec))])


def _hz_to_midi(f0: float) -> float:
    """Log-pitch (MIDI note number) for f0>0, else 0. Log scaling makes the
    hi/mid/lo tom spacing roughly linear, which classifiers separate better."""
    if f0 <= 0.0:
        return 0.0
    return 69.0 + 12.0 * np.log2(f0 / 440.0)


def pitch_temporal_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Compact 5-dim scalar vector to fuse onto an audio embedding (brief §3).

    Targets what timbre embeddings miss: pitch (the tom problem) and gross
    amplitude-envelope shape.
      [0] f0_hz            low-band resonant fundamental
      [1] f0_midi          log-scaled pitch (linear hi/mid/lo spacing)
      [2] log_attack_time  log seconds from onset to envelope peak
      [3] temporal_centroid  envelope centre of mass (s) — decay character
      [4] decay_slope      slope of the log-envelope after the peak (1/s, <=0)
    """
    if audio.size < 256:
        return np.zeros(5, dtype=np.float32)

    f0 = low_band_f0(audio, sr)
    f0_midi = _hz_to_midi(f0)

    # Smoothed amplitude envelope.
    env = np.abs(audio).astype(np.float64)
    if env.size > TRIM_SMOOTH_SAMPLES:
        pad = TRIM_SMOOTH_SAMPLES // 2
        kernel = np.ones(TRIM_SMOOTH_SAMPLES) / TRIM_SMOOTH_SAMPLES
        env = np.convolve(np.pad(env, pad, mode="edge"), kernel, "valid")[:audio.size]
    peak_idx = int(np.argmax(env))
    peak = float(env[peak_idx]) or 1e-9

    log_attack_time = float(np.log10(max(peak_idx / sr, 1e-4)))

    t = np.arange(env.size) / sr
    esum = float(env.sum()) or 1e-9
    temporal_centroid = float((t * env).sum() / esum)

    # Decay slope: linear fit of log-envelope from peak to end (negative = decays).
    decay = env[peak_idx:]
    if decay.size > 8:
        log_decay = np.log(np.clip(decay / peak, 1e-6, None))
        td = t[peak_idx:peak_idx + decay.size] - t[peak_idx]
        decay_slope = float(np.polyfit(td, log_decay, 1)[0])
    else:
        decay_slope = 0.0

    return np.array(
        [f0, f0_midi, log_attack_time, temporal_centroid, decay_slope],
        dtype=np.float32,
    )


def _tail_end_index(
    chunk: np.ndarray,
    threshold_db: float,
) -> int:
    """Index where the sample's tail falls below threshold and stays there.

    Walks the smoothed envelope from the transient (start of chunk) forward and
    returns the last index that is still at or above the trim threshold + 1.
    Unlike trim_silence this never trims the *front* — the chunk already starts
    at the transient, so we only decide where the decay ends.
    """
    if chunk.size == 0:
        return 0
    threshold_linear = 10.0 ** (float(threshold_db) / 20.0)
    abs_audio = np.abs(chunk).astype(np.float64)
    if abs_audio.size > TRIM_SMOOTH_SAMPLES:
        pad = TRIM_SMOOTH_SAMPLES // 2
        padded = np.pad(abs_audio, pad, mode="edge")
        kernel = np.ones(TRIM_SMOOTH_SAMPLES, dtype=np.float64) / TRIM_SMOOTH_SAMPLES
        smoothed = np.convolve(padded, kernel, mode="valid")[:abs_audio.size]
    else:
        smoothed = abs_audio
    above = np.where(smoothed >= threshold_linear)[0]
    if len(above) == 0:
        return 0
    return int(above[-1]) + 1


def normalize_sample(audio: np.ndarray, peak_target: float = 0.95) -> np.ndarray:
    """Peak-normalize a single sample so its loudest point hits peak_target."""
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-9:
        return audio
    return (audio * (peak_target / peak)).astype(np.float32, copy=False)


def segment_features(
    chunk: np.ndarray,
    trim_threshold_db: float = TRIM_THRESHOLD_DB_DEFAULT,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """THE canonical 'audio slice -> (processed sample, 57-dim features)' path.

    This is the single implementation shared by the live pipeline AND classifier
    training, so train-time and serve-time features cannot drift (train/serve
    skew). `chunk` must already start at the transient and be capped to the
    desired maximum length (the caller applies the segment-length cap / clip-at-
    next-onset). Here we tail-trim, peak-normalize, and featurize — identically
    for both paths.

    Returns (sample, features), or None if the slice is shorter than the minimum
    usable length before or after tail-trimming.
    """
    if chunk.size < MIN_SEGMENT_SAMPLES:
        return None
    tail = _tail_end_index(chunk, trim_threshold_db)
    if tail < MIN_SEGMENT_SAMPLES:
        return None
    sample = normalize_sample(chunk[:tail])
    return sample, compute_features(sample)


def features_from_oneshot(
    audio: np.ndarray,
    trim_threshold_db: float = TRIM_THRESHOLD_DB_DEFAULT,
    feature_length_s: float = CLASSIFY_FEATURE_LEN_S,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Push a full one-shot through the SAME segmentation classification serving
    uses, so train-time and serve-time classifier features cannot drift.

    Training bins hold clean full-length one-shots, but at inference the model
    only sees onset-aligned, length-capped, tail-trimmed, peak-normalized slices.
    The cap here is the FIXED CLASSIFY_FEATURE_LEN_S (not the user's playback
    Sample Len) — see that constant. Front-trim to the transient, cap, then run
    the shared `segment_features`. Returns (sample, features) or None.
    """
    audio = trim_silence(audio, trim_threshold_db)
    cap = int(feature_length_s * TARGET_SR)
    return segment_features(audio[:cap], trim_threshold_db)


def extract_segments(
    audio: np.ndarray,
    onset_times: np.ndarray,
    segment_length_s: float = SEGMENT_LENGTH_S,
    trim_threshold_db: float = TRIM_THRESHOLD_DB_DEFAULT,
    noise_gate_db: Optional[float] = None,
    clip_at_next_onset: bool = False,
    feature_length_s: Optional[float] = None,
) -> list[Segment]:
    """Grab one normalized sample per transient, per the data-pipeline spec:

    For each detected onset t_i (already filtered to those above the noise gate
    upstream, but re-checked here):
      1. Start exactly at the transient t_i.
      2. End at whichever comes first:
           - the tail falling below the trim threshold,
           - the sample-length cap (t_i + L),
           - the next transient t_{i+1}  *only if clip_at_next_onset is on*.
      3. Drop if the chunk's peak is below the noise gate.
      4. Drop if shorter than MIN_SEGMENT_S.
      5. Normalize each surviving sample to a common peak.
      6. Exact-bytes dedupe as a safety net.

    `feature_length_s`: if given (classification mode passes CLASSIFY_FEATURE_LEN_S),
    the stored feature vector is computed over a FIXED window from the onset,
    independent of the playback sample's length — so the slider can't reintroduce
    train/serve skew. The played `audio` still uses the user's segment_length_s.
    If None (clustering mode), features come from the played sample, as before.
    """
    segment_samples = int(segment_length_s * TARGET_SR)
    segments: list[Segment] = []
    seen_hashes: set[int] = set()

    gate_linear = (10.0 ** (float(noise_gate_db) / 20.0)
                   if noise_gate_db is not None else 0.0)

    for i, t in enumerate(onset_times):
        start = int(t * TARGET_SR)
        # Hard upper bound = sample-length cap.
        end = start + segment_samples

        # Optional clipper: also stop at the next transient if it lands sooner.
        if clip_at_next_onset and i + 1 < len(onset_times):
            next_start = int(onset_times[i + 1] * TARGET_SR)
            if next_start < end:
                end = next_start

        chunk = audio[start:end]
        if chunk.size < MIN_SEGMENT_SAMPLES:
            continue

        # Noise gate: drop the whole transient if it never gets loud enough.
        # (Serving-only — training keeps every labeled sample.)
        peak = float(np.abs(chunk).max())
        if peak < gate_linear:
            continue

        # Tail-trim, normalize, and featurize via the shared train/serve path.
        # `sample`/`feats` describe the played slice (user's segment_length_s).
        result = segment_features(chunk, trim_threshold_db)
        if result is None:
            continue
        sample, feats = result

        # Classification: recompute the feature vector over the FIXED window so
        # it matches training regardless of the playback length. Ignores
        # clip-at-next (training doesn't clip). Falls back to the played slice's
        # features if the fixed window is unusable. Also stash the fixed-window
        # slice so embedding classifiers can re-embed it consistently.
        classify_audio = None
        if feature_length_s is not None:
            fcap = int(feature_length_s * TARGET_SR)
            fresult = segment_features(audio[start:start + fcap], trim_threshold_db)
            if fresult is not None:
                classify_audio, feats = fresult

        # Exact-bytes guard against perfect duplicates.
        h = hash(sample.tobytes())
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        rms = float(np.sqrt(np.mean(sample.astype(np.float64) ** 2)))
        dom_freq = dominant_frequency(sample)
        segments.append(
            Segment(
                audio=sample,
                onset_time=float(t),
                features=feats,
                rms=rms,
                dominant_freq=dom_freq,
                classify_audio=classify_audio,
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


def _normalize_features(segments: list[Segment],
                        feature_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """Z-score the clustering features. By default uses each segment's 57-dim
    hand-crafted vector; pass `feature_matrix` (e.g. CLAP+pitch embeddings) to
    cluster in a different space instead."""
    if feature_matrix is None:
        X = np.stack([s.features for s in segments]).astype(np.float64)
    else:
        X = np.asarray(feature_matrix, dtype=np.float64)
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


def cluster_segments(segments: list[Segment], n_clusters: int,
                     feature_matrix: Optional[np.ndarray] = None) -> list[Segment]:
    if not segments:
        return segments
    n_clusters = max(1, min(n_clusters, len(segments)))
    Xn = _normalize_features(segments, feature_matrix)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(Xn)
    for seg, lab in zip(segments, labels):
        seg.cluster = int(lab)
    return segments


def cluster_segments_auto(segments: list[Segment], threshold: float,
                          feature_matrix: Optional[np.ndarray] = None) -> list[Segment]:
    if not segments:
        return segments
    if len(segments) == 1:
        segments[0].cluster = 0
        return segments
    Xn = _normalize_features(segments, feature_matrix)
    ac = AgglomerativeClustering(
        n_clusters=None, distance_threshold=float(threshold), linkage="ward"
    )
    labels = ac.fit_predict(Xn)
    for seg, lab in zip(segments, labels):
        seg.cluster = int(lab)
    return segments


def cluster_segments_hdbscan(segments: list[Segment], min_cluster_size: int,
                             feature_matrix: Optional[np.ndarray] = None) -> list[Segment]:
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

    Xn = _normalize_features(segments, feature_matrix)
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


# Fixed drum-machine keyboard layout across one visible octave (17 keys, C..E).
# (key_index, class, cap label). Classes with several keys (snare, hats, fx)
# get their samples spread round-robin across those keys for variety.
DRUM_KEYMAP: list[tuple[int, str, str]] = [
    (0,  "kick",   "KICK"),
    (1,  "snare",  "SNR1"),
    (2,  "snare",  "SNR2"),
    (3,  "clap",   "CLAP"),
    (4,  "snare",  "SNR3"),
    (5,  "lotom",  "LO-T"),
    (6,  "hats",   "HH1"),
    (7,  "midtom", "MID-T"),
    (8,  "hats",   "HH2"),
    (9,  "hitom",  "HI-T"),
    (10, "hats",   "HH3"),
    (11, "fx",     "FX1"),
    (12, "fx",     "FX2"),
    (13, "crash",  "CRASH"),
    (14, "fx",     "FX3"),
    (15, "ride",   "RIDE"),
    (16, "fx",     "FX4"),
]
# class -> ordered list of key indices it occupies
CLASS_KEYS: dict[str, list[int]] = {}
for _k, _c, _lab in DRUM_KEYMAP:
    CLASS_KEYS.setdefault(_c, []).append(_k)
# key index -> cap label, for the GUI
KEY_CAP_LABEL: dict[int, str] = {k: lab for k, _c, lab in DRUM_KEYMAP}


def classify_feature_matrix(segments: list[Segment], feature_type: str) -> np.ndarray:
    """Feature matrix for a classifier, matching how its model was trained.

    "hand"       : the 57-dim hand-crafted vector already on each segment.
    "fused_clap" : CLAP audio embedding ++ pitch/temporal scalars (brief §2+§3),
                   computed on the fixed-window `classify_audio` slice so it
                   stays train/serve-consistent (§1). The heavy embedder is
                   imported lazily, so clustering users never pay for torch.
    """
    if feature_type == "fused_clap":
        from embedders import get_embedder
        emb = get_embedder("clap")
        audios = [s.classify_audio if s.classify_audio is not None else s.audio
                  for s in segments]
        E = emb.embed_batch(audios, TARGET_SR)
        P = np.stack([pitch_temporal_features(a) for a in audios])
        return np.hstack([E, P]).astype(np.float64)
    return np.stack([s.features for s in segments]).astype(np.float64)


def _classify_ensemble(segments: list[Segment], members: list) -> None:
    """Majority-vote consensus across several models (brief-style ensemble).

    `members` is a list of (fitted_pipeline, feature_type). Each model votes its
    predicted label per segment; the most-voted label wins. Ties break toward the
    earlier member (callers order the strongest model first). Feature matrices are
    computed once per distinct feature_type and reused, so the CLAP embedding pass
    runs at most once.
    """
    from collections import Counter
    ballots: list[list[str]] = [[] for _ in segments]
    feat_cache: dict[str, np.ndarray] = {}
    for pipe, ftype in members:
        if ftype not in feat_cache:
            feat_cache[ftype] = classify_feature_matrix(segments, ftype)
        preds = pipe.predict(feat_cache[ftype])
        for i, p in enumerate(preds):
            ballots[i].append(str(p))
    for seg, votes in zip(segments, ballots):
        seg.label = Counter(votes).most_common(1)[0][0]


def classify_segments(segments: list[Segment], model,
                      feature_type: str = "hand") -> list[Segment]:
    """Tag each segment with its predicted class label (stored on seg.label).
    The fixed-layout key placement happens in build_instrument_drumkeys.

    `model` is any fitted sklearn classifier exposing predict() over the feature
    matrix selected by `feature_type`. Special case: feature_type=="ensemble"
    means `model` is a list of (pipeline, feature_type) members to majority-vote.
    Kept model-agnostic on purpose — pipeline.py must not import classifier.py
    (that would be circular); the GUI passes the loaded model(s) in.
    """
    if not segments:
        return segments
    if feature_type == "ensemble":
        _classify_ensemble(segments, model)
        return segments
    X = classify_feature_matrix(segments, feature_type)
    preds = model.predict(X)
    for seg, p in zip(segments, preds):
        seg.label = str(p)
    return segments


def cluster_consensus(segments: list[Segment], model,
                      feature_matrix: np.ndarray) -> dict[int, tuple[str, float]]:
    """Type each CLUSTER by SOFT CONSENSUS and return {cluster_id: (label, conf)}.

    For each cluster, average the classifier's per-slice class probabilities and
    take the argmax — so the cluster is typed as a whole (robust to a few noisy
    per-slice predictions). `conf` is that mean winning probability, used by hard
    placement to rank which cluster best deserves a contested key. Mutates
    seg.label.
    """
    if not segments:
        return {}
    X = feature_matrix
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_)
    else:  # hard fallback: one-hot the discrete predictions
        preds = model.predict(X)
        classes = np.asarray(sorted(set(map(str, preds))))
        idx_of = {c: i for i, c in enumerate(classes)}
        proba = np.zeros((len(preds), len(classes)), dtype=float)
        for i, p in enumerate(preds):
            proba[i, idx_of[str(p)]] = 1.0

    by_cluster: dict[int, list[int]] = {}
    for i, s in enumerate(segments):
        by_cluster.setdefault(s.cluster, []).append(i)
    info: dict[int, tuple[str, float]] = {}
    for cid, rows in by_cluster.items():
        mean = proba[rows].mean(axis=0)
        j = int(mean.argmax())
        label = str(classes[j])
        info[cid] = (label, float(mean[j]))
        for i in rows:
            segments[i].label = label
    return info


def build_instrument_drumkeys_hard(
    segments: list[Segment],
    cluster_info: dict[int, tuple[str, float]],
    n_keys: int = MAX_KEYS,
) -> Instrument:
    """Place clusters on the drum kit ONE PER KEY — no doubling up.

    Each drum key holds at most one cluster. Clusters are placed in order of
    confidence: the most confident cluster claims the first free key of its type
    (the "most kick-like cluster" really does win KICK). When a type's keys are
    all taken, that cluster overflows to the nearest remaining empty drum key
    rather than stacking — so distinct clusters stay on distinct keys.
    """
    if not segments:
        return Instrument()
    clusters: dict[int, list[Segment]] = {}
    for s in segments:
        clusters.setdefault(s.cluster, []).append(s)

    drum_keys = [k for k, _c, _l in DRUM_KEYMAP]          # 0..16, layout order
    taken: set[int] = set()
    inst = Instrument()
    # Most confident clusters get first pick of their type's keys.
    for cid in sorted(clusters, key=lambda c: cluster_info.get(c, ("", 0.0))[1],
                      reverse=True):
        label = cluster_info.get(cid, ("", 0.0))[0]
        prefs = [k for k in CLASS_KEYS.get(label, []) if k not in taken]
        if prefs:
            key = prefs[0]
        else:
            spill = [k for k in drum_keys if k not in taken]
            if not spill:
                continue  # kit full (>17 clusters) — drop the overflow
            key = spill[0]
        taken.add(key)
        segs = sorted(clusters[cid], key=lambda s: s.rms)
        for s in segs:
            s.cluster = key
        if 0 <= key < n_keys:
            inst.notes[key] = segs
    return inst


def build_instrument_drumkeys_by_cluster(
    segments: list[Segment], n_keys: int = MAX_KEYS) -> Instrument:
    """Place whole CLUSTERS onto the drum kit by their consensus type.

    Unlike build_instrument_drumkeys (which round-robins individual slices and
    so mixes clusters), this keeps each cluster intact: every cluster lands as a
    unit on a key of its type. When a type has several clusters, they spread
    across that type's keys (largest cluster first), so e.g. two snare-ish
    clusters take SNR1 and SNR2 rather than being blended.
    """
    if not segments:
        return Instrument()
    clusters: dict[int, list[Segment]] = {}
    for s in segments:
        clusters.setdefault(s.cluster, []).append(s)

    # Group clusters by their (shared) consensus label.
    by_label: dict[str, list[list[Segment]]] = {}
    for segs in clusters.values():
        by_label.setdefault(segs[0].label, []).append(segs)

    by_key: dict[int, list[Segment]] = {}
    for label, clist in by_label.items():
        keys = CLASS_KEYS.get(label)
        if not keys:
            continue  # type with no key on this layout — dropped
        clist.sort(key=len, reverse=True)  # biggest cluster takes the first key
        for i, segs in enumerate(clist):
            k = keys[i % len(keys)]
            for s in segs:
                s.cluster = k
            by_key.setdefault(k, []).extend(segs)

    inst = Instrument()
    for k, segs in by_key.items():
        if 0 <= k < n_keys:
            segs.sort(key=lambda s: s.rms)
            inst.notes[k] = segs
    return inst


def build_instrument_drumkeys(segments: list[Segment], n_keys: int = MAX_KEYS) -> Instrument:
    """Lay predicted segments onto the fixed drum keyboard. Each class's samples
    are distributed round-robin across the key(s) assigned to that class (e.g.
    snares spread over SNR1/SNR2/SNR3), so every key holds a cyclable set."""
    if not segments:
        return Instrument()
    by_class: dict[str, list[Segment]] = {}
    for s in segments:
        by_class.setdefault(s.label, []).append(s)

    by_key: dict[int, list[Segment]] = {}
    for cls, segs in by_class.items():
        keys = CLASS_KEYS.get(cls)
        if not keys:
            continue  # class with no key on this layout — dropped
        for i, s in enumerate(segs):
            k = keys[i % len(keys)]
            s.cluster = k
            by_key.setdefault(k, []).append(s)

    # No cluster-loudness pass: samples are peak-normalized individually at
    # extraction, so they already play back at a consistent level.
    inst = Instrument()
    for k, segs in by_key.items():
        if 0 <= k < n_keys:
            segs.sort(key=lambda s: s.rms)
            inst.notes[k] = segs
    return inst


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
    """One-time build → assign keys. Samples are already peak-normalized
    individually at extraction, so no cluster-loudness pass is needed."""
    if not segments:
        return Instrument()
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
    classifier_model=None,
    classifier_feature_type: str = "hand",
    cluster_clap_mode: str = "off",   # off | soft | hard (clustering placement)
    clip_at_next_onset: bool = False,
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
        clip_at_next_onset=clip_at_next_onset,
        # Classification features come from a fixed window so the Sample Len
        # slider can't reintroduce train/serve skew; clustering uses the slider.
        feature_length_s=(CLASSIFY_FEATURE_LEN_S if mode == "classify" else None),
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

    use_clap = cluster_clap_mode in ("soft", "hard")
    cluster_info = None  # set when CLAP types clusters for drum-kit placement
    if mode == "classify":
        report(f"Classifying {n_kept} segments…", 0.70)
        classify_segments(segments, classifier_model, classifier_feature_type)
    else:
        # Clustering. Optionally cluster in CLAP-embedding space (richer timbre
        # grouping) instead of the 57-dim hand-crafted features.
        clap_feats = None
        if use_clap:
            report(f"Embedding {n_kept} segments (CLAP)…", 0.62)
            clap_feats = classify_feature_matrix(segments, "fused_clap")
        space = "CLAP" if use_clap else "57-dim"
        if mode == "hdbscan":
            report(f"HDBSCAN on {n_kept} ({space}, min size {min_cluster_size})…", 0.70)
            cluster_segments_hdbscan(segments, min_cluster_size, clap_feats)
        elif mode == "auto":
            report(f"Agglomerative on {n_kept} ({space}, threshold {threshold:.1f})…", 0.70)
            cluster_segments_auto(segments, threshold, clap_feats)
        else:
            n_eff = max(1, min(n_clusters, n_kept))
            report(f"KMeans on {n_kept} ({space}, k={n_eff})…", 0.70)
            cluster_segments(segments, n_eff, clap_feats)
        # With CLAP + a classifier, type each cluster and lay it on the drum kit
        # (kick-like cluster → kick key, …) instead of generic cluster-per-key.
        if use_clap and classifier_model is not None:
            report("Typing clusters → drum kit…", 0.85)
            cluster_info = cluster_consensus(segments, classifier_model, clap_feats)

    n_clusters_found = len({s.cluster for s in segments})
    report("Building instrument…", 0.90)
    if mode == "classify":
        instrument = build_instrument_drumkeys(segments, n_keys=n_keys)
    elif cluster_info is not None and cluster_clap_mode == "hard":
        instrument = build_instrument_drumkeys_hard(segments, cluster_info, n_keys=n_keys)
    elif cluster_info is not None:  # soft
        instrument = build_instrument_drumkeys_by_cluster(segments, n_keys=n_keys)
    else:
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
    grouping = "classes" if mode == "classify" else "clusters"
    parts.append(f"→ {n_clusters_found} {grouping} → {n_loaded} keys")
    parts.append(f"across {n_octaves} octave{'s' if n_octaves != 1 else ''}.")
    report(" ".join(parts), 1.0)
    return instrument
