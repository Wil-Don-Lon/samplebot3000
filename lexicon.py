"""CLAP text-lexicon "semantic fingerprint" features.

Instead of asking CLAP a single yes/no ("does this sound like a kick drum?") and
taking the argmax over 8 drum names (what `pipeline.clap_sort_clusters` does), we
score each sample against a whole LEXICON of descriptor phrases — drum-identity
nouns AND timbral / tonal / envelope adjectives. Each word becomes one dimension:
the cosine similarity of the sample's CLAP audio embedding to that word's CLAP
text embedding. The result is an (N, n_words) "fingerprint" that can be fused with
the acoustic features and fed to a trained sorter.

Mirrors pipeline._drum_text_vectors: the text matrix is embedded once and memoized;
CLAP audio/text embeddings are already L2-normalized by embedders.py, so the dot
product IS cosine.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


# Ordered descriptor lexicon. Phrased as short captions (CLAP text was trained on
# captions, not bare tokens). Each phrase = one fingerprint dimension.
LEXICON: dict[str, list[str]] = {
    # Drum-identity anchors
    "identity": [
        "a kick drum", "a bass drum", "a snare drum", "a rimshot",
        "a cross-stick click", "a hand clap", "a closed hi-hat", "an open hi-hat",
        "a tom drum", "a floor tom", "a crash cymbal", "a ride cymbal",
        "a ride bell", "a cowbell", "a tambourine", "a shaker", "a percussion hit",
    ],
    # Timbral / spectral
    "timbre": [
        "dark", "bright", "deep", "low", "boomy", "subby", "warm", "muffled",
        "metallic", "bell-like", "shimmering", "airy", "crisp",
    ],
    # Noise vs tonality
    "tonality": [
        "noisy", "hissy", "white noise", "tonal", "pitched", "ringing",
        "resonant", "buzzy", "rattly",
    ],
    # Envelope / temporal
    "envelope": [
        "short", "snappy", "punchy", "tight", "clicky", "sustained",
        "long decay", "washy",
    ],
}


@lru_cache(maxsize=1)
def lexicon_words() -> tuple[str, ...]:
    """Flat ordered word list — the fingerprint's columns."""
    return tuple(w for group in LEXICON.values() for w in group)


@lru_cache(maxsize=1)
def _lexicon_text_matrix() -> np.ndarray:
    """(n_words, 512) L2-normalized CLAP text embeddings, memoized once."""
    from embedders import get_embedder
    emb = get_embedder("clap")
    return emb.embed_text(list(lexicon_words())).astype(np.float64)


def lexicon_fingerprint(audio_emb: np.ndarray) -> np.ndarray:
    """(N, 512) L2-normed CLAP audio embeddings -> (N, n_words) cosine fingerprint."""
    A = np.asarray(audio_emb, dtype=np.float64)
    return (A @ _lexicon_text_matrix().T).astype(np.float32)


def fingerprint_from_slices(slices: list[np.ndarray], sr: int) -> np.ndarray:
    """Convenience: CLAP-embed the slices, then fingerprint. (N, n_words)."""
    from embedders import get_embedder
    emb = get_embedder("clap")
    return lexicon_fingerprint(emb.embed_batch(slices, sr))


def n_words() -> int:
    return len(lexicon_words())
