"""Pretrained audio embedding backends for Samplebot-3000 (brief §2).

An `Embedder` turns one audio slice into a fixed-length vector that replaces (or
augments) the hand-crafted 57-dim feature vector. Backends are swappable so we
can A/B them in the LOMO harness; pick the best by macro-F1 and keep it.

The frozen embedding feeds the SAME sklearn heads as today (classifier.make_model)
— we are only swapping the representation, not fine-tuning.

Heavy deps (torch/transformers) are imported lazily inside each backend so this
module — and the app — load fine when no embedder is in use.

Backends:
  clap  : LAION CLAP via HuggingFace transformers (512-dim). Installs cleanly on
          arm64/py3.12 and doubles as the §5 zero-shot text backend. Implemented.
  passt : PaSST (hear21passt) — TODO, brief's default recommendation.
  openl3: OpenL3 (needs TensorFlow) — TODO.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np


def _as_tensor(out):
    """transformers >=5 returns BaseModelOutputWithPooling from get_*_features,
    where .pooler_output is the projected (+L2-normalized) embedding; older
    versions returned the tensor directly. Accept either."""
    if hasattr(out, "pooler_output"):
        return out.pooler_output
    return out


class Embedder(ABC):
    """audio slice -> fixed-length embedding vector."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """One slice -> (dim,) float32 embedding."""

    def embed_batch(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        """Default: loop. Backends override for real batching speedups."""
        return np.stack([self.embed(a, sr) for a in audios]).astype(np.float32)


class ClapEmbedder(Embedder):
    """LAION CLAP audio embedding (512-dim) via transformers.

    CLAP expects 48 kHz mono. We resample, pad very short one-shots up to a
    minimum length (drum hits are << CLAP's window), and pool the model's audio
    projection to one vector per slice. Embeddings are L2-normalized so a linear
    head sees a cosine-friendly space.
    """

    name = "clap"
    dim = 512
    SR = 48_000
    MODEL_ID = "laion/clap-htsat-unfused"
    MIN_SECONDS = 1.0  # pad shorter slices to the model's minimum temporal support

    def __init__(self, device: str | None = None, l2_normalize: bool = True):
        self._device = device
        self._l2 = l2_normalize
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import ClapModel, ClapProcessor

        if self._device is None:
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = ClapModel.from_pretrained(self.MODEL_ID).to(self._device).eval()
        self._processor = ClapProcessor.from_pretrained(self.MODEL_ID)

    def _prep(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        if sr != self.SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SR)
        min_len = int(self.MIN_SECONDS * self.SR)
        if audio.size < min_len:
            audio = np.pad(audio, (0, min_len - audio.size))
        return audio

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return self.embed_batch([audio], sr)[0]

    def embed_batch(self, audios: list[np.ndarray], sr: int) -> np.ndarray:
        import torch
        self._ensure_loaded()
        prepped = [self._prep(a, sr) for a in audios]
        # transformers >=5 renamed the CLAP processor's audio kwarg audios->audio.
        inputs = self._processor(
            audio=prepped, sampling_rate=self.SR, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_audio_features(**inputs)
        feats = _as_tensor(out).float().cpu().numpy()
        if self._l2:
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.clip(norms, 1e-8, None)
        return feats.astype(np.float32)

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """Text embeddings for §5 zero-shot (cosine vs audio embedding)."""
        import torch
        self._ensure_loaded()
        inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.get_text_features(**inputs)
        feats = _as_tensor(out).float().cpu().numpy()
        if self._l2:
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            feats = feats / np.clip(norms, 1e-8, None)
        return feats.astype(np.float32)


_BACKENDS = {"clap": ClapEmbedder}


@lru_cache(maxsize=None)
def get_embedder(name: str = "clap") -> Embedder:
    """Cached singleton per backend name (loads the model once)."""
    name = name.lower()
    if name not in _BACKENDS:
        raise ValueError(f"unknown embedder {name!r} (have {list(_BACKENDS)})")
    return _BACKENDS[name]()


def embedder_available() -> bool:
    """True if the optional `[clap]` extras (torch + transformers) are installed.

    Uses find_spec so it never imports the heavy libraries — cheap enough to
    call at GUI build time to decide whether CLAP-backed models can run.
    """
    import importlib.util
    return all(importlib.util.find_spec(m) is not None
               for m in ("torch", "transformers"))
