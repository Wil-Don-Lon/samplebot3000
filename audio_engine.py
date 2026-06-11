"""Polyphonic real-time audio mixer with ADSR envelope + master lowpass filter.

Architecture:
- One OutputStream for the app lifetime.
- Each voice has its own envelope state machine (Attack -> Decay -> Sustain ->
  Release -> Done). The audio callback advances envelopes sample-by-sample and
  applies them to the playing audio.
- Voices carry a `note_id` (typically the keyboard cluster index). release_note
  finds matching voices and transitions them to the release phase.
- After mixing all voices, a Butterworth lowpass filter is applied to the
  master output. The filter is stateful across callback blocks for continuity.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.signal import lfilter


# Envelope phases
_ATTACK = 0
_DECAY = 1
_SUSTAIN = 2
_RELEASE = 3
_DONE = -1


@dataclass
class _Voice:
    audio: np.ndarray
    pos: int = 0
    gain: float = 1.0
    note_id: int = -1
    loop: bool = False              # if True, wrap pos back to 0 when audio ends
    env_phase: int = _ATTACK
    env_t: int = 0                  # samples elapsed in current phase
    env_release_start: float = 1.0  # envelope level when release began


class AudioEngine:
    def __init__(self, sample_rate: int = 44100, block_size: int = 256) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._voices: list[_Voice] = []
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None

        # ADSR — seconds for A/D/R, level [0,1] for S
        self._a: float = 0.001
        self._d: float = 0.001
        self._s: float = 1.0
        self._r: float = 0.05

        # When True, new voices loop their audio buffer while held.
        # Captured per-voice at play() time so toggling it doesn't affect
        # currently-playing notes.
        self._loop_enabled: bool = False

        # Master lowpass — cascaded one-pole IIR.
        # Each stage's state is its last output sample, which stays meaningful
        # when alpha (cutoff) changes between blocks. That's what kills the
        # scratchiness you get with Butterworth/SOS when you turn the knob.
        self._lpf_cutoff_hz: float = float(sample_rate) / 2.0
        self._lpf_n_stages = 4   # cascade for ~24 dB/oct rolloff
        self._lpf_state: list[np.ndarray] = [
            np.zeros(1, dtype=np.float64) for _ in range(self._lpf_n_stages)
        ]
        # Per-pole cutoff is higher than the labeled cutoff so the cascade's
        # -3 dB point matches the slider value. For N=4: factor ≈ 2.3
        self._lpf_compensation = 1.0 / float(np.sqrt(2.0 ** (1.0 / self._lpf_n_stages) - 1.0))

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
            with self._lock:
                self._voices.clear()

    # ---------- controls ----------

    def set_adsr(self, attack_s: float, decay_s: float, sustain: float, release_s: float) -> None:
        with self._lock:
            self._a = max(0.0, float(attack_s))
            self._d = max(0.0, float(decay_s))
            self._s = max(0.0, min(1.0, float(sustain)))
            self._r = max(0.0, float(release_s))

    def set_loop_enabled(self, enabled: bool) -> None:
        """Affects voices created after this call; already-playing voices keep
        whatever loop flag they had when they started."""
        with self._lock:
            self._loop_enabled = bool(enabled)

    def set_lpf_cutoff(self, hz: float) -> None:
        with self._lock:
            nyq = self.sample_rate / 2.0
            self._lpf_cutoff_hz = max(20.0, min(nyq, float(hz)))
            # No state reset, no coefficient precomputation. Coefficients are
            # recomputed cheaply each callback from the current cutoff; state
            # persists, so the transition is continuous.

    def _rebuild_lpf(self) -> None:
        # No-op retained for backward compatibility — state is now persistent.
        return

    # ---------- triggers ----------

    def play(self, audio: np.ndarray, gain: float = 1.0, note_id: int = -1) -> None:
        """Non-blocking, thread-safe: append a fresh voice with envelope in ATTACK."""
        if audio is None:
            return
        a = np.asarray(audio)
        if a.size == 0:
            return
        if a.dtype != np.float32:
            a = a.astype(np.float32, copy=False)
        if a.ndim == 2:
            a = a.mean(axis=1).astype(np.float32, copy=False)
        elif a.ndim != 1:
            a = a.flatten().astype(np.float32, copy=False)
        with self._lock:
            v = _Voice(
                audio=a, pos=0, gain=float(gain), note_id=int(note_id),
                loop=self._loop_enabled,
            )
            self._voices.append(v)

    def release_note(self, note_id: int) -> None:
        """Trigger release phase on all voices matching note_id."""
        if note_id < 0:
            return
        with self._lock:
            for v in self._voices:
                if v.note_id == note_id and v.env_phase not in (_RELEASE, _DONE):
                    v.env_release_start = self._current_env_level(v)
                    v.env_phase = _RELEASE
                    v.env_t = 0

    def panic(self) -> None:
        with self._lock:
            self._voices.clear()

    # ---------- envelope math ----------

    def _current_env_level(self, v: _Voice) -> float:
        """Caller holds lock."""
        sr = self.sample_rate
        if v.env_phase == _ATTACK:
            a_s = max(1, int(self._a * sr))
            return min(1.0, v.env_t / a_s)
        if v.env_phase == _DECAY:
            d_s = max(1, int(self._d * sr))
            return 1.0 + (self._s - 1.0) * min(1.0, v.env_t / d_s)
        if v.env_phase == _SUSTAIN:
            return self._s
        if v.env_phase == _RELEASE:
            r_s = max(1, int(self._r * sr))
            return v.env_release_start * max(0.0, 1.0 - v.env_t / r_s)
        return 0.0

    def _voice_envelope_block(self, v: _Voice, n: int) -> np.ndarray:
        """Compute n envelope samples, advance v's phase/t. Caller holds lock."""
        out = np.zeros(n, dtype=np.float32)
        idx = 0
        sr = self.sample_rate
        a_s = max(1, int(self._a * sr))
        d_s = max(1, int(self._d * sr))
        r_s = max(1, int(self._r * sr))
        s_level = self._s

        while idx < n and v.env_phase != _DONE:
            if v.env_phase == _ATTACK:
                if self._a <= 0:
                    v.env_phase = _DECAY
                    v.env_t = 0
                    continue
                take = min(n - idx, max(0, a_s - v.env_t))
                if take > 0:
                    idxs = v.env_t + np.arange(take, dtype=np.float32)
                    out[idx:idx + take] = idxs / a_s
                    idx += take
                    v.env_t += take
                if v.env_t >= a_s:
                    v.env_phase = _DECAY
                    v.env_t = 0
            elif v.env_phase == _DECAY:
                if self._d <= 0:
                    v.env_phase = _SUSTAIN
                    v.env_t = 0
                    continue
                take = min(n - idx, max(0, d_s - v.env_t))
                if take > 0:
                    idxs = v.env_t + np.arange(take, dtype=np.float32)
                    out[idx:idx + take] = 1.0 + (s_level - 1.0) * (idxs / d_s)
                    idx += take
                    v.env_t += take
                if v.env_t >= d_s:
                    v.env_phase = _SUSTAIN
                    v.env_t = 0
            elif v.env_phase == _SUSTAIN:
                take = n - idx
                out[idx:idx + take] = s_level
                idx += take
                v.env_t += take
            elif v.env_phase == _RELEASE:
                if self._r <= 0:
                    v.env_phase = _DONE
                    break
                take = min(n - idx, max(0, r_s - v.env_t))
                if take > 0:
                    idxs = v.env_t + np.arange(take, dtype=np.float32)
                    out[idx:idx + take] = v.env_release_start * np.maximum(
                        0.0, 1.0 - idxs / r_s
                    )
                    idx += take
                    v.env_t += take
                if v.env_t >= r_s:
                    v.env_phase = _DONE
                    break
        return out

    # ---------- audio callback ----------

    def _callback(self, outdata, frames, time_info, status) -> None:
        mix = np.zeros(frames, dtype=np.float32)
        with self._lock:
            still_active: list[_Voice] = []
            for v in self._voices:
                if v.env_phase == _DONE:
                    continue

                # Produce `frames` samples for this voice. If the audio buffer
                # runs out mid-block, either wrap (loop=True) or terminate the
                # voice (loop=False). The envelope timer keeps advancing in
                # wall-clock samples regardless of any audio wraps.
                produced = 0
                while produced < frames and v.env_phase != _DONE:
                    remaining_audio = len(v.audio) - v.pos
                    if remaining_audio <= 0:
                        if v.loop:
                            v.pos = 0
                            remaining_audio = len(v.audio)
                        else:
                            v.env_phase = _DONE
                            break

                    n = min(frames - produced, remaining_audio)
                    env = self._voice_envelope_block(v, n)
                    mix[produced:produced + n] += v.audio[v.pos:v.pos + n] * v.gain * env
                    v.pos += n
                    produced += n

                if v.env_phase != _DONE:
                    still_active.append(v)
            self._voices = still_active

            # Master lowpass — cascaded one-pole IIR. Coefficients computed
            # from current cutoff each block; state preserved across blocks.
            nyq = self.sample_rate / 2.0
            if self._lpf_cutoff_hz < nyq * 0.98:
                fc_pole = min(self._lpf_cutoff_hz * self._lpf_compensation, nyq * 0.99)
                alpha = 1.0 - float(np.exp(-2.0 * np.pi * fc_pole / self.sample_rate))
                alpha = max(0.0, min(1.0, alpha))
                b = np.array([alpha], dtype=np.float64)
                a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
                out = mix.astype(np.float64, copy=False)
                for stage in range(self._lpf_n_stages):
                    out, self._lpf_state[stage] = lfilter(
                        b, a, out, zi=self._lpf_state[stage]
                    )
                mix = out.astype(np.float32, copy=False)

        np.clip(mix, -1.0, 1.0, out=mix)
        outdata[:, 0] = mix
