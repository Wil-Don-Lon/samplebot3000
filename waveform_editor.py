"""Per-sample waveform editor.

Shows a segment's padded RAW source buffer (with runway before the onset and
after the tail) and lets you drag the two ends of the highlighted region to
retrim or elongate the played/exported slice. Emits the new bounds (indices into
`seg.source`) live while dragging and once more on release; the host applies
them via pipeline.reslice_segment.

Purely a view + input widget — it never mutates the segment itself.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import QWidget

import pipeline as P

# Theme accent — spaceage Moog amber, in sync with gui.py / adsr_widget.py
ACCENT = QColor("#ff8a1e")
ACCENT_SOFT = QColor(255, 138, 30, 70)
WAVE_IN = QColor("#ffb15a")           # waveform inside the kept region
WAVE_OUT = QColor(120, 96, 60, 130)   # waveform in the padding (dimmed)
HANDLE = QColor("#ffd089")
GRID = QColor("#2c2114")
BG = QColor("#0b0907")
TEXT = QColor("#c9b48f")

_HANDLE_GRAB_PX = 10                   # how close a click must be to grab a handle


class WaveformEditor(QWidget):
    """Draggable start/end trim over a segment's padded source waveform."""

    # (start, end) sample indices into seg.source. `boundsChanged` fires live
    # during a drag; `editCommitted` once on mouse release (host re-auditions).
    boundsChanged = Signal(int, int)
    editCommitted = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMouseTracking(True)
        self._seg: Optional[object] = None
        self._env: Optional[np.ndarray] = None   # (cols, 2) min/max per column
        self._n: int = 0                          # source length in samples
        self._sr: int = P.TARGET_SR
        self._start = 0
        self._end = 0
        self._drag: Optional[str] = None          # "start" | "end" | None
        self._hover: Optional[str] = None

    # ---------- public API ----------

    def set_segment(self, seg, sample_rate: int = P.TARGET_SR) -> None:
        """Show `seg` (or clear on None). Builds source if the segment lacks one."""
        if seg is None:
            self._seg = None
            self._env = None
            self._n = 0
            self.update()
            return
        P.ensure_source(seg)
        self._seg = seg
        self._sr = int(sample_rate)
        src = np.asarray(seg.source, dtype=np.float32).ravel()
        self._n = int(src.shape[0])
        self._start = int(getattr(seg, "src_start", 0))
        self._end = int(getattr(seg, "src_end", self._n))
        self._env = self._build_envelope(src)
        self.update()

    def clear(self) -> None:
        self.set_segment(None)

    # ---------- envelope ----------

    def _build_envelope(self, src: np.ndarray, cols: int = 1400) -> np.ndarray:
        """Downsample to per-column (min, max) for fast, faithful drawing."""
        n = src.shape[0]
        if n == 0:
            return np.zeros((1, 2), dtype=np.float32)
        cols = min(cols, n)
        edges = np.linspace(0, n, cols + 1, dtype=np.int64)
        out = np.zeros((cols, 2), dtype=np.float32)
        for i in range(cols):
            a, b = edges[i], edges[i + 1]
            if b <= a:
                b = a + 1
            chunk = src[a:b]
            out[i, 0] = float(chunk.min())
            out[i, 1] = float(chunk.max())
        return out

    # ---------- geometry ----------

    def _plot_rect(self) -> QRectF:
        return QRectF(10, 10, max(1, self.width() - 20), max(1, self.height() - 34))

    def _x_of(self, sample: int) -> float:
        r = self._plot_rect()
        if self._n <= 1:
            return r.left()
        return r.left() + r.width() * (sample / self._n)

    def _sample_of(self, x: float) -> int:
        r = self._plot_rect()
        frac = (x - r.left()) / r.width() if r.width() else 0.0
        return int(round(max(0.0, min(1.0, frac)) * self._n))

    # ---------- painting ----------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), BG)
        r = self._plot_rect()

        if self._seg is None or self._env is None or self._n == 0:
            p.setPen(QPen(TEXT))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "press a key (or ◀ ▶) to edit a sample")
            return

        mid = r.top() + r.height() / 2.0
        half = r.height() / 2.0 - 2

        sx = self._x_of(self._start)
        ex = self._x_of(self._end)

        # Kept-region highlight behind the waveform.
        p.fillRect(QRectF(sx, r.top(), max(1.0, ex - sx), r.height()), ACCENT_SOFT)

        # Waveform: min→max vertical bar per column, bright inside the region.
        cols = self._env.shape[0]
        for i in range(cols):
            cx = r.left() + r.width() * (i + 0.5) / cols
            lo = mid - self._env[i, 1] * half
            hi = mid - self._env[i, 0] * half
            inside = sx <= cx <= ex
            p.setPen(QPen(WAVE_IN if inside else WAVE_OUT, 1))
            p.drawLine(int(cx), int(lo), int(cx), int(hi))

        # Zero line.
        p.setPen(QPen(GRID, 1))
        p.drawLine(int(r.left()), int(mid), int(r.right()), int(mid))

        # Handles.
        for x, which in ((sx, "start"), (ex, "end")):
            lit = self._drag == which or self._hover == which
            p.setPen(QPen(HANDLE, 3 if lit else 2))
            p.drawLine(int(x), int(r.top()), int(x), int(r.bottom()))
            grip = QRectF(x - 3, mid - 9, 6, 18)
            p.fillRect(grip, QBrush(HANDLE))

        # Readout: start offset from the buffer start and slice length.
        p.setPen(QPen(TEXT))
        length_s = (self._end - self._start) / float(self._sr)
        start_s = self._start / float(self._sr)
        p.drawText(int(r.left()), int(self.height() - 8),
                   f"start {start_s:.3f}s")
        p.drawText(QRectF(r.left(), self.height() - 22, r.width(), 16),
                   Qt.AlignRight, f"length {length_s:.3f}s")

    # ---------- mouse ----------

    def _handle_at(self, x: float) -> Optional[str]:
        if self._seg is None:
            return None
        if abs(x - self._x_of(self._start)) <= _HANDLE_GRAB_PX:
            return "start"
        if abs(x - self._x_of(self._end)) <= _HANDLE_GRAB_PX:
            return "end"
        return None

    def mousePressEvent(self, event) -> None:
        if self._seg is None:
            return
        self._drag = self._handle_at(event.position().x())
        if self._drag:
            self.update()

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if self._drag is None:
            hov = self._handle_at(x)
            if hov != self._hover:
                self._hover = hov
                self.setCursor(Qt.SizeHorCursor if hov else Qt.ArrowCursor)
                self.update()
            return
        s = self._sample_of(x)
        if self._drag == "start":
            self._start = max(0, min(s, self._end - P.MIN_SEGMENT_SAMPLES))
        else:
            self._end = min(self._n, max(s, self._start + P.MIN_SEGMENT_SAMPLES))
        self.boundsChanged.emit(self._start, self._end)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is not None:
            self._drag = None
            self.editCommitted.emit(self._start, self._end)
            self.update()
