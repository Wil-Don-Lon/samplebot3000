"""Small ADSR envelope visualizer.

Draws the attack/decay/sustain/release curve as a line, sized to fit the
controls next to it. Update with set_adsr(a, d, s, r); auto-repaints.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPainterPath, QLinearGradient
from PySide6.QtWidgets import QWidget


# Theme accent — keep in sync with gui.py (spaceage Moog amber)
ACCENT = QColor("#ff8a1e")
ACCENT_DIM = QColor(255, 138, 30, 40)
GRID = QColor("#2c2114")
BG = QColor("#0b0907")


class ADSREnvelopeWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.setMaximumHeight(90)
        # Default reasonable shape
        self._a = 0.001
        self._d = 0.001
        self._s = 1.0
        self._r = 0.05

    def set_adsr(self, a: float, d: float, s: float, r: float) -> None:
        self._a, self._d, self._s, self._r = a, d, s, r
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), BG)

        w = self.width()
        h = self.height()
        pad_x = 8
        pad_y = 8
        plot_w = w - 2 * pad_x
        plot_h = h - 2 * pad_y

        # Light grid
        grid_pen = QPen(GRID, 1)
        p.setPen(grid_pen)
        for frac in (0.25, 0.5, 0.75):
            y = pad_y + plot_h * frac
            p.drawLine(pad_x, int(y), w - pad_x, int(y))

        # Use a visible sustain slice so the curve shows the shape clearly.
        sustain_slice = max(0.05, 0.20)
        total = self._a + self._d + sustain_slice + self._r
        if total <= 0:
            return

        a_w = (self._a / total) * plot_w
        d_w = (self._d / total) * plot_w
        s_w = (sustain_slice / total) * plot_w
        r_w = (self._r / total) * plot_w

        # Y axis: 0 at bottom, 1 at top
        def y_at(level: float) -> float:
            return pad_y + plot_h * (1.0 - level)

        x0 = pad_x
        x1 = x0 + a_w
        x2 = x1 + d_w
        x3 = x2 + s_w
        x4 = x3 + r_w

        # Filled area underneath
        fill = QPainterPath()
        fill.moveTo(x0, y_at(0))
        fill.lineTo(x0, y_at(0))
        fill.lineTo(x1, y_at(1.0))
        fill.lineTo(x2, y_at(self._s))
        fill.lineTo(x3, y_at(self._s))
        fill.lineTo(x4, y_at(0))
        fill.lineTo(x0, y_at(0))

        grad = QLinearGradient(0, pad_y, 0, pad_y + plot_h)
        grad.setColorAt(0, ACCENT_DIM)
        grad.setColorAt(1, QColor(74, 243, 243, 0))
        p.fillPath(fill, grad)

        # Stroke the envelope
        path = QPainterPath()
        path.moveTo(x0, y_at(0))
        path.lineTo(x1, y_at(1.0))
        path.lineTo(x2, y_at(self._s))
        path.lineTo(x3, y_at(self._s))
        path.lineTo(x4, y_at(0))

        pen = QPen(ACCENT, 1.5)
        p.setPen(pen)
        p.drawPath(path)
