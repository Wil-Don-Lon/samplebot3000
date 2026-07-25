"""Skeuomorphic Moog-style control widgets: rotary Knob, ToggleSwitch, LED,
and a clickable 16-step sequencer grid.

Knob and ToggleSwitch are intentionally API-compatible with the QSlider /
QCheckBox they replace (setRange/value/setValue/valueChanged ;
isChecked/setChecked/toggled) so they drop into the existing GUI wiring.
"""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient, QConicalGradient,
    QLinearGradient, QFont, QMouseEvent,
)
from PySide6.QtWidgets import QWidget, QSizePolicy


AMBER = QColor("#ff8a1e")
AMBER_HI = QColor("#ffd29a")
LED_RED = QColor("#ff3b1e")
LED_OFF = QColor("#3a1410")
METAL_HI = QColor("#cfc8bd")
METAL_LO = QColor("#3a352e")
PANEL = QColor("#15110b")
EDGE = QColor("#4a3820")


class Knob(QWidget):
    """Rotary knob. Drag vertically (or wheel) to change. Drop-in for QSlider:
    setRange / setValue / value / valueChanged(int)."""

    valueChanged = Signal(int)

    # Pointer sweep, in degrees, measured from straight-down.
    _SWEEP = 135.0

    def __init__(self, parent: Optional[QWidget] = None, diameter: int = 40) -> None:
        super().__init__(parent)
        self._min = 0
        self._max = 100
        self._value = 0
        self._diameter = diameter
        self._drag_y: Optional[float] = None
        self._drag_v0 = 0
        self.setFixedSize(diameter + 6, diameter + 6)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # QSlider-compatible API ------------------------------------------------
    def setRange(self, lo: int, hi: int) -> None:
        self._min, self._max = int(lo), int(hi)
        self.setValue(self._value)

    def setValue(self, v: int) -> None:
        v = int(max(self._min, min(self._max, v)))
        if v != self._value:
            self._value = v
            self.update()
            self.valueChanged.emit(v)
        else:
            self._value = v
            self.update()

    def value(self) -> int:
        return self._value

    # interaction -----------------------------------------------------------
    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.LeftButton:
            self._drag_y = e.position().y()
            self._drag_v0 = self._value

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._drag_y is None:
            return
        # 180 px of vertical travel spans the full range; up = increase.
        span = self._max - self._min
        delta = (self._drag_y - e.position().y()) / 180.0 * span
        self.setValue(int(round(self._drag_v0 + delta)))

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._drag_y = None

    def wheelEvent(self, e) -> None:
        step = 1 if (self._max - self._min) <= 130 else 4
        self.setValue(self._value + (step if e.angleDelta().y() > 0 else -step))

    # paint -----------------------------------------------------------------
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(3, 3, self._diameter, self._diameter)
        cx, cy = rect.center().x(), rect.center().y()
        rad = self._diameter / 2.0

        frac = 0.0
        if self._max > self._min:
            frac = (self._value - self._min) / (self._max - self._min)

        # Tick ring
        p.setPen(QPen(QColor("#6a5638"), 1.4))
        for i in range(11):
            a = math.radians(-90 - self._SWEEP + (2 * self._SWEEP) * (i / 10.0))
            r0, r1 = rad + 1, rad + 3.5
            p.drawLine(QPointF(cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
                       QPointF(cx + r1 * math.cos(a), cy + r1 * math.sin(a)))

        # Brushed-metal body
        g = QRadialGradient(cx - rad * 0.3, cy - rad * 0.4, rad * 1.6)
        g.setColorAt(0.0, METAL_HI)
        g.setColorAt(0.55, QColor("#8a8276"))
        g.setColorAt(1.0, METAL_LO)
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor("#1a1610"), 1.5))
        p.drawEllipse(rect)

        # Inner dish
        inner = rect.adjusted(rad * 0.42, rad * 0.42, -rad * 0.42, -rad * 0.42)
        p.setBrush(QBrush(QColor("#2a251e")))
        p.setPen(Qt.NoPen)
        p.drawEllipse(inner)

        # Pointer
        ang = math.radians(-90 - self._SWEEP + (2 * self._SWEEP) * frac)
        p.setPen(QPen(AMBER, 2.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QPointF(cx + rad * 0.30 * math.cos(ang),
                           cy + rad * 0.30 * math.sin(ang)),
                   QPointF(cx + rad * 0.86 * math.cos(ang),
                           cy + rad * 0.86 * math.sin(ang)))


class ToggleSwitch(QWidget):
    """Two-state rocker switch. Drop-in for QCheckBox:
    isChecked / setChecked / toggled(bool)."""

    toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None, checked: bool = False) -> None:
        super().__init__(parent)
        self._checked = checked
        self._enabled = True
        self.setFixedSize(50, 24)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, b: bool) -> None:
        b = bool(b)
        if b != self._checked:
            self._checked = b
            self.update()
            self.toggled.emit(b)

    def setEnabled(self, b: bool) -> None:  # keep QWidget behaviour + repaint
        self._enabled = bool(b)
        super().setEnabled(bool(b))
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if self._enabled and e.button() == Qt.LeftButton:
            self.setChecked(not self._checked)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        track = QRectF(1, 4, 48, 16)
        on = self._checked and self._enabled
        p.setBrush(QBrush(QColor("#23170c") if not on else QColor("#5a3208")))
        p.setPen(QPen(EDGE, 1))
        p.drawRoundedRect(track, 8, 8)
        # knob
        kx = 26 if self._checked else 2
        kr = QRectF(kx, 2, 22, 20)
        g = QLinearGradient(kr.topLeft(), kr.bottomLeft())
        g.setColorAt(0.0, METAL_HI if self._enabled else QColor("#6a6358"))
        g.setColorAt(1.0, METAL_LO)
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor("#1a1610"), 1))
        p.drawRoundedRect(kr, 5, 5)
        if on:
            p.setBrush(QBrush(AMBER))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(kr.center().x() - 2.5, kr.center().y() - 2.5, 5, 5))


class LED(QWidget):
    """Small glowing indicator. setOn(bool)."""

    def __init__(self, parent: Optional[QWidget] = None,
                 color: QColor = LED_RED, size: int = 14) -> None:
        super().__init__(parent)
        self._on = False
        self._color = color
        self._size = size
        self.setFixedSize(size + 4, size + 4)

    def setOn(self, on: bool) -> None:
        on = bool(on)
        if on != self._on:
            self._on = on
            self.update()

    def isOn(self) -> bool:
        return self._on

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(2, 2, self._size, self._size)
        if self._on:
            glow = QRadialGradient(r.center(), self._size)
            glow.setColorAt(0.0, self._color.lighter(140))
            glow.setColorAt(0.5, self._color)
            glow.setColorAt(1.0, self._color.darker(220))
            p.setBrush(QBrush(glow))
            p.setPen(QPen(self._color.darker(300), 1))
        else:
            p.setBrush(QBrush(LED_OFF))
            p.setPen(QPen(QColor("#1a0c08"), 1))
        p.drawEllipse(r)
