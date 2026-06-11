"""One-octave piano keyboard widget.

Pure Qt — no audio or pipeline dependencies. Emits note_pressed / note_released
(note index 0..12 within the currently visible octave; the main window handles
octave translation). White keys are 8 equal columns; black keys overlay at 55%
column width and 62% height.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QBrush, QMouseEvent
from PySide6.QtWidgets import QWidget


NOTES = [
    (False, "A"),   # 0  C
    (True,  "W"),   # 1  C#
    (False, "S"),   # 2  D
    (True,  "E"),   # 3  D#
    (False, "D"),   # 4  E
    (False, "F"),   # 5  F
    (True,  "T"),   # 6  F#
    (False, "G"),   # 7  G
    (True,  "Y"),   # 8  G#
    (False, "H"),   # 9  A
    (True,  "U"),   # 10 A#
    (False, "J"),   # 11 B
    (False, "K"),   # 12 C(oct)
    (True,  "O"),   # 13 C#(oct)
    (False, "L"),   # 14 D(oct)
    (True,  "P"),   # 15 D#(oct)
    (False, ";"),   # 16 E(oct)
]

WHITE_KEY_INDICES = [0, 2, 4, 5, 7, 9, 11, 12, 14, 16]   # 10 whites
BLACK_KEY_RIGHT_OF_WHITE = {
    1: 0, 3: 1, 6: 3, 8: 4, 10: 5, 13: 7, 15: 8,
}
N_WHITE_KEYS = 10

BLACK_WIDTH_FRACTION = 0.55
BLACK_HEIGHT_FRACTION = 0.62


# Theme — keep in sync with gui.py / adsr_widget.py
COLOR_BG          = QColor("#0d0d12")
COLOR_WHITE_DEF   = QColor("#e8eef0")
COLOR_WHITE_LOAD  = QColor("#bcd9d9")    # subtle cyan tint
COLOR_WHITE_PRESS = QColor("#4af3f3")
COLOR_BLACK_DEF   = QColor("#16161e")
COLOR_BLACK_LOAD  = QColor("#1f2c30")    # cooler dark
COLOR_BLACK_PRESS = QColor("#4af3f3")
COLOR_BORDER      = QColor("#2a2a35")
COLOR_LABEL_LIGHT = QColor("#e8e8ee")
COLOR_LABEL_DARK  = QColor("#3a3a45")


class PianoKeyboardWidget(QWidget):
    note_pressed = Signal(int)
    note_released = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(140)
        self._pressed: set[int] = set()
        self._loaded: set[int] = set()
        self._mouse_note: Optional[int] = None

    def press_note(self, note_index: int) -> None:
        if note_index in self._pressed:
            return
        self._pressed.add(note_index)
        self.update()
        self.note_pressed.emit(note_index)

    def release_note(self, note_index: int) -> None:
        if note_index not in self._pressed:
            return
        self._pressed.discard(note_index)
        self.update()
        self.note_released.emit(note_index)

    def set_loaded(self, note_indices: set[int]) -> None:
        self._loaded = set(note_indices)
        self.update()

    # ---------- geometry ----------

    def _white_key_rect(self, white_visual_index: int) -> QRectF:
        w = self.width() / float(N_WHITE_KEYS)
        return QRectF(white_visual_index * w, 0, w, self.height())

    def _black_key_rect(self, black_chromatic: int) -> QRectF:
        white_left = BLACK_KEY_RIGHT_OF_WHITE[black_chromatic]
        w = self.width() / float(N_WHITE_KEYS)
        bw = w * BLACK_WIDTH_FRACTION
        bh = self.height() * BLACK_HEIGHT_FRACTION
        cx = (white_left + 1) * w
        return QRectF(cx - bw / 2.0, 0, bw, bh)

    def _note_at(self, x: float, y: float) -> Optional[int]:
        for chromatic in BLACK_KEY_RIGHT_OF_WHITE:
            if self._black_key_rect(chromatic).contains(x, y):
                return chromatic
        for visual_idx, chromatic in enumerate(WHITE_KEY_INDICES):
            if self._white_key_rect(visual_idx).contains(x, y):
                return chromatic
        return None

    # ---------- painting ----------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), COLOR_BG)

        for visual_idx, chromatic in enumerate(WHITE_KEY_INDICES):
            r = self._white_key_rect(visual_idx)
            pressed = chromatic in self._pressed
            loaded = chromatic in self._loaded
            if pressed:
                color = COLOR_WHITE_PRESS
            elif loaded:
                color = COLOR_WHITE_LOAD
            else:
                color = COLOR_WHITE_DEF
            p.setBrush(QBrush(color))
            p.setPen(QPen(COLOR_BORDER, 1.0))
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 2, 2)
            self._draw_label(p, r, NOTES[chromatic][1], COLOR_LABEL_DARK)

        for chromatic in BLACK_KEY_RIGHT_OF_WHITE:
            r = self._black_key_rect(chromatic)
            pressed = chromatic in self._pressed
            loaded = chromatic in self._loaded
            if pressed:
                color = COLOR_BLACK_PRESS
            elif loaded:
                color = COLOR_BLACK_LOAD
            else:
                color = COLOR_BLACK_DEF
            p.setBrush(QBrush(color))
            p.setPen(QPen(QColor("#000"), 1.0))
            p.drawRoundedRect(r, 2, 2)
            label_color = COLOR_LABEL_DARK if pressed else COLOR_LABEL_LIGHT
            self._draw_label(p, r, NOTES[chromatic][1], label_color)

    def _draw_label(self, p: QPainter, r: QRectF, text: str, color: QColor) -> None:
        font = QFont(p.font())
        font.setPointSize(9)
        font.setBold(False)
        p.setFont(font)
        p.setPen(color)
        label_rect = QRectF(r.left(), r.bottom() - 22, r.width(), 18)
        p.drawText(label_rect, Qt.AlignHCenter | Qt.AlignVCenter, text)

    # ---------- mouse ----------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        note = self._note_at(event.position().x(), event.position().y())
        if note is None:
            return
        self._mouse_note = note
        self.press_note(note)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self._mouse_note is not None:
            self.release_note(self._mouse_note)
            self._mouse_note = None
