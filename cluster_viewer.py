"""Cluster viewer dialog.

Shows each loaded key's segments with audition buttons and a waveform
preview. Lets the user hear (and see) exactly what's inside each cluster
and which segment is the representative. With multi-octave clusters, key
labels include the octave offset.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame, QSizePolicy,
)

from pipeline import Instrument, Segment, NOTE_NAMES_12, TARGET_SR
from audio_engine import AudioEngine


WAVEFORM_COLOR = QColor("#ff8a1e")
WAVEFORM_COLOR_REP = QColor("#7fe6dc")
WAVEFORM_BG = QColor("#0b0907")
WAVEFORM_CENTERLINE = QColor("#2c2114")


class WaveformWidget(QWidget):
    """Min/max-binned waveform rendered as a filled polygon.

    For audio longer than its pixel width, each pixel column shows the
    vertical span between the bin's min and max sample, giving a faithful
    envelope view at any zoom level. For audio shorter than the width,
    samples are stretched.
    """

    def __init__(
        self,
        audio: Optional[np.ndarray],
        color: QColor = WAVEFORM_COLOR,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._audio = audio
        self._color = color
        self.setFixedHeight(28)
        self.setMinimumWidth(160)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), WAVEFORM_BG)

        w = max(1, self.width())
        h = self.height()
        center_y = h / 2.0

        # Subtle center line
        p.setPen(QPen(WAVEFORM_CENTERLINE, 1))
        p.drawLine(0, int(center_y), w, int(center_y))

        if self._audio is None or self._audio.size == 0:
            return

        n = int(self._audio.size)
        scale = (h / 2.0) - 1.0

        if n >= w:
            bin_size = n // w
            usable_n = bin_size * w
            reshaped = self._audio[:usable_n].reshape(w, bin_size)
            bin_mins = reshaped.min(axis=1)
            bin_maxs = reshaped.max(axis=1)
        else:
            idx = np.linspace(0, n - 1, w).astype(int)
            bin_mins = self._audio[idx]
            bin_maxs = self._audio[idx]

        bin_mins = np.clip(bin_mins, -1.0, 1.0)
        bin_maxs = np.clip(bin_maxs, -1.0, 1.0)

        # Build a single closed polygon: top edge (max values) left→right,
        # then bottom edge (min values) right→left.
        pts = []
        for i in range(w):
            pts.append(QPointF(float(i), center_y - float(bin_maxs[i]) * scale))
        for i in range(w - 1, -1, -1):
            pts.append(QPointF(float(i), center_y - float(bin_mins[i]) * scale))

        p.setBrush(QBrush(self._color))
        p.setPen(Qt.NoPen)
        p.drawPolygon(QPolygonF(pts))


def _key_label(cluster_idx: int) -> str:
    octave = cluster_idx // 12
    pos = cluster_idx % 12
    name = NOTE_NAMES_12[pos]
    if octave == 0:
        return name
    return f"{name} +{octave}"


class ClusterViewerDialog(QDialog):
    def __init__(
        self,
        instrument: Instrument,
        engine: AudioEngine,
        get_gain: Callable[[], float],
        labels: Optional[dict[int, str]] = None,
        confidences: Optional[dict[int, float]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Cluster Viewer")
        self.resize(700, 680)
        self._instrument = instrument
        self._engine = engine
        self._get_gain = get_gain
        self._labels = labels or {}
        self._confidences = confidences or {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        loaded = sorted(self._instrument.loaded_notes())
        total_segments = sum(len(self._instrument.notes[k]) for k in loaded)
        n_octaves = self._instrument.max_octave() + 1 if loaded else 0

        header = QLabel(
            f"{len(loaded)} keys · {total_segments} segments · "
            f"{n_octaves} octave{'s' if n_octaves != 1 else ''}"
        )
        header.setObjectName("dialogHeader")
        root.addWidget(header)

        subhead = QLabel(
            "Click ▶ to hear a segment. Velocity layering: low velocity plays "
            "the top of each list (quietest), high velocity plays the bottom (loudest). "
            "Representative segments are highlighted."
        )
        subhead.setWordWrap(True)
        subhead.setObjectName("dialogSubtle")
        root.addWidget(subhead)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(10)

        if not loaded:
            empty = QLabel("(no clusters loaded — run the analysis first)")
            empty.setObjectName("dialogSubtle")
            col.addWidget(empty)
        else:
            for key_idx in loaded:
                col.addWidget(self._cluster_group(key_idx))

        col.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("CLOSE")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

    def _cluster_group(self, key_idx: int) -> QWidget:
        segs = self._instrument.notes[key_idx]
        rep = self._instrument.representative(key_idx)

        group = QFrame()
        group.setObjectName("clusterGroup")
        v = QVBoxLayout(group)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(4)

        n = len(segs)
        title_html = (
            f"<span style='color:#ffb15a'>KEY {_key_label(key_idx).upper()}</span>"
            f"<span style='color:#666'> · {n} segment{'s' if n != 1 else ''}</span>"
        )
        label = self._labels.get(key_idx)
        if label:
            conf = self._confidences.get(key_idx)
            conf_txt = f" {conf*100:.0f}%" if conf is not None else ""
            title_html += (
                f"<span style='color:#7fe6dc'> · {label.upper()}{conf_txt}</span>"
            )
        title = QLabel(title_html)
        title.setObjectName("clusterTitle")
        v.addWidget(title)

        for seg in segs:
            v.addWidget(self._segment_row(seg, is_rep=(seg is rep)))

        return group

    def _segment_row(self, seg: Segment, is_rep: bool) -> QWidget:
        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        play_btn = QPushButton("▶")
        play_btn.setObjectName("playBtn")
        play_btn.setFixedWidth(34)
        play_btn.clicked.connect(lambda checked=False, s=seg: self._play_segment(s))
        h.addWidget(play_btn)

        wave_color = WAVEFORM_COLOR_REP if is_rep else WAVEFORM_COLOR
        waveform = WaveformWidget(seg.audio, color=wave_color)
        waveform.setFixedWidth(220)
        h.addWidget(waveform)

        duration_s = seg.audio.size / float(TARGET_SR) if seg.audio is not None else 0.0
        text = f"t={seg.onset_time:6.2f}s · rms={seg.rms:.3f} · dur={duration_s:.2f}s"
        if is_rep:
            text = f"<span style='color:#7fe6dc'><b>{text} · rep</b></span>"
        lbl = QLabel(text)
        lbl.setObjectName("segmentRow")
        h.addWidget(lbl, 1)

        return row

    def _play_segment(self, seg: Segment) -> None:
        # No note_id — audition voices play through naturally, the global
        # ADSR still shapes them, but they're never released.
        self._engine.play(seg.audio, gain=self._get_gain(), note_id=-1)
