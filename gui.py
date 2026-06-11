"""Main window: integrates pipeline + audio engine + piano widget.

Layout: two columns of controls above a piano keyboard.
- Left column   : source + clustering controls + run
- Right column  : playback controls (velocity, ADSR, LPF, octave)
- Bottom        : keyboard
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QSpinBox, QSlider, QProgressBar,
    QFrame, QSizePolicy, QComboBox, QCheckBox,
)

from pipeline import (
    run_pipeline, assign_keys_to_clusters, Instrument, NOTE_NAMES, MAX_KEYS,
)
from audio_engine import AudioEngine
from piano_widget import PianoKeyboardWidget
from cluster_viewer import ClusterViewerDialog
from adsr_widget import ADSREnvelopeWidget


KEY_TO_NOTE = {
    Qt.Key_A: 0,  Qt.Key_W: 1,  Qt.Key_S: 2,  Qt.Key_E: 3,  Qt.Key_D: 4,
    Qt.Key_F: 5,  Qt.Key_T: 6,  Qt.Key_G: 7,  Qt.Key_Y: 8,  Qt.Key_H: 9,
    Qt.Key_U: 10, Qt.Key_J: 11, Qt.Key_K: 12, Qt.Key_O: 13, Qt.Key_L: 14,
    Qt.Key_P: 15, Qt.Key_Semicolon: 16,
}

# 17 visible keys per octave (C through E in next octave). Octave step is
# 12 semitones, so the high 5 keys overlap with the low 5 keys of the next
# octave — preserves the "C overlap" rule and adds four free overlap notes.
N_VISIBLE_KEYS = 17
OCTAVE_STEP = 12


# ---------- futuristic minimalist stylesheet ----------

STYLESHEET = """
QMainWindow, QDialog {
    background-color: #0d0d12;
}
QWidget {
    color: #c8c8d0;
    font-family: -apple-system, "SF Pro Display", "Segoe UI", system-ui, sans-serif;
    font-size: 12px;
}
QLabel { background: transparent; }

QLabel#title {
    color: #e8e8ee;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 4px;
}
QLabel#subtitle {
    color: #555560;
    font-size: 10px;
    letter-spacing: 2px;
}
QLabel#sectionHeader {
    color: #4af3f3;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 3px;
    padding-top: 4px;
}
QLabel#controlLabel {
    color: #777787;
    font-size: 10px;
    letter-spacing: 1.5px;
}
QLabel#valueLabel {
    color: #4af3f3;
    font-family: "SF Mono", "Menlo", monospace;
    font-size: 11px;
}
QLabel#statusLabel {
    color: #888896;
    font-size: 11px;
}
QLabel#hintLabel {
    color: #555565;
    font-size: 10px;
    letter-spacing: 2px;
}
QLabel#octaveValue {
    color: #4af3f3;
    font-family: "SF Mono", "Menlo", monospace;
    font-size: 13px;
    font-weight: 600;
}

/* Cluster viewer labels */
QLabel#dialogHeader {
    color: #e8e8ee;
    font-size: 11px;
    letter-spacing: 2px;
}
QLabel#dialogSubtle, QLabel#segmentRow {
    color: #888896;
    font-size: 11px;
}
QLabel#segmentRow {
    font-family: "SF Mono", "Menlo", monospace;
}
QLabel#clusterTitle {
    font-size: 11px;
    letter-spacing: 2px;
    padding-bottom: 2px;
}

QPushButton {
    background-color: transparent;
    color: #4af3f3;
    border: 1px solid #2a2a35;
    padding: 7px 16px;
    border-radius: 1px;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 500;
}
QPushButton:hover { border-color: #4af3f3; background-color: rgba(74, 243, 243, 12); }
QPushButton:pressed { background-color: rgba(74, 243, 243, 25); }
QPushButton:disabled { color: #2a2a35; border-color: #1a1a22; }

QPushButton#playBtn {
    color: #4af3f3;
    padding: 2px;
    font-size: 11px;
    letter-spacing: 0;
}
QPushButton#octaveBtn {
    padding: 4px 10px;
    font-size: 11px;
    min-width: 24px;
}

QFrame#clusterGroup {
    background-color: #14141c;
    border: 1px solid #1f1f28;
    border-radius: 2px;
}

QSpinBox, QComboBox {
    background-color: #14141c;
    color: #e8e8ee;
    border: 1px solid #2a2a35;
    padding: 5px 10px;
    border-radius: 1px;
    selection-background-color: #4af3f3;
    selection-color: #0d0d12;
    min-width: 80px;
}
QSpinBox:hover, QComboBox:hover { border-color: #3a3a45; }
QSpinBox::up-button, QSpinBox::down-button {
    background: transparent;
    border: none;
    width: 16px;
}
QComboBox::drop-down { border: none; width: 22px; }
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid #4af3f3;
    margin-right: 7px;
}
QComboBox QAbstractItemView {
    background-color: #14141c;
    color: #e8e8ee;
    border: 1px solid #2a2a35;
    selection-background-color: #4af3f3;
    selection-color: #0d0d12;
    outline: none;
}

QSlider::groove:horizontal {
    border: none;
    background: #1c1c24;
    height: 2px;
    border-radius: 1px;
}
QSlider::handle:horizontal {
    background: #4af3f3;
    border: none;
    width: 10px;
    height: 10px;
    margin: -5px 0;
    border-radius: 5px;
}
QSlider::handle:horizontal:hover { background: #7ff; }
QSlider::sub-page:horizontal { background: #4af3f3; height: 2px; border-radius: 1px; }

QProgressBar {
    background-color: #1c1c24;
    border: none;
    border-radius: 0;
    height: 2px;
    text-align: center;
}
QProgressBar::chunk { background-color: #4af3f3; }

QScrollArea { background-color: transparent; border: none; }
QScrollBar:vertical { background: #0d0d12; width: 6px; margin: 0; }
QScrollBar::handle:vertical {
    background: #2a2a35; border-radius: 3px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #4af3f3; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    background: none; height: 0;
}

QFrame[frameShape="4"] { color: #1c1c24; background: #1c1c24; max-height: 1px; }

QCheckBox { color: #c8c8d0; spacing: 8px; }
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #2a2a35;
    background: transparent;
    border-radius: 1px;
}
QCheckBox::indicator:hover { border-color: #4af3f3; }
QCheckBox::indicator:checked {
    background: #4af3f3;
    border-color: #4af3f3;
    image: none;
}
"""


class PipelineWorker(QObject):
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        audio_path: str,
        mode: str,
        n_clusters: int,
        threshold: float,
        min_cluster_size: int,
        sensitivity: float,
        segment_length_s: float,
        trim_threshold_db: float,
        noise_gate_db,
    ) -> None:
        super().__init__()
        self._audio_path = audio_path
        self._mode = mode
        self._n_clusters = n_clusters
        self._threshold = threshold
        self._min_cluster_size = min_cluster_size
        self._sensitivity = sensitivity
        self._segment_length_s = segment_length_s
        self._trim_threshold_db = trim_threshold_db
        self._noise_gate_db = noise_gate_db

    @Slot()
    def run(self) -> None:
        try:
            inst = run_pipeline(
                self._audio_path,
                mode=self._mode,
                n_clusters=self._n_clusters,
                threshold=self._threshold,
                min_cluster_size=self._min_cluster_size,
                sensitivity=self._sensitivity,
                segment_length_s=self._segment_length_s,
                trim_threshold_db=self._trim_threshold_db,
                noise_gate_db=self._noise_gate_db,
                n_keys=MAX_KEYS,
                progress_callback=lambda msg, frac: self.progress.emit(msg, frac),
            )
            self.finished.emit(inst)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("samplebot-3000")
        self.resize(940, 620)
        self.setFocusPolicy(Qt.StrongFocus)

        self._audio_path: Optional[str] = None
        self._instrument: Instrument = Instrument()
        # Flat list of every clustered segment, cached so the pitch-sort toggle
        # can re-assign keys without re-running the whole pipeline.
        self._all_segments: list = []
        self._octave: int = 0
        self._thread: Optional[QThread] = None
        self._worker: Optional[PipelineWorker] = None

        self._engine = AudioEngine()
        self._engine.start()

        self._build_ui()
        self._sync_adsr_to_engine()
        self._sync_lpf_to_engine()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(14)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("SAMPLEBOT-3000")
        title.setObjectName("title")
        subtitle = QLabel("UNSUPERVISED SAMPLER")
        subtitle.setObjectName("subtitle")
        header_row.addWidget(title)
        header_row.addSpacing(12)
        header_row.addWidget(subtitle)
        header_row.addStretch(1)
        outer.addLayout(header_row)

        outer.addWidget(self._hline())

        # File / run row (full width above the two columns)
        outer.addLayout(self._build_top_row())

        outer.addWidget(self._hline())

        # Two columns
        cols = QHBoxLayout()
        cols.setSpacing(24)
        cols.addLayout(self._build_left_column(), 1)
        cols.addWidget(self._vline())
        cols.addLayout(self._build_right_column(), 1)
        outer.addLayout(cols, 1)

        outer.addWidget(self._hline())

        # Piano
        self.piano = PianoKeyboardWidget()
        self.piano.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.piano.note_pressed.connect(self._on_note_pressed)
        self.piano.note_released.connect(self._on_note_released)
        outer.addWidget(self.piano, 1)

        # Hint
        hint = QLabel("A W S E D F T G Y H U J K O L P ;     ·     Z / X  OCTAVE")
        hint.setObjectName("hintLabel")
        hint.setAlignment(Qt.AlignCenter)
        outer.addWidget(hint)

        self.setStyleSheet(STYLESHEET)

    # ---------- top row (file + run) ----------

    def _build_top_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)

        self.pick_btn = QPushButton("SELECT FILE")
        self.pick_btn.clicked.connect(self._on_pick_file)
        row.addWidget(self.pick_btn)

        self.file_label = QLabel("no file selected")
        self.file_label.setObjectName("statusLabel")
        row.addWidget(self.file_label, 1)

        self.run_btn = QPushButton("RUN ANALYSIS")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run)
        row.addWidget(self.run_btn)

        self.view_btn = QPushButton("VIEW CLUSTERS")
        self.view_btn.setEnabled(False)
        self.view_btn.clicked.connect(self._on_view_clusters)
        row.addWidget(self.view_btn)

        return row

    # ---------- left column (analysis) ----------

    def _build_left_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(10)

        col.addWidget(self._section_header("ANALYSIS"))

        # Cluster mode
        col.addLayout(self._labeled_row(
            "MODE",
            self._make_mode_combo(),
        ))

        # Manual / Auto / HDBSCAN param (only one visible at a time)
        self.manual_row = self._make_manual_row()
        self.auto_row = self._make_auto_row()
        self.hdb_row = self._make_hdbscan_row()
        col.addWidget(self.manual_row)
        col.addWidget(self.auto_row)
        col.addWidget(self.hdb_row)
        self.auto_row.setVisible(False)
        self.hdb_row.setVisible(False)

        # Sample length (affects analysis — requires Run to apply)
        self.length_slider = QSlider(Qt.Horizontal)
        self.length_slider.setRange(1, 50)   # 0.1s .. 5.0s
        self.length_slider.setValue(5)
        self.length_value_lbl = QLabel("0.5 s")
        self.length_value_lbl.setObjectName("valueLabel")
        self.length_value_lbl.setMinimumWidth(50)
        self.length_slider.valueChanged.connect(
            lambda v: self.length_value_lbl.setText(f"{v/10:.1f} s")
        )
        col.addLayout(self._slider_row("SAMPLE LEN", self.length_slider, self.length_value_lbl))

        # Sensitivity
        col.addLayout(self._slider_row(
            "TRANSIENT SENS",
            self._make_sens_slider(),
            self.sens_value_lbl,
        ))

        # Noise gate: peak-based, drops whole segments below this threshold
        self.gate_slider = QSlider(Qt.Horizontal)
        self.gate_slider.setRange(0, 100)
        self.gate_slider.setValue(0)
        self.gate_value_lbl = QLabel("OFF")
        self.gate_value_lbl.setObjectName("valueLabel")
        self.gate_value_lbl.setMinimumWidth(50)
        self.gate_slider.valueChanged.connect(self._on_gate_changed)
        col.addLayout(self._slider_row("NOISE GATE", self.gate_slider, self.gate_value_lbl))

        # Trim threshold: silence-trim edges below this dBFS level
        self.trim_slider = QSlider(Qt.Horizontal)
        self.trim_slider.setRange(0, 100)
        self.trim_slider.setValue(40)
        self.trim_value_lbl = QLabel("-38 dB")
        self.trim_value_lbl.setObjectName("valueLabel")
        self.trim_value_lbl.setMinimumWidth(50)
        self.trim_slider.valueChanged.connect(self._on_trim_changed)
        col.addLayout(self._slider_row("TRIM THRESHOLD", self.trim_slider, self.trim_value_lbl))

        # Progress + status
        col.addWidget(self._progress_bar())
        self.status = QLabel("ready")
        self.status.setObjectName("statusLabel")
        col.addWidget(self.status)

        col.addStretch(1)
        return col

    def _make_mode_combo(self) -> QWidget:
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Manual k (KMeans)", "manual")
        self.mode_combo.addItem("Auto threshold (Agglomerative)", "auto")
        self.mode_combo.addItem("Auto density (HDBSCAN)", "hdbscan")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        return self.mode_combo

    def _make_manual_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel("COUNT")
        label.setObjectName("controlLabel")
        label.setMinimumWidth(110)
        row.addWidget(label)
        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, MAX_KEYS)
        self.cluster_spin.setValue(8)
        row.addWidget(self.cluster_spin)
        row.addStretch(1)
        return w

    def _make_auto_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel("CLUSTER DIVERSITY")
        label.setObjectName("controlLabel")
        label.setMinimumWidth(130)
        row.addWidget(label)
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(50)
        self.thresh_value_lbl = QLabel("50")
        self.thresh_value_lbl.setObjectName("valueLabel")
        self.thresh_value_lbl.setMinimumWidth(50)
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_value_lbl.setText(str(v))
        )
        row.addWidget(self.thresh_slider, 1)
        row.addWidget(self.thresh_value_lbl)
        return w

    def _make_hdbscan_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel("CLUSTER DIVERSITY")
        label.setObjectName("controlLabel")
        label.setMinimumWidth(130)
        row.addWidget(label)
        self.mcs_slider = QSlider(Qt.Horizontal)
        self.mcs_slider.setRange(0, 100)
        self.mcs_slider.setValue(50)
        self.mcs_value_lbl = QLabel("50")
        self.mcs_value_lbl.setObjectName("valueLabel")
        self.mcs_value_lbl.setMinimumWidth(50)
        self.mcs_slider.valueChanged.connect(
            lambda v: self.mcs_value_lbl.setText(str(v))
        )
        row.addWidget(self.mcs_slider, 1)
        row.addWidget(self.mcs_value_lbl)
        return w

    def _make_sens_slider(self) -> QSlider:
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(0, 100)
        self.sens_slider.setValue(50)
        self.sens_value_lbl = QLabel("0.50")
        self.sens_value_lbl.setObjectName("valueLabel")
        self.sens_value_lbl.setMinimumWidth(50)
        self.sens_slider.valueChanged.connect(
            lambda v: self.sens_value_lbl.setText(f"{v/100:.2f}")
        )
        return self.sens_slider

    def _progress_bar(self) -> QProgressBar:
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        return self.progress_bar

    # ---------- right column (playback) ----------

    def _build_right_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(10)

        col.addWidget(self._section_header("PLAYBACK"))

        # Velocity
        self.vel_slider = QSlider(Qt.Horizontal)
        self.vel_slider.setRange(1, 127)
        self.vel_slider.setValue(100)
        self.vel_value_lbl = QLabel("100")
        self.vel_value_lbl.setObjectName("valueLabel")
        self.vel_value_lbl.setMinimumWidth(50)
        self.vel_slider.valueChanged.connect(
            lambda v: self.vel_value_lbl.setText(str(v))
        )
        col.addLayout(self._slider_row("VELOCITY", self.vel_slider, self.vel_value_lbl))

        # Loop while held
        loop_row = QHBoxLayout()
        loop_label = QLabel("LOOP")
        loop_label.setObjectName("controlLabel")
        loop_label.setMinimumWidth(110)
        self.loop_check = QCheckBox("while held")
        self.loop_check.toggled.connect(self._on_loop_toggled)
        loop_row.addWidget(loop_label)
        loop_row.addWidget(self.loop_check, 1)
        col.addLayout(loop_row)

        # Pitch sort: assign clusters to keys by dominant frequency
        sort_row = QHBoxLayout()
        sort_label = QLabel("PITCH SORT")
        sort_label.setObjectName("controlLabel")
        sort_label.setMinimumWidth(110)
        self.sort_check = QCheckBox("low → high")
        self.sort_check.setChecked(True)
        self.sort_check.toggled.connect(self._on_sort_toggled)
        sort_row.addWidget(sort_label)
        sort_row.addWidget(self.sort_check, 1)
        col.addLayout(sort_row)

        # ADSR section
        col.addSpacing(4)
        col.addWidget(self._section_header("ENVELOPE"))

        self.adsr_viz = ADSREnvelopeWidget()
        col.addWidget(self.adsr_viz)

        # ADSR sliders in a compact grid
        adsr_grid = QGridLayout()
        adsr_grid.setHorizontalSpacing(8)
        adsr_grid.setVerticalSpacing(4)

        def make_adsr_slider(default_val: int, fmt_fn):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(default_val)
            value_lbl = QLabel(fmt_fn(default_val))
            value_lbl.setObjectName("valueLabel")
            value_lbl.setMinimumWidth(60)
            slider.valueChanged.connect(lambda v: value_lbl.setText(fmt_fn(v)))
            slider.valueChanged.connect(self._on_adsr_changed)
            return slider, value_lbl

        # Attack: 0..1000ms linear
        self.a_slider, self.a_lbl = make_adsr_slider(
            0, lambda v: f"{int(v * 10)} ms"
        )
        # Decay: 0..1000ms linear
        self.d_slider, self.d_lbl = make_adsr_slider(
            0, lambda v: f"{int(v * 10)} ms"
        )
        # Sustain: 0..1.00
        self.s_slider, self.s_lbl = make_adsr_slider(
            100, lambda v: f"{v/100:.2f}"
        )
        # Release: 0..2000ms linear
        self.r_slider, self.r_lbl = make_adsr_slider(
            5, lambda v: f"{int(v * 20)} ms"
        )

        for row_idx, (name, sl, lb) in enumerate([
            ("A", self.a_slider, self.a_lbl),
            ("D", self.d_slider, self.d_lbl),
            ("S", self.s_slider, self.s_lbl),
            ("R", self.r_slider, self.r_lbl),
        ]):
            tag = QLabel(name)
            tag.setObjectName("controlLabel")
            tag.setMinimumWidth(20)
            adsr_grid.addWidget(tag, row_idx, 0)
            adsr_grid.addWidget(sl, row_idx, 1)
            adsr_grid.addWidget(lb, row_idx, 2)
        col.addLayout(adsr_grid)

        # LPF
        col.addSpacing(4)
        col.addWidget(self._section_header("FILTER"))
        self.lpf_slider = QSlider(Qt.Horizontal)
        self.lpf_slider.setRange(0, 100)
        self.lpf_slider.setValue(100)
        self.lpf_value_lbl = QLabel("20000 Hz")
        self.lpf_value_lbl.setObjectName("valueLabel")
        self.lpf_value_lbl.setMinimumWidth(70)
        self.lpf_slider.valueChanged.connect(
            lambda v: self.lpf_value_lbl.setText(f"{int(self._lpf_from_slider(v))} Hz")
        )
        self.lpf_slider.valueChanged.connect(self._on_lpf_changed)
        col.addLayout(self._slider_row("CUTOFF", self.lpf_slider, self.lpf_value_lbl))

        # Octave
        col.addSpacing(4)
        col.addWidget(self._section_header("OCTAVE"))
        oct_row = QHBoxLayout()
        oct_row.setSpacing(8)
        self.oct_down_btn = QPushButton("◀")
        self.oct_down_btn.setObjectName("octaveBtn")
        self.oct_down_btn.clicked.connect(lambda: self._shift_octave(-1))
        self.oct_up_btn = QPushButton("▶")
        self.oct_up_btn.setObjectName("octaveBtn")
        self.oct_up_btn.clicked.connect(lambda: self._shift_octave(+1))
        self.oct_value_lbl = QLabel("0 / 0")
        self.oct_value_lbl.setObjectName("octaveValue")
        self.oct_value_lbl.setAlignment(Qt.AlignCenter)
        self.oct_value_lbl.setMinimumWidth(80)
        oct_row.addWidget(self.oct_down_btn)
        oct_row.addWidget(self.oct_value_lbl, 1)
        oct_row.addWidget(self.oct_up_btn)
        col.addLayout(oct_row)

        col.addStretch(1)
        return col

    # ---------- small builders ----------

    def _section_header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionHeader")
        return lbl

    def _labeled_row(self, label_text: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setObjectName("controlLabel")
        lbl.setMinimumWidth(110)
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return row

    def _slider_row(self, label_text: str, slider: QSlider, value_lbl: QLabel) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setObjectName("controlLabel")
        lbl.setMinimumWidth(110)
        row.addWidget(lbl)
        row.addWidget(slider, 1)
        row.addWidget(value_lbl)
        return row

    def _hline(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        return line

    def _vline(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: #1c1c24; background: #1c1c24; max-width: 1px;")
        return line

    # ============================================================
    # parameter mappings
    # ============================================================

    @staticmethod
    def _threshold_from_slider(v: int) -> float:
        # Cluster Diversity: high slider = high threshold = few broad clusters
        return 1.0 + 14.0 * (v / 100.0)

    @staticmethod
    def _min_cluster_size_from_slider(v: int) -> int:
        # Cluster Diversity: high slider = larger min size = fewer, broader clusters.
        # Range 2..30; the pipeline caps this at half the dataset size at runtime.
        return max(2, int(round(2 + 28 * (v / 100.0))))

    @staticmethod
    def _volume_gate_db_from_slider(v: int):
        """Slider 0 = OFF (None); 1..100 maps to -60..-6 dB."""
        if v <= 0:
            return None
        return -60.0 + (v - 1) * (54.0 / 99.0)

    @staticmethod
    def _trim_db_from_slider(v: int) -> float:
        """Slider 0..100 maps to -60..-6 dB (always on)."""
        return -60.0 + v * (54.0 / 100.0)

    @staticmethod
    def _lpf_from_slider(v: int) -> float:
        # 20 Hz -> 20000 Hz on a log scale
        if v >= 100:
            return 20000.0
        log_lo = math.log10(20.0)
        log_hi = math.log10(20000.0)
        return 10.0 ** (log_lo + (log_hi - log_lo) * (v / 100.0))

    def _adsr_values(self) -> tuple[float, float, float, float]:
        a = self.a_slider.value() * 0.01    # 0..1 s
        d = self.d_slider.value() * 0.01    # 0..1 s
        s = self.s_slider.value() / 100.0   # 0..1
        r = self.r_slider.value() * 0.02    # 0..2 s
        return a, d, s, r

    # ============================================================
    # slots
    # ============================================================

    @Slot()
    def _on_pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select audio file", "",
            "Audio files (*.wav *.aif *.aiff *.flac *.mp3 *.ogg *.m4a);;All files (*)",
        )
        if not path:
            return
        self._audio_path = path
        self.file_label.setText(Path(path).name)
        self.run_btn.setEnabled(True)

    @Slot()
    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        self.manual_row.setVisible(mode == "manual")
        self.auto_row.setVisible(mode == "auto")
        self.hdb_row.setVisible(mode == "hdbscan")

    @Slot()
    def _on_run(self) -> None:
        if not self._audio_path:
            return
        self.run_btn.setEnabled(False)
        self.pick_btn.setEnabled(False)
        self.view_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status.setText("starting…")

        mode = self.mode_combo.currentData()
        threshold = self._threshold_from_slider(self.thresh_slider.value())
        min_cluster_size = self._min_cluster_size_from_slider(self.mcs_slider.value())
        noise_gate_db = self._volume_gate_db_from_slider(self.gate_slider.value())
        trim_threshold_db = self._trim_db_from_slider(self.trim_slider.value())

        self._thread = QThread(self)
        self._worker = PipelineWorker(
            self._audio_path,
            mode=mode,
            n_clusters=self.cluster_spin.value(),
            threshold=threshold,
            min_cluster_size=min_cluster_size,
            sensitivity=self.sens_slider.value() / 100.0,
            segment_length_s=self.length_slider.value() / 10.0,
            trim_threshold_db=trim_threshold_db,
            noise_gate_db=noise_gate_db,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._on_failed)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    @Slot(str, float)
    def _on_progress(self, msg: str, frac: float) -> None:
        self.progress_bar.setValue(int(frac * 100))
        self.status.setText(msg)

    @Slot(object)
    def _on_finished(self, instrument: object) -> None:
        assert isinstance(instrument, Instrument)
        self._instrument = instrument
        # Flat segment list for re-assigning keys on pitch-sort toggle.
        self._all_segments = [s for segs in instrument.notes.values() for s in segs]
        loaded = instrument.loaded_notes()
        self._octave = 0
        self._refresh_octave_view()
        if loaded:
            self.view_btn.setEnabled(True)
        else:
            self.view_btn.setEnabled(False)
            self.status.setText("0 keys loaded — try raising sensitivity or lowering threshold.")

    @Slot(str)
    def _on_failed(self, err: str) -> None:
        self.status.setText(f"failed: {err}")
        self.progress_bar.setValue(0)

    @Slot()
    def _on_thread_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.pick_btn.setEnabled(True)
        if self._instrument.loaded_notes():
            self.view_btn.setEnabled(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    @Slot()
    def _on_view_clusters(self) -> None:
        dlg = ClusterViewerDialog(
            self._instrument, self._engine,
            get_gain=lambda: self.vel_slider.value() / 127.0,
            parent=self,
        )
        dlg.exec()

    @Slot()
    def _on_adsr_changed(self) -> None:
        self._sync_adsr_to_engine()

    @Slot()
    def _on_lpf_changed(self) -> None:
        self._sync_lpf_to_engine()

    @Slot(bool)
    def _on_loop_toggled(self, checked: bool) -> None:
        self._engine.set_loop_enabled(checked)

    @Slot(int)
    def _on_gate_changed(self, v: int) -> None:
        if v <= 0:
            self.gate_value_lbl.setText("OFF")
        else:
            db = self._volume_gate_db_from_slider(v)
            self.gate_value_lbl.setText(f"{db:.0f} dB")

    @Slot(int)
    def _on_trim_changed(self, v: int) -> None:
        db = self._trim_db_from_slider(v)
        self.trim_value_lbl.setText(f"{db:.0f} dB")

    @Slot(bool)
    def _on_sort_toggled(self, checked: bool) -> None:
        if not self._all_segments:
            return
        # Cluster->key mapping is about to change, so release anything held.
        for n in list(self.piano._pressed):  # type: ignore[attr-defined]
            cluster_idx = n + self._octave * OCTAVE_STEP
            self._engine.release_note(cluster_idx)
            self.piano.release_note(n)
        self._instrument = assign_keys_to_clusters(
            self._all_segments, sort_by_freq=checked
        )
        self._octave = 0
        self._refresh_octave_view()

    def _sync_adsr_to_engine(self) -> None:
        a, d, s, r = self._adsr_values()
        self._engine.set_adsr(a, d, s, r)
        self.adsr_viz.set_adsr(a, d, s, r)

    def _sync_lpf_to_engine(self) -> None:
        hz = self._lpf_from_slider(self.lpf_slider.value())
        self._engine.set_lpf_cutoff(hz)

    # ============================================================
    # octave system
    # ============================================================

    def _shift_octave(self, delta: int) -> None:
        max_o = self._instrument.max_octave()
        new_octave = max(0, min(max_o, self._octave + delta))
        if new_octave == self._octave:
            return
        # Release any pressed visible keys before switching (avoid stuck notes)
        for n in list(self.piano._pressed):  # type: ignore[attr-defined]
            cluster_idx = n + self._octave * OCTAVE_STEP
            self._engine.release_note(cluster_idx)
            self.piano.release_note(n)
        self._octave = new_octave
        self._refresh_octave_view()

    def _refresh_octave_view(self) -> None:
        max_o = self._instrument.max_octave()
        base = self._octave * OCTAVE_STEP
        # Octave spans cluster indices [base, base + N_VISIBLE_KEYS - 1].
        # The high keys overlap with the next octave's low keys.
        loaded_in_octave = {
            ci - base
            for ci in self._instrument.loaded_notes()
            if base <= ci <= base + N_VISIBLE_KEYS - 1
        }
        self.piano.set_loaded(loaded_in_octave)
        self.oct_value_lbl.setText(f"{self._octave} / {max_o}")
        self.oct_down_btn.setEnabled(self._octave > 0)
        self.oct_up_btn.setEnabled(self._octave < max_o)

    # ============================================================
    # note triggering (with velocity-layered crossfade)
    # ============================================================

    @Slot(int)
    def _on_note_pressed(self, note_index: int) -> None:
        cluster_idx = note_index + self._octave * OCTAVE_STEP
        segs = self._instrument.notes.get(cluster_idx)
        if not segs:
            return

        velocity = self.vel_slider.value()      # 1..127
        master_gain = velocity / 127.0

        if len(segs) == 1:
            self._engine.play(segs[0].audio, gain=master_gain, note_id=cluster_idx)
            return

        # Velocity layering with equal-power crossfade between adjacent layers.
        # Fractional position in [0, N-1]
        pos = (velocity - 1) / 126.0 * (len(segs) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(segs) - 1)
        frac = pos - lo

        if lo == hi:
            self._engine.play(segs[lo].audio, gain=master_gain, note_id=cluster_idx)
            return

        # Equal-power (cos/sin) curves keep perceived loudness uniform.
        w_lo = math.cos(frac * math.pi / 2.0)
        w_hi = math.sin(frac * math.pi / 2.0)
        self._engine.play(segs[lo].audio, gain=master_gain * w_lo, note_id=cluster_idx)
        self._engine.play(segs[hi].audio, gain=master_gain * w_hi, note_id=cluster_idx)

    @Slot(int)
    def _on_note_released(self, note_index: int) -> None:
        cluster_idx = note_index + self._octave * OCTAVE_STEP
        self._engine.release_note(cluster_idx)

    # ============================================================
    # window-level keyboard
    # ============================================================

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_Z:
            self._shift_octave(-1)
            return
        if event.key() == Qt.Key_X:
            self._shift_octave(+1)
            return
        note = KEY_TO_NOTE.get(event.key())
        if note is None:
            super().keyPressEvent(event)
            return
        self.piano.press_note(note)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        if event.key() in (Qt.Key_Z, Qt.Key_X):
            return
        note = KEY_TO_NOTE.get(event.key())
        if note is None:
            super().keyReleaseEvent(event)
            return
        self.piano.release_note(note)

    # ============================================================
    # lifecycle
    # ============================================================

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
        self._engine.stop()
        super().closeEvent(event)
