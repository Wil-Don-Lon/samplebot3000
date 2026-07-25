"""Main window: integrates pipeline + audio engine + piano widget.

Layout: two columns of controls above a piano keyboard.
- Left column   : source + clustering controls + run
- Right column  : playback controls (velocity, ADSR, LPF, octave)
- Bottom        : keyboard
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QProgressBar,
    QFrame, QSizePolicy, QButtonGroup, QScrollArea,
    QApplication, QLineEdit, QDialog, QListWidget, QListWidgetItem,
)

import pipeline as P
from pipeline import (
    run_pipeline, assign_keys_to_clusters, build_instrument_drumkeys_gm,
    Instrument, NOTE_NAMES, NOTE_NAMES_12, MAX_KEYS, KEY_CAP_LABEL, DRUM_KEYMAP,
)
from audio_engine import AudioEngine
from piano_widget import PianoKeyboardWidget
from recategorize_dialog import RecategorizeDialog
from adsr_widget import ADSREnvelopeWidget
from moog_widgets import Knob, ToggleSwitch, LED
from waveform_editor import WaveformEditor
import kit_store
import logic_export
from logic_export import BASE_MIDI_NOTE
from midi_input import MidiListener
from loudness import equal_loudness_gains

try:
    from scipy.signal import butter, lfilter
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False



from dataclasses import dataclass


@dataclass
class PlaybackSettings:
    """Per-sample playback parameters, stored in widget (slider) domain so they
    load back into the controls exactly. Converted to engine units at play time.
    Defaults match the control defaults."""
    velocity: int = 100   # 1..127
    loop: bool = False
    a: int = 0            # 0..100  -> *0.01 s   attack
    d: int = 0            # 0..100  -> *0.01 s   decay
    s: int = 100          # 0..100  -> /100      sustain level
    r: int = 100          # 0..100  -> *0.02 s   release (default fully maxed)
    lpf: int = 100        # 0..100  -> _lpf_from_slider -> Hz
    volume: int = 100     # 0..100  -> /100      per-sample level trim (unity=100)


def velocity_lowpass(audio: np.ndarray, velocity: int, sr: int = 44100) -> np.ndarray:
    """Slight velocity-dependent lowpass: softer hits are a touch darker.
    Full bandwidth at max velocity; gentle 2nd-order rolloff toward ~6 kHz at
    the lowest velocity. No-op without scipy."""
    if velocity >= 127 or not _HAVE_SCIPY or audio.size < 16:
        return audio
    frac = max(0.0, min(1.0, velocity / 127.0))
    cutoff = 6000.0 * (sr * 0.45 / 6000.0) ** frac   # ~6 kHz .. ~Nyquist
    if cutoff >= sr * 0.45:
        return audio
    b, a = butter(2, cutoff / (sr * 0.5), btype="low")
    return lfilter(b, a, audio).astype(np.float32, copy=False)


def bake_lowpass(audio: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """Bake the per-sample lowpass into `audio` — a 4th-order Butterworth
    (~24 dB/oct), matching the engine's per-voice cascade closely enough. No-op
    at/above ~0.45·SR (the slider's 'open' end) or without scipy."""
    nyq = sr * 0.5
    if not _HAVE_SCIPY or audio.size < 16 or cutoff_hz >= sr * 0.45:
        return audio
    b, a = butter(4, max(20.0, cutoff_hz) / nyq, btype="low")
    return lfilter(b, a, audio).astype(np.float32, copy=False)


def bake_adsr(audio: np.ndarray, a_s: float, d_s: float, s_level: float,
              sr: int) -> np.ndarray:
    """Bake the Attack→Decay→Sustain amplitude envelope into `audio`. Release is
    a live note-off articulation, so it isn't baked (a one-shot drum plays at the
    sustain level to its end). No-op for the default 0/0/1.0 envelope."""
    n = audio.shape[0]
    a = max(0, int(a_s * sr))
    d = max(0, int(d_s * sr))
    if (a == 0 and d == 0) or n == 0:
        return audio if s_level >= 1.0 else (audio * np.float32(s_level))
    env = np.empty(n, dtype=np.float32)
    i = 0
    if a > 0:                                   # attack: 0 → 1
        k = min(a, n)
        env[:k] = np.linspace(0.0, 1.0, k, endpoint=False, dtype=np.float32)
        i = k
    if i < n and d > 0:                         # decay: 1 → sustain
        k = min(d, n - i)
        env[i:i + k] = 1.0 + (s_level - 1.0) * np.linspace(
            0.0, 1.0, k, endpoint=False, dtype=np.float32)
        i += k
    if i < n:                                   # sustain hold
        env[i:] = s_level
    return (audio * env).astype(np.float32, copy=False)


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
/* ── SPACEAGE MOOG · warm amber analog-synth panel ───────────────── */
QMainWindow, QDialog {
    background-color: #0b0907;
}
/* Wood cabinet around the panel */
QWidget#wood {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2e1a0c, stop:0.06 #5a3818, stop:0.12 #3a2410,
        stop:0.5 #4a2e14, stop:0.88 #3a2410, stop:0.94 #5a3818, stop:1 #2e1a0c);
}
/* Brushed-metal control panel */
QFrame#panel {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1c1812, stop:0.5 #15110b, stop:1 #100d08);
    border: 2px solid #060504;
    border-radius: 6px;
}
QWidget {
    color: #d8cdbb;
    font-family: -apple-system, "SF Pro Display", "Segoe UI", system-ui, sans-serif;
    font-size: 12px;
}
QLabel { background: transparent; }

QLabel#title {
    color: #ffb15a;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 6px;
}
QLabel#subtitle {
    color: #8a6a3a;
    font-size: 10px;
    letter-spacing: 3px;
}
QLabel#sectionHeader {
    color: #ff8a1e;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 4px;
    padding-top: 6px;
}
QLabel#controlLabel {
    color: #9c8463;
    font-size: 10px;
    letter-spacing: 1.5px;
}
QLabel#valueLabel {
    color: #7fe6dc;                 /* pale cyan LED readout */
    font-family: "SF Mono", "Menlo", monospace;
    font-size: 11px;
}
QLabel#statusLabel {
    color: #a08c6a;
    font-size: 11px;
}
QLabel#hintLabel {
    color: #6a5638;
    font-size: 10px;
    letter-spacing: 2px;
}
QLabel#octaveValue {
    color: #7fe6dc;
    font-family: "SF Mono", "Menlo", monospace;
    font-size: 13px;
    font-weight: 600;
}

/* Cluster viewer labels */
QLabel#dialogHeader {
    color: #ffb15a;
    font-size: 11px;
    letter-spacing: 2px;
}
QLabel#dialogSubtle, QLabel#segmentRow {
    color: #a08c6a;
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
    background-color: #17120c;
    color: #ffae57;
    border: 1px solid #4a3820;
    padding: 7px 16px;
    border-radius: 3px;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 600;
}
QPushButton:hover { border-color: #ff8a1e; color: #ffd29a; background-color: #221a10; }
QPushButton:pressed { background-color: #ff8a1e; color: #1a1206; }
QPushButton:disabled { color: #4a3c28; border-color: #2a2014; background-color: #120e09; }

/* Method toggle — segmented two-button switch */
QPushButton#toggleLeft  { border-top-right-radius: 0; border-bottom-right-radius: 0; }
QPushButton#toggleRight { border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: none; }
QPushButton#toggleLeft:checked, QPushButton#toggleRight:checked {
    background-color: #ff8a1e;
    color: #160f06;
    border-color: #ff8a1e;
}

QPushButton#playBtn {
    color: #ff8a1e;
    background: transparent;
    border: none;
    padding: 2px;
    font-size: 13px;
    letter-spacing: 0;
}
QPushButton#playBtn:hover { color: #ffd29a; }
QPushButton#octaveBtn {
    padding: 4px 10px;
    font-size: 11px;
    min-width: 24px;
}

QFrame#clusterGroup {
    background-color: #15110b;
    border: 1px solid #2c2114;
    border-radius: 4px;
}

QSpinBox, QComboBox {
    background-color: #17120c;
    color: #f0e3cd;
    border: 1px solid #4a3820;
    padding: 5px 10px;
    border-radius: 3px;
    selection-background-color: #ff8a1e;
    selection-color: #160f06;
    min-width: 80px;
}
QSpinBox:hover, QComboBox:hover { border-color: #ff8a1e; }

QLineEdit {
    background-color: #17120c;
    color: #f0e3cd;
    border: 1px solid #4a3820;
    padding: 5px 10px;
    border-radius: 3px;
    selection-background-color: #ff8a1e;
    selection-color: #160f06;
}
QLineEdit:hover { border-color: #ff8a1e; }
QLineEdit:focus { border-color: #ff8a1e; }

QListWidget {
    background-color: #17120c;
    color: #f0e3cd;
    border: 1px solid #4a3820;
    border-radius: 3px;
    outline: none;
}
QListWidget::item { padding: 6px 8px; }
QListWidget::item:selected { background-color: #ff8a1e; color: #160f06; }
/* Recategorize drag-and-drop list */
QListWidget#recatList { background-color: #0f0b07; border: 1px solid #3a2c18; }
QListWidget#recatList::item { padding: 4px 8px; }
QListWidget#recatList::item:selected { background-color: #ff8a1e; color: #160f06; }
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
    border-top: 5px solid #ff8a1e;
    margin-right: 7px;
}
QComboBox QAbstractItemView {
    background-color: #17120c;
    color: #f0e3cd;
    border: 1px solid #4a3820;
    selection-background-color: #ff8a1e;
    selection-color: #160f06;
    outline: none;
}

QSlider::groove:horizontal {
    border: none;
    background: #2a2014;
    height: 3px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #ffae57;
    border: 2px solid #1a1206;
    width: 14px;
    height: 14px;
    margin: -7px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover { background: #ffd29a; }
QSlider::sub-page:horizontal { background: #ff8a1e; height: 3px; border-radius: 2px; }

QProgressBar {
    background-color: #2a2014;
    border: none;
    border-radius: 0;
    height: 3px;
    text-align: center;
}
QProgressBar::chunk { background-color: #ff8a1e; }

QScrollArea { background-color: transparent; border: none; }
QScrollBar:vertical { background: #0b0907; width: 6px; margin: 0; }
QScrollBar::handle:vertical {
    background: #4a3820; border-radius: 3px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #ff8a1e; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    background: none; height: 0;
}

QFrame[frameShape="4"] { color: #2c2114; background: #2c2114; max-height: 1px; }
QFrame[frameShape="5"] { color: #2c2114; background: #2c2114; max-width: 1px; }

QCheckBox { color: #d8cdbb; spacing: 8px; }
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #4a3820;
    background: #17120c;
    border-radius: 2px;
}
QCheckBox::indicator:hover { border-color: #ff8a1e; }
QCheckBox::indicator:checked {
    background: #ff8a1e;
    border-color: #ff8a1e;
    image: none;
}
QCheckBox:disabled { color: #5a4c38; }
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
        min_cluster_size: float,
        sensitivity: float,
        segment_length_s: float,
        trim_threshold_db: float,
        noise_gate_db,
        clap_sort: bool = False,
        clip_at_next_onset: bool = False,
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
        self._clap_sort = clap_sort
        self._clip_at_next_onset = clip_at_next_onset

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
                clap_sort=self._clap_sort,
                clip_at_next_onset=self._clip_at_next_onset,
                progress_callback=lambda msg, frac: self.progress.emit(msg, frac),
            )
            self.finished.emit(inst)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class KitBrowserDialog(QDialog):
    """Pick a saved kit to load, or delete kits. `selected_path` is set on load."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Kit")
        self.resize(420, 440)
        self.selected_path = None
        self._build_ui()
        self._reload()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QLabel("SAVED KITS")
        header.setObjectName("dialogHeader")
        root.addWidget(header)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _i: self._on_load())
        root.addWidget(self.list, 1)

        self.empty_lbl = QLabel("(no kits saved yet)")
        self.empty_lbl.setObjectName("dialogSubtle")
        self.empty_lbl.setVisible(False)
        root.addWidget(self.empty_lbl)

        row = QHBoxLayout()
        self.delete_btn = QPushButton("DELETE")
        self.delete_btn.clicked.connect(self._on_delete)
        row.addWidget(self.delete_btn)
        row.addStretch(1)
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.clicked.connect(self.reject)
        self.load_btn = QPushButton("LOAD")
        self.load_btn.clicked.connect(self._on_load)
        row.addWidget(cancel_btn)
        row.addWidget(self.load_btn)
        root.addLayout(row)

    def _reload(self) -> None:
        self.list.clear()
        kits = kit_store.list_kits()
        for k in kits:
            layout = "drum kit" if k["is_classify"] else "clustered"
            item = QListWidgetItem(
                f"{k['name']}   ·   {k['n_samples']} samples / {k['n_keys']} keys   ·   {layout}"
            )
            item.setData(Qt.UserRole, str(k["path"]))
            self.list.addItem(item)
        has = self.list.count() > 0
        self.empty_lbl.setVisible(not has)
        self.load_btn.setEnabled(has)
        self.delete_btn.setEnabled(has)
        if has:
            self.list.setCurrentRow(0)

    def _current_path(self):
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def _on_load(self) -> None:
        path = self._current_path()
        if path is None:
            return
        self.selected_path = path
        self.accept()

    def _on_delete(self) -> None:
        path = self._current_path()
        if path is None:
            return
        try:
            kit_store.delete_kit(path)
        except OSError:
            pass
        self._reload()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("samplebot-3000")
        self.setFocusPolicy(Qt.StrongFocus)
        # Fit the initial window to the available screen so it never opens
        # taller/wider than the display (the panel scrolls if it needs more).
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        w = min(940, avail.width() - 40) if avail else 940
        h = min(620, avail.height() - 80) if avail else 620
        self.resize(w, h)

        self._audio_path: Optional[str] = None
        self._instrument: Instrument = Instrument()
        # Flat list of every clustered segment, cached so the pitch-sort toggle
        # can re-assign keys without re-running the whole pipeline.
        self._all_segments: list = []
        self._octave: int = 0
        self._thread: Optional[QThread] = None
        self._worker: Optional[PipelineWorker] = None

        # True when the last run produced the fixed GM drum layout (CLAP-sort),
        # which drives the drum-cap key labels and disables pitch-sort.
        self._result_is_classify: bool = False

        # Per-key chosen sample (arrow-tab), selection, and step patterns.
        self._sample_choice: dict[int, int] = {}      # key -> chosen sample idx
        # Sample-pick mode for keys holding several stacked samples:
        #   "random"   → a random sample from the cluster fires on each hit.
        #   "velocity" → the velocity slider picks the layer (soft→loud by rms).
        self._sample_mode: str = "random"
        # Cluster→key sort mode: "frequency" (pitch-ordered) | "auto" (autosorter,
        # GM keys). Default frequency (matches the old PITCH-SORT-on default).
        self._sort_mode: str = "frequency"
        # When False (default), envelope + filter edits apply to the whole cluster
        # (every sample on the selected key); when True, only the active sample.
        self._per_sample_edit: bool = False
        # A-weighted equal-loudness normalization (off by default; opt-in).
        self._equal_loudness: bool = False
        self._selected_key: Optional[int] = None
        # True while pushing a sample's settings into the controls, so the
        # control-changed handlers don't write them straight back.
        self._loading_settings: bool = False

        self._engine = AudioEngine()
        self._engine.start()

        self._build_ui()
        self._sync_adsr_to_engine()
        self._sync_lpf_to_engine()

        # USB MIDI: play the clustered kit from a keyboard/pads. Optional and
        # hot-pluggable — a 2s poll (re)connects the first device and detects
        # unplug. MIDI note N triggers cluster N-BASE_MIDI_NOTE (same mapping as
        # the Logic export), so playing here matches the exported kit.
        self._midi = MidiListener()
        self._midi.note_on.connect(self._on_midi_note_on)
        self._midi.note_off.connect(self._on_midi_note_off)
        self._midi_timer = QTimer(self)
        self._midi_timer.timeout.connect(self._poll_midi)
        self._midi_timer.start(2000)
        self._poll_midi()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self) -> None:
        # Scroll area as the central widget so the panel can never be taller
        # than the window — it scrolls instead of clipping off-screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setCentralWidget(scroll)

        central = QWidget()
        central.setObjectName("wood")
        scroll.setWidget(central)
        shell = QVBoxLayout(central)
        shell.setContentsMargins(16, 16, 16, 16)   # wood border peeks through
        panel = QFrame()
        panel.setObjectName("panel")
        shell.addWidget(panel)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(14)

        # Header
        header_row = QHBoxLayout()
        title = QLabel("SAMPLEBOT-3000")
        title.setObjectName("title")
        subtitle = QLabel("ANALOG · DIGITAL DRUM ENGINE")
        subtitle.setObjectName("subtitle")
        header_row.addWidget(title)
        header_row.addSpacing(12)
        header_row.addWidget(subtitle)
        header_row.addStretch(1)
        self.midi_lbl = QLabel("MIDI: —")
        self.midi_lbl.setObjectName("subtitle")
        header_row.addWidget(self.midi_lbl)
        outer.addLayout(header_row)

        outer.addWidget(self._hline())

        # File / run row (full width above the two columns)
        outer.addLayout(self._build_top_row())

        # Kit row: name + save/load/recategorize/export-to-Logic
        outer.addLayout(self._build_kit_row())

        outer.addWidget(self._hline())

        # Two columns
        cols = QHBoxLayout()
        cols.setSpacing(24)
        cols.addLayout(self._build_left_column(), 1)
        cols.addWidget(self._vline())
        cols.addLayout(self._build_right_column(), 1)
        outer.addLayout(cols, 1)

        outer.addWidget(self._hline())

        # Active-key indicator + sample selector. Pick a key, then tab through
        # its samples with the arrows; the key plays the chosen sample.
        active_row = QHBoxLayout()
        active_row.setSpacing(8)
        self.active_led = LED()
        active_row.addWidget(self.active_led)
        active_cap = QLabel("ACTIVE")
        active_cap.setObjectName("controlLabel")
        active_row.addWidget(active_cap)
        self.active_lbl = QLabel("—")
        self.active_lbl.setObjectName("octaveValue")
        active_row.addWidget(self.active_lbl)
        active_row.addSpacing(16)
        self.sample_prev_btn = QPushButton("◀")
        self.sample_prev_btn.setObjectName("octaveBtn")
        self.sample_prev_btn.setEnabled(False)
        self.sample_prev_btn.clicked.connect(lambda: self._step_sample(-1))
        active_row.addWidget(self.sample_prev_btn)
        self.sample_lbl = QLabel("– / –")
        self.sample_lbl.setObjectName("valueLabel")
        self.sample_lbl.setMinimumWidth(60)
        self.sample_lbl.setAlignment(Qt.AlignCenter)
        active_row.addWidget(self.sample_lbl)
        self.sample_next_btn = QPushButton("▶")
        self.sample_next_btn.setObjectName("octaveBtn")
        self.sample_next_btn.setEnabled(False)
        self.sample_next_btn.clicked.connect(lambda: self._step_sample(+1))
        active_row.addWidget(self.sample_next_btn)
        active_row.addStretch(1)
        outer.addLayout(active_row)

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

        return row

    # ---------- kit row (save / load / recategorize / export) ----------

    def _build_kit_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        kit_cap = QLabel("KIT")
        kit_cap.setObjectName("sectionHeader")
        row.addWidget(kit_cap)

        self.kit_name_edit = QLineEdit()
        self.kit_name_edit.setPlaceholderText("kit name")
        self.kit_name_edit.setMaximumWidth(220)
        row.addWidget(self.kit_name_edit, 1)

        self.save_kit_btn = QPushButton("SAVE KIT")
        self.save_kit_btn.clicked.connect(self._on_save_kit)
        row.addWidget(self.save_kit_btn)

        self.load_kit_btn = QPushButton("LOAD KIT")
        self.load_kit_btn.clicked.connect(self._on_load_kit)
        row.addWidget(self.load_kit_btn)

        self.recat_btn = QPushButton("RECATEGORIZE")
        self.recat_btn.setEnabled(False)
        self.recat_btn.clicked.connect(self._on_recategorize)
        row.addWidget(self.recat_btn)

        self.export_btn = QPushButton("EXPORT → LOGIC")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export_logic)
        row.addWidget(self.export_btn)

        return row

    # ---------- left column (analysis) ----------

    def _build_left_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(10)

        col.addWidget(self._section_header("ANALYSIS"))

        # HDBSCAN is the only clustering model now. KMeans (spheroid-constrained)
        # and Agglomerative were dropped — audio features are too complex for
        # those assumptions; HDBSCAN's density model is leagues better. Only its
        # CLUSTER DIVERSITY knob remains.
        self.hdb_row = self._make_hdbscan_row()
        col.addWidget(self.hdb_row)

        # Sample length (affects analysis — requires Run to apply)
        # SAMPLE LEN sets the PLAYED/exported sample length only — clustering
        # always sorts on a fixed short internal window (CLUSTER_FEATURE_LEN_S),
        # so a long ring-out here never affects the (tuned) sorting. Default 4.0s
        # + CLIP AT NEXT on = each hit rings out fully up to the next onset.
        self.length_slider = Knob()
        self.length_slider.setRange(1, 50)   # 0.1s .. 5.0s
        self.length_slider.setValue(40)      # 4.0s playback cap
        self.length_value_lbl = QLabel("4.0 s")
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
            label_attr="sens_label",
        ))

        # Noise gate: peak-based, drops whole segments below this threshold
        self.gate_slider = Knob()
        self.gate_slider.setRange(0, 100)
        self.gate_slider.setValue(0)
        self.gate_value_lbl = QLabel("OFF")
        self.gate_value_lbl.setObjectName("valueLabel")
        self.gate_value_lbl.setMinimumWidth(50)
        self.gate_slider.valueChanged.connect(self._on_gate_changed)
        col.addLayout(self._slider_row("NOISE GATE", self.gate_slider, self.gate_value_lbl))

        # Trim threshold: silence-trim edges below this dBFS level
        self.trim_slider = Knob()
        self.trim_slider.setRange(0, 100)
        self.trim_slider.setValue(33)        # -42 dB — tuned on the user's kit
        self.trim_value_lbl = QLabel("-42 dB")
        self.trim_value_lbl.setObjectName("valueLabel")
        self.trim_value_lbl.setMinimumWidth(50)
        self.trim_slider.valueChanged.connect(self._on_trim_changed)
        col.addLayout(self._slider_row("TRIM THRESHOLD", self.trim_slider,
                                       self.trim_value_lbl, label_attr="trim_label"))

        # Clipper: when on, a played sample ends at the next transient; when off
        # it runs to the trim-threshold decay or the sample-length cap. Default ON
        # so the (now-longer) playback window rings out fully up to the next hit
        # instead of bleeding it into the tail.
        clip_row = QHBoxLayout()
        clip_lbl = QLabel("CLIP AT NEXT")
        clip_lbl.setObjectName("controlLabel")
        clip_lbl.setMinimumWidth(110)
        self.clip_check = ToggleSwitch(checked=True)
        clip_row.addWidget(clip_lbl)
        clip_row.addWidget(self.clip_check)
        clip_row.addStretch(1)
        col.addLayout(clip_row)

        # SORT mode: how finished clusters map to keys.
        #  FREQUENCY = order clusters by pitch onto the keyboard (raw, no labels).
        #  AUTO      = the auto-sorter labels each cluster and drops it on its GM
        #              key, one cluster per key.
        sort_row = QHBoxLayout()
        sort_lbl = QLabel("SORT")
        sort_lbl.setObjectName("controlLabel")
        sort_lbl.setMinimumWidth(110)
        sort_row.addWidget(sort_lbl)
        sort_row.addWidget(self._make_sort_mode_toggle(), 1)
        col.addLayout(sort_row)

        # Progress + status
        col.addWidget(self._progress_bar())
        self.status = QLabel("ready")
        self.status.setObjectName("statusLabel")
        # Long status/summary text must wrap, not widen the window. Allow the
        # label to shrink below its text's ideal width and grow vertically.
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        col.addWidget(self.status)

        # Per-sample waveform editor. Press a key (or use ◀ ▶) to load the active
        # sample; drag the two ends to retrim the start or elongate the tail into
        # the padded runway. Edits change what plays AND what exports.
        col.addSpacing(4)
        editor_head = QHBoxLayout()
        editor_head.addWidget(self._section_header("SAMPLE EDITOR"))
        editor_head.addStretch(1)
        # AUTOTRIM: arm it, then dial TRANSIENT SENS / TRIM THRESHOLD (they light
        # up and the editor previews the cut on the current sample); APPLY commits
        # that trim to every sample on the SELECTED KEY's cluster, so each cluster
        # can be judged on its own before moving to the next.
        self.autotrim_btn = QPushButton("AUTOTRIM")
        self.autotrim_btn.setObjectName("octaveBtn")
        self.autotrim_btn.setCheckable(True)
        self.autotrim_btn.setEnabled(False)
        self.autotrim_btn.toggled.connect(self._on_autotrim_toggled)
        editor_head.addWidget(self.autotrim_btn)
        self.autotrim_apply_btn = QPushButton("APPLY TO CLUSTER")
        self.autotrim_apply_btn.setObjectName("octaveBtn")
        self.autotrim_apply_btn.setVisible(False)
        self.autotrim_apply_btn.clicked.connect(self._apply_autotrim_cluster)
        editor_head.addWidget(self.autotrim_apply_btn)
        col.addLayout(editor_head)

        self.autotrim_hint = QLabel("")
        self.autotrim_hint.setObjectName("dialogSubtle")
        self.autotrim_hint.setWordWrap(True)
        col.addWidget(self.autotrim_hint)

        self.wave_editor = WaveformEditor()
        self.wave_editor.boundsChanged.connect(self._on_wave_bounds_changed)
        self.wave_editor.editCommitted.connect(self._on_wave_edit_committed)
        col.addWidget(self.wave_editor, 1)

        col.addStretch(0)
        return col

    def _make_sample_mode_toggle(self) -> QWidget:
        """Segmented RANDOM | VEL LAYER switch for per-hit sample selection."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.btn_random = QPushButton("RANDOM")
        self.btn_random.setObjectName("toggleLeft")
        self.btn_random.setCheckable(True)
        self.btn_random.setChecked(self._sample_mode == "random")
        self.btn_vellayer = QPushButton("VEL LAYER")
        self.btn_vellayer.setObjectName("toggleRight")
        self.btn_vellayer.setCheckable(True)
        self.btn_vellayer.setChecked(self._sample_mode == "velocity")
        grp = QButtonGroup(w)
        grp.setExclusive(True)
        grp.addButton(self.btn_random)
        grp.addButton(self.btn_vellayer)
        self.btn_random.clicked.connect(lambda: self._set_sample_mode("random"))
        self.btn_vellayer.clicked.connect(lambda: self._set_sample_mode("velocity"))
        h.addWidget(self.btn_random, 1)
        h.addWidget(self.btn_vellayer, 1)
        return w

    def _make_sort_mode_toggle(self) -> QWidget:
        """Segmented FREQUENCY | AUTO switch for cluster→key sorting."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.btn_freq = QPushButton("FREQUENCY")
        self.btn_freq.setObjectName("toggleLeft")
        self.btn_freq.setCheckable(True)
        self.btn_freq.setChecked(self._sort_mode == "frequency")
        self.btn_auto = QPushButton("AUTO")
        self.btn_auto.setObjectName("toggleRight")
        self.btn_auto.setCheckable(True)
        self.btn_auto.setChecked(self._sort_mode == "auto")
        grp = QButtonGroup(w)
        grp.setExclusive(True)
        grp.addButton(self.btn_freq)
        grp.addButton(self.btn_auto)
        self.btn_freq.clicked.connect(lambda: self._set_sort_mode("frequency"))
        self.btn_auto.clicked.connect(lambda: self._set_sort_mode("auto"))
        h.addWidget(self.btn_freq, 1)
        h.addWidget(self.btn_auto, 1)
        return w

    def _make_hdbscan_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel("CLUSTER DIVERSITY")
        label.setObjectName("controlLabel")
        label.setMinimumWidth(130)
        row.addWidget(label)
        # Decimal diversity: slider units are tenths (20..150 = 2.0..15.0). The
        # value is min_cluster_size; the fractional part becomes a small
        # cluster-merge epsilon downstream. Default 4.0 was tuned (grid search
        # over the user's kit) to best reproduce their hand-drawn per-drum slices.
        self.mcs_slider = QSlider(Qt.Horizontal)
        self.mcs_slider.setRange(20, 150)
        self.mcs_slider.setValue(40)   # 4.0 — tuned to reproduce the user's slices
        self.mcs_value_lbl = QLabel("4.0")
        self.mcs_value_lbl.setObjectName("valueLabel")
        self.mcs_value_lbl.setMinimumWidth(50)
        self.mcs_slider.valueChanged.connect(
            lambda v: self.mcs_value_lbl.setText(f"{v/10:.1f}")
        )
        row.addWidget(self.mcs_slider, 1)
        row.addWidget(self.mcs_value_lbl)
        return w

    def _make_sens_slider(self) -> QWidget:
        self.sens_slider = Knob()
        self.sens_slider.setRange(0, 100)
        self.sens_slider.setValue(35)        # 0.35 — tuned on the user's kit
        self.sens_value_lbl = QLabel("0.35")
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
        self.vel_slider = Knob()
        self.vel_slider.setRange(1, 127)
        self.vel_slider.setValue(100)
        self.vel_value_lbl = QLabel("100")
        self.vel_value_lbl.setObjectName("valueLabel")
        self.vel_value_lbl.setMinimumWidth(50)
        self.vel_slider.valueChanged.connect(
            lambda v: self.vel_value_lbl.setText(str(v))
        )
        self.vel_slider.valueChanged.connect(self._on_playback_control_changed)
        # In VEL LAYER mode the slider chooses the layer — keep the
        # ◀ vel N/M ▶ readout in sync as it moves.
        self.vel_slider.valueChanged.connect(
            lambda _v: self._sample_mode == "velocity" and self._update_sample_selector())
        col.addLayout(self._slider_row("VELOCITY", self.vel_slider, self.vel_value_lbl))

        # Per-sample VOLUME: a stored level trim on the sample(s), separate from
        # the live VELOCITY hit gain. Like ENVELOPE/FILTER it's governed by the
        # PER-SAMPLE EDIT toggle (whole cluster by default, active sample when on).
        self.volume_slider = Knob()
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_value_lbl = QLabel("100%")
        self.volume_value_lbl.setObjectName("valueLabel")
        self.volume_value_lbl.setMinimumWidth(50)
        self.volume_slider.valueChanged.connect(
            lambda v: self.volume_value_lbl.setText(f"{v}%")
        )
        self.volume_slider.valueChanged.connect(self._on_playback_control_changed)
        col.addLayout(self._slider_row("VOLUME", self.volume_slider, self.volume_value_lbl))

        # Loop while held
        loop_row = QHBoxLayout()
        loop_label = QLabel("LOOP")
        loop_label.setObjectName("controlLabel")
        loop_label.setMinimumWidth(110)
        self.loop_check = ToggleSwitch()
        self.loop_check.toggled.connect(self._on_loop_toggled)
        loop_row.addWidget(loop_label)
        loop_row.addWidget(self.loop_check, 1)
        col.addLayout(loop_row)

        # Sample pick: how a key with several stacked samples chooses which one
        # fires on each hit. RANDOM = a random sample from the cluster every hit;
        # VEL LAYER = the velocity slider selects the layer (soft→loud).
        smode_row = QHBoxLayout()
        smode_label = QLabel("SAMPLE PICK")
        smode_label.setObjectName("controlLabel")
        smode_label.setMinimumWidth(110)
        smode_row.addWidget(smode_label)
        smode_row.addWidget(self._make_sample_mode_toggle(), 1)
        col.addLayout(smode_row)

        # Equal loudness: A-weighted (Fletcher-Munson) perceived-loudness
        # normalization — tames bright/harsh samples so the whole kit sits at
        # even perceived loudness. Applies to playback AND the Logic export.
        eqloud_row = QHBoxLayout()
        eqloud_label = QLabel("EQUAL LOUDNESS")
        eqloud_label.setObjectName("controlLabel")
        eqloud_label.setMinimumWidth(110)
        self.eqloud_check = ToggleSwitch()
        self.eqloud_check.toggled.connect(self._on_equal_loudness_toggled)
        eqloud_row.addWidget(eqloud_label)
        eqloud_row.addWidget(self.eqloud_check, 1)
        col.addLayout(eqloud_row)

        # Per-sample edit: by default, volume + envelope + filter edits apply to
        # the WHOLE cluster (every sample on the selected key). Flip this on to
        # sculpt just the active sample instead.
        psedit_row = QHBoxLayout()
        psedit_label = QLabel("PER-SAMPLE EDIT")
        psedit_label.setObjectName("controlLabel")
        psedit_label.setMinimumWidth(110)
        self.psedit_check = ToggleSwitch()
        self.psedit_check.toggled.connect(self._on_per_sample_edit_toggled)
        psedit_row.addWidget(psedit_label)
        psedit_row.addWidget(self.psedit_check, 1)
        col.addLayout(psedit_row)

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
        # Release: 0..2000ms linear. Default fully maxed (2000 ms) so samples
        # ring out by default rather than being cut short.
        self.r_slider, self.r_lbl = make_adsr_slider(
            100, lambda v: f"{int(v * 20)} ms"
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
        self.lpf_slider = Knob()
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

    def _labeled_row_widget(self, label_text: str, widget: QWidget) -> QWidget:
        """Same as _labeled_row but wrapped in a QWidget so it can be shown/hidden."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setObjectName("controlLabel")
        lbl.setMinimumWidth(110)
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return w

    def _slider_row(self, label_text: str, slider: QSlider, value_lbl: QLabel,
                    label_attr: Optional[str] = None) -> QHBoxLayout:
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setObjectName("controlLabel")
        lbl.setMinimumWidth(110)
        if label_attr:                       # keep a handle so it can be highlighted
            setattr(self, label_attr, lbl)
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
    def _on_run(self) -> None:
        if not self._audio_path:
            return
        self.run_btn.setEnabled(False)
        self.pick_btn.setEnabled(False)
        self._enable_instrument_actions(False)
        self.progress_bar.setValue(0)
        self.status.setText("starting…")

        # Cluster-first flow only. Stage 1 = HDBSCAN clustering; Stage 2 = the
        # AUTO sorter drops finished clusters onto GM keys (FREQUENCY skips it).
        mode = "hdbscan"   # the only clustering model
        clap_sort = self._sort_mode == "auto"
        self._result_is_classify = clap_sort   # AUTO → fixed GM layout + cap labels

        # HDBSCAN diversity is a float (tenths): int part = min_cluster_size,
        # fractional part feeds a small merge epsilon downstream.
        min_cluster_size = self.mcs_slider.value() / 10.0
        noise_gate_db = self._volume_gate_db_from_slider(self.gate_slider.value())
        trim_threshold_db = self._trim_db_from_slider(self.trim_slider.value())

        self._thread = QThread(self)
        self._worker = PipelineWorker(
            self._audio_path,
            mode=mode,
            n_clusters=8,        # unused by HDBSCAN; only its KMeans fallback
            threshold=5.0,       # unused by HDBSCAN
            min_cluster_size=min_cluster_size,
            sensitivity=self.sens_slider.value() / 100.0,
            segment_length_s=self.length_slider.value() / 10.0,
            trim_threshold_db=trim_threshold_db,
            noise_gate_db=noise_gate_db,
            clap_sort=clap_sort,
            clip_at_next_onset=self.clip_check.isChecked(),
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
        # Flat segment list for re-assigning keys on pitch-sort toggle. Includes
        # the unassigned outliers so a re-sort re-buckets (not loses) them.
        self._all_segments = (
            [s for segs in instrument.notes.values() for s in segs]
            + list(instrument.unassigned))
        self._apply_loudness_gains()
        loaded = instrument.loaded_notes()
        self._octave = 0
        # Fresh instrument → reset sample choice and selection.
        self._sample_choice = {}
        self._selected_key = None
        self.piano.set_selected(None)
        self.active_led.setOn(False)
        self.active_lbl.setText("—")
        self._update_sample_selector()
        self.wave_editor.clear()
        self._refresh_octave_view()
        if loaded:
            self._enable_instrument_actions(True)
        elif instrument.unassigned:
            # No keys, but outliers to triage — allow recategorize to place them.
            self._enable_instrument_actions(True)
        else:
            self._enable_instrument_actions(False)
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
            self._enable_instrument_actions(True)
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    # ============================================================
    # kit: save / load / recategorize / export to Logic
    # ============================================================

    def _current_key_label(self, key: int) -> str:
        """Human label for a key: drum cap (KICK/SNR1/…) in drum-layout mode,
        else the note name with octave offset."""
        if self._result_is_classify:
            return KEY_CAP_LABEL.get(key, f"KEY {key}")
        note = NOTE_NAMES_12[key % 12]
        octave = key // 12
        return f"{note}+{octave}" if octave else note

    def _enable_instrument_actions(self, enabled: bool) -> None:
        self.recat_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)
        self.autotrim_btn.setEnabled(enabled)
        if not enabled and self.autotrim_btn.isChecked():
            self.autotrim_btn.setChecked(False)

    def _refresh_after_instrument_change(self, status_text: str) -> None:
        """Shared reset + redraw after the instrument's key mapping changes
        (kit load or recategorize). Releases held notes, clears per-key state,
        rebuilds labels, and repaints the octave view."""
        for n in list(self.piano._pressed):  # type: ignore[attr-defined]
            cluster_idx = n + self._octave * OCTAVE_STEP
            self._engine.release_note(cluster_idx)
            self.piano.release_note(n)
        self._all_segments = (
            [s for segs in self._instrument.notes.values() for s in segs]
            + list(self._instrument.unassigned))
        self._apply_loudness_gains()
        self._octave = 0
        self._sample_choice = {}
        self._selected_key = None
        self.piano.set_selected(None)
        self.active_led.setOn(False)
        self.active_lbl.setText("—")
        self._update_sample_selector()
        self.wave_editor.clear()
        self._refresh_octave_view()
        loaded = bool(self._instrument.loaded_notes())
        self._enable_instrument_actions(loaded)
        self.status.setText(status_text)

    @Slot()
    def _on_save_kit(self) -> None:
        name = self.kit_name_edit.text().strip()
        if not name:
            self.status.setText("enter a kit name before saving.")
            return
        if not self._instrument.loaded_notes():
            self.status.setText("nothing to save — run an analysis or load a kit first.")
            return
        try:
            kit_store.save_kit(
                name, self._instrument, self._result_is_classify,
                sample_choice=self._sample_choice,
                key_label=self._current_key_label,
            )
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"save failed: {exc}")
            return
        n = sum(len(v) for v in self._instrument.notes.values())
        self.status.setText(f"saved kit '{name}'  ·  {n} samples")

    @Slot()
    def _on_load_kit(self) -> None:
        dlg = KitBrowserDialog(self)
        if dlg.exec() != QDialog.Accepted or dlg.selected_path is None:
            return
        path = dlg.selected_path
        try:
            instrument, is_classify = kit_store.load_kit(
                path, playback_factory=PlaybackSettings)
            sample_choice = kit_store.load_sample_choice(path)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"load failed: {exc}")
            return
        self._instrument = instrument
        self._result_is_classify = is_classify
        kit_display_name = Path(path).name
        self.kit_name_edit.setText(kit_display_name)
        self._refresh_after_instrument_change(f"loaded kit '{kit_display_name}'")
        # Restore the arrow-tab sample choice after the reset.
        self._sample_choice = sample_choice

    @Slot()
    def _on_recategorize(self) -> None:
        if not self._instrument.loaded_notes() and not self._instrument.unassigned:
            self.status.setText("nothing to recategorize — run an analysis or load a kit first.")
            return
        dlg = RecategorizeDialog(
            self._instrument, self._engine,
            get_gain=lambda: self.vel_slider.value() / 127.0,
            is_classify=self._result_is_classify,
            key_label=self._current_key_label,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted or dlg.new_notes is None:
            return
        self._instrument.notes = dlg.new_notes
        self._instrument.unassigned = dlg.new_unassigned
        n_keys = len(self._instrument.loaded_notes())
        n_un = len(self._instrument.unassigned)
        extra = f"  ·  {n_un} unassigned" if n_un else ""
        self._refresh_after_instrument_change(
            f"recategorized  ·  {n_keys} keys{extra}")

    @Slot()
    def _on_export_logic(self) -> None:
        if not self._instrument.loaded_notes():
            self.status.setText("nothing to export — run an analysis or load a kit first.")
            return
        name = self.kit_name_edit.text().strip() or "Samplebot Kit"
        # Default into a single "Samplebot-3000" folder inside Logic's user
        # Sampler Instruments — all app kits group there (Logic → Sampler →
        # Samplebot-3000 → kit), instead of each kit making its own top-level
        # folder. Let the user redirect anywhere.
        start_dir = logic_export.LOGIC_SAMPLER_INSTRUMENTS / "Samplebot-3000"
        try:
            if start_dir.parent.parent.exists():
                start_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        if not start_dir.exists():
            start_dir = Path.home()
        dest = QFileDialog.getExistingDirectory(
            self, "Export kit to…", str(start_dir))
        if not dest:
            return
        # Mirror the app's SAMPLE PICK mode into the Logic layout: RANDOM →
        # round-robin groups (a different sample each hit); VEL LAYER → velocity
        # split. Single-sample keys are identical either way.
        round_robin = self._sample_mode == "random"
        # Bake each sample's playback effects (VOLUME trim, per-sample FILTER,
        # ENVELOPE) — plus equal-loudness when on — into the exported WAVs so the
        # Logic kit sounds like the app. Velocity and its live darkening lowpass
        # stay live (the exported kit responds to how hard you play).
        process_audio = self._make_export_processor()
        prog = lambda msg, frac: self._on_progress(f"export: {msg}", frac)
        # Hi-hat choke (from the reference instrument): F#1/G#1/A#1 — the GM hat
        # notes 42/44/46 — ALWAYS join exclusive class 1 with mono voices, so
        # closed/pedal hats cut the open hat's ring. Those slots are hats in
        # every layout this app produces (and per the user: "they will typically
        # always be high hats"). Every other key keeps full polyphony + release.
        choke_notes = {
            logic_export.BASE_MIDI_NOTE + k
            for k, fam, _lab in DRUM_KEYMAP if fam == "hat"
        }
        try:
            zones = logic_export.instrument_to_zonespecs(
                self._instrument.notes, key_label=self._current_key_label,
                round_robin=round_robin, process_audio=process_audio)
            exs_path = logic_export.write_exs_kit(
                name, zones, self._instrument.sample_rate, dest,
                choke_notes=choke_notes, progress=prog)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"export failed: {exc}")
            return
        into_logic = str(logic_export.LOGIC_SAMPLER_INSTRUMENTS) in str(exs_path)
        where = "Logic → Sampler → instrument menu" if into_logic else str(exs_path.parent)
        layout = "round-robin" if round_robin else "velocity-split"
        loud = "  ·  equal-loudness" if self._equal_loudness else ""
        choked = "  ·  hat choke" if choke_notes else ""
        self.status.setText(f"exported '{name}'  ·  {layout}{loud}{choked}  ·  {where}")

    def _make_export_processor(self):
        """Return process_audio(seg, raw) that bakes a sample's stored playback
        settings into its audio the same way _play_selected_sample renders them:
        VOLUME trim × equal-loudness gain, the A→D→S envelope, then the per-sample
        lowpass. Velocity (and its darkening lowpass) is a live hit control, so it
        is NOT baked. Samples left at defaults come out unchanged."""
        sr = int(self._instrument.sample_rate)
        equal_loudness = self._equal_loudness

        def process(seg, raw):
            st = self._settings_for(seg)
            audio = np.asarray(raw, dtype=np.float32).ravel().copy()
            gain = getattr(st, "volume", 100) / 100.0
            if equal_loudness:
                gain *= float(getattr(seg, "loudness_gain", 1.0))
            if gain != 1.0:
                audio *= np.float32(gain)
            audio = bake_adsr(audio, st.a * 0.01, st.d * 0.01, st.s / 100.0, sr)
            audio = bake_lowpass(audio, self._lpf_from_slider(st.lpf), sr)
            return audio

        return process

    @Slot()
    def _on_adsr_changed(self) -> None:
        a, d, s, r = self._adsr_values()
        self.adsr_viz.set_adsr(a, d, s, r)
        self._store_controls_to_active()

    @Slot()
    def _on_lpf_changed(self) -> None:
        self._store_controls_to_active()

    @Slot()
    def _on_playback_control_changed(self) -> None:
        self._store_controls_to_active()

    @Slot(bool)
    def _on_loop_toggled(self, checked: bool) -> None:
        self._store_controls_to_active()

    @Slot(bool)
    def _on_equal_loudness_toggled(self, checked: bool) -> None:
        # Gains are precomputed per kit; the toggle just enables applying them.
        self._equal_loudness = checked

    def _apply_loudness_gains(self) -> None:
        """(Re)compute A-weighted equal-loudness gains over the whole current kit
        (all keys + unassigned) and stash one on each segment. Cheap; run after
        any analysis / kit load / recategorize so the gains match the sample set."""
        segs = [s for v in self._instrument.notes.values() for s in v]
        segs += list(self._instrument.unassigned)
        if not segs:
            return
        gains = equal_loudness_gains(
            [s.audio for s in segs], self._instrument.sample_rate)
        for s, g in zip(segs, gains):
            s.loudness_gain = float(g)

    def _set_sample_mode(self, mode: str) -> None:
        """Switch the per-hit sample-pick mode (RANDOM ↔ VEL LAYER) and refresh
        the ◀ N/M ▶ readout so it reflects what drives playback."""
        self._sample_mode = mode
        self._update_sample_selector()

    @Slot(bool)
    def _on_per_sample_edit_toggled(self, checked: bool) -> None:
        """Flip volume/envelope/filter edits between whole-cluster (off) and
        single active sample (on). Reload the controls from the active sample so
        what's shown matches what the next edit will write."""
        self._per_sample_edit = checked
        seg = self._active_segment()
        if seg is not None:
            self._load_settings_into_controls(self._settings_for(seg))

    # ---------- per-sample playback settings ----------

    def _active_segment(self):
        """The Segment for the selected key + chosen sample, or None."""
        key = self._selected_key
        if key is None:
            return None
        segs = self._instrument.notes.get(key)
        if not segs:
            return None
        idx = self._sample_choice.get(key, 0) % len(segs)
        return segs[idx]

    def _edit_target_segments(self) -> list:
        """Segments the VOLUME/ENVELOPE/FILTER controls write to. Whole cluster
        (the selected key's samples) by default; just the active sample when
        PER-SAMPLE EDIT is on."""
        if self._per_sample_edit:
            seg = self._active_segment()
            return [seg] if seg is not None else []
        key = self._selected_key
        segs = self._instrument.notes.get(key) if key is not None else None
        return list(segs) if segs else []

    def _settings_for(self, seg) -> PlaybackSettings:
        """Get (creating if needed) the PlaybackSettings stored on a segment."""
        if seg.playback is None:
            seg.playback = PlaybackSettings()
        return seg.playback

    def _store_controls_to_active(self) -> None:
        """Write the current volume/envelope/filter/loop controls onto the edit
        target (whole cluster, or the active sample when PER-SAMPLE EDIT is on).

        Velocity is a live, global hit control now (both sample-pick modes use
        the slider as the hit velocity), so it's never stamped per-sample — each
        segment keeps whatever velocity it already had."""
        if self._loading_settings:
            return
        for seg in self._edit_target_segments():
            cur = self._settings_for(seg)
            seg.playback = PlaybackSettings(
                velocity=cur.velocity,
                loop=self.loop_check.isChecked(),
                a=self.a_slider.value(),
                d=self.d_slider.value(),
                s=self.s_slider.value(),
                r=self.r_slider.value(),
                lpf=self.lpf_slider.value(),
                volume=self.volume_slider.value(),
            )

    def _load_settings_into_controls(self, st: PlaybackSettings) -> None:
        """Push a sample's settings into the controls without writing back.
        Velocity is a global live control, so key/sample selection never moves
        the velocity slider."""
        self._loading_settings = True
        try:
            self.loop_check.setChecked(st.loop)
            self.a_slider.setValue(st.a)
            self.d_slider.setValue(st.d)
            self.s_slider.setValue(st.s)
            self.r_slider.setValue(st.r)
            self.lpf_slider.setValue(st.lpf)
            self.volume_slider.setValue(getattr(st, "volume", 100))
        finally:
            self._loading_settings = False
        a, d, s, r = self._adsr_values()
        self.adsr_viz.set_adsr(a, d, s, r)

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

    def _set_sort_mode(self, mode: str) -> None:
        """Switch cluster→key sorting. FREQUENCY re-sorts live (cheap). AUTO
        re-applies the GM one-per-key placement if the clusters were already
        auto-sorted (labels present); otherwise it just sets the mode for the next
        RUN (the auto-sorter needs a fresh pass to label clusters)."""
        self._sort_mode = mode
        if not self._all_segments:
            return
        # Cluster→key mapping is about to change — release anything held.
        for n in list(self.piano._pressed):  # type: ignore[attr-defined]
            self._engine.release_note(n + self._octave * OCTAVE_STEP)
            self.piano.release_note(n)
        if mode == "frequency":
            self._instrument = assign_keys_to_clusters(
                self._all_segments, sort_by_freq=True)
            self._result_is_classify = False
        else:  # auto
            if any(getattr(s, "label", "") for s in self._all_segments):
                self._instrument = build_instrument_drumkeys_gm(self._all_segments)
                self._result_is_classify = True
            else:
                self.status.setText("AUTO sort — press RUN ANALYSIS to auto-sort onto GM keys.")
                return
        self._octave = 0
        self._apply_loudness_gains()
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
        # In AUTO-sort mode each loaded key shows its GM cap (KICK, SNR, HH-C, …).
        # The visible index n maps to the ABSOLUTE GM key n+base for its label, so
        # higher-octave keys (e.g. BELL/RIDE2) label correctly. Freq-sort: no labels.
        if self._result_is_classify:
            key_labels = {n: KEY_CAP_LABEL[n + base] for n in loaded_in_octave
                          if (n + base) in KEY_CAP_LABEL}
        else:
            key_labels = {}
        self.piano.set_key_labels(key_labels)
        self.oct_value_lbl.setText(f"{self._octave} / {max_o}")
        self.oct_down_btn.setEnabled(self._octave > 0)
        self.oct_up_btn.setEnabled(self._octave < max_o)

    # ============================================================
    # note triggering — sample cycling + velocity volume/lowpass
    # ============================================================

    @Slot(int)
    def _on_note_pressed(self, note_index: int) -> None:
        cluster_idx = note_index + self._octave * OCTAVE_STEP
        segs = self._instrument.notes.get(cluster_idx)
        if not segs:
            return

        # Pressing a key selects it (for arrows / sequencer / active light).
        self._select_key(cluster_idx, note_index)
        self._play_selected_sample(cluster_idx)

    def _velocity_layer_index(self, n: int, vel: Optional[int] = None) -> int:
        """Map a velocity (1..127) to a stacked-sample index (0..n-1). Segments
        are pre-sorted by loudness (rms), so soft→loud picks low→high. Defaults to
        the VELOCITY slider; a live hit (e.g. MIDI) passes its own velocity."""
        v = self.vel_slider.value() if vel is None else vel
        idx = int((v - 1) / 126.0 * n)
        return max(0, min(n - 1, idx))

    def _play_selected_sample(self, cluster_idx: int,
                              force_idx: Optional[int] = None,
                              velocity: Optional[int] = None) -> None:
        """Play a sample on a key using that sample's own playback settings.

        Sample choice:
          - `force_idx` given   → play exactly that index (arrow auditioning).
          - VEL LAYER mode      → the hit velocity picks the sample (by rms).
          - RANDOM mode         → a random sample from the cluster fires.
          - single-sample key   → that one sample.
        Hit velocity is the VELOCITY slider by default, or `velocity` when a live
        controller (MIDI keyboard) supplies a real per-hit velocity; it drives
        gain, the darkening lowpass, and VEL LAYER selection.
        """
        segs = self._instrument.notes.get(cluster_idx)
        if not segs:
            return
        vel = self.vel_slider.value() if velocity is None else velocity
        if force_idx is not None:
            idx = force_idx % len(segs)
        elif len(segs) > 1 and self._sample_mode == "velocity":
            idx = self._velocity_layer_index(len(segs), vel)
        elif len(segs) > 1 and self._sample_mode == "random":
            idx = random.randrange(len(segs))
        else:
            idx = self._sample_choice.get(cluster_idx, 0) % len(segs)
        seg = segs[idx]
        st = self._settings_for(seg)
        # Live hit velocity × the sample's own stored VOLUME trim, × the
        # A-weighted equal-loudness gain when EQUAL LOUDNESS is on.
        gain = (vel / 127.0) * (getattr(st, "volume", 100) / 100.0)
        if self._equal_loudness:
            gain *= float(getattr(seg, "loudness_gain", 1.0))
        adsr = (st.a * 0.01, st.d * 0.01, st.s / 100.0, st.r * 0.02)
        cutoff_hz = self._lpf_from_slider(st.lpf)
        # Velocity also applies the subtle darkening lowpass on top of the
        # per-sample filter.
        audio = velocity_lowpass(seg.audio, vel)
        self._engine.play(audio, gain=gain, note_id=cluster_idx,
                          adsr=adsr, cutoff_hz=cutoff_hz, loop=st.loop)

    def _select_key(self, cluster_idx: int, note_index: int) -> None:
        """Mark a key active: outline it, light the indicator, wire the arrow
        sample-selector, load the active sample's settings, show its pattern."""
        self._selected_key = cluster_idx
        self.piano.set_selected(note_index)
        # Cap label comes from the ABSOLUTE GM key index (spans octaves), not the
        # visible index — so a selected BELL/RIDE2 in a higher octave labels right.
        cap = KEY_CAP_LABEL.get(cluster_idx, f"KEY {cluster_idx + 1}")
        self.active_led.setOn(True)
        self.active_lbl.setText(cap)
        self._update_sample_selector()
        seg = self._active_segment()
        if seg is not None:
            self._load_settings_into_controls(self._settings_for(seg))
        self._show_active_in_editor()

    def _update_sample_selector(self) -> None:
        """Refresh the ◀ N/M ▶ readout for the selected key. In VEL LAYER mode the
        slider (not the arrows) drives playback, so the arrows are an auditioning
        aid and the readout shows the live velocity-selected layer. In RANDOM mode
        the readout shows the arrow-chosen sample (used for auditioning/editing)."""
        key = self._selected_key
        segs = self._instrument.notes.get(key) if key is not None else None
        n = len(segs) if segs else 0
        has = n > 0
        self.sample_prev_btn.setEnabled(has and n > 1)
        self.sample_next_btn.setEnabled(has and n > 1)
        if not has:
            self.sample_lbl.setText("– / –")
            return
        if self._sample_mode == "velocity" and n > 1:
            idx = self._velocity_layer_index(n)
            self.sample_lbl.setText(f"vel {idx + 1} / {n}")
        else:
            idx = self._sample_choice.get(key, 0) % n
            self.sample_lbl.setText(f"{idx + 1} / {n}")

    def _step_sample(self, delta: int) -> None:
        """Tab the chosen sample for the active key and audition it."""
        key = self._selected_key
        segs = self._instrument.notes.get(key) if key is not None else None
        if not segs:
            return
        n = len(segs)
        idx = (self._sample_choice.get(key, 0) + delta) % n
        self._sample_choice[key] = idx
        self._update_sample_selector()
        # Load the newly chosen sample's settings into the controls + editor.
        seg = self._active_segment()
        if seg is not None:
            self._load_settings_into_controls(self._settings_for(seg))
        self._show_active_in_editor()
        # Audition exactly the arrow-chosen sample, regardless of sample-pick
        # mode (which would otherwise pick by velocity or at random on a key press).
        self._play_selected_sample(key, force_idx=idx)

    # ============================================================
    # per-sample waveform editor
    # ============================================================

    def _show_active_in_editor(self) -> None:
        """Load the active (arrow-chosen) sample into the waveform editor."""
        self.wave_editor.set_segment(self._active_segment(),
                                     self._instrument.sample_rate)
        if getattr(self, "autotrim_btn", None) and self.autotrim_btn.isChecked():
            self._autotrim_preview()   # keep the preview on the newly-shown sample

    def _on_wave_bounds_changed(self, start: int, end: int) -> None:
        """Live drag: retrim the active sample so playback/export follow instantly."""
        seg = self._active_segment()
        if seg is not None:
            P.reslice_segment(seg, start, end)

    def _on_wave_edit_committed(self, start: int, end: int) -> None:
        """Drag released: finalize the trim and audition the result."""
        seg = self._active_segment()
        if seg is None:
            return
        P.reslice_segment(seg, start, end)
        self._update_sample_selector()
        if self._selected_key is not None:
            idx = self._sample_choice.get(self._selected_key, 0)
            self._play_selected_sample(self._selected_key, force_idx=idx)

    # ---------- autotrim (batch) ----------

    def _autotrim_params(self) -> tuple[float, float]:
        """The two generation knobs, read live: (sensitivity 0..1, trim dB)."""
        return (self.sens_slider.value() / 100.0,
                self._trim_db_from_slider(self.trim_slider.value()))

    def _on_autotrim_toggled(self, on: bool) -> None:
        """Arm/disarm autotrim: light the two knobs, live-preview on the current
        sample as they're tuned, and reveal APPLY TO CLUSTER."""
        self.autotrim_apply_btn.setVisible(on)
        hl = "color:#ff8a1e; font-weight:bold;" if on else ""
        self.sens_label.setStyleSheet(hl)
        self.trim_label.setStyleSheet(hl)
        self.autotrim_hint.setText(
            "Tune TRANSIENT SENS + TRIM THRESHOLD (cyan lines preview this "
            "sample), then APPLY TO CLUSTER; select another key for the next."
            if on else "")
        if on:
            self.sens_slider.valueChanged.connect(self._autotrim_preview)
            self.trim_slider.valueChanged.connect(self._autotrim_preview)
            self._autotrim_preview()
        else:
            for sl in (self.sens_slider, self.trim_slider):
                try:
                    sl.valueChanged.disconnect(self._autotrim_preview)
                except (TypeError, RuntimeError):
                    pass
            self.wave_editor.set_preview(None)

    def _autotrim_preview(self, *_args) -> None:
        """Overlay the proposed autotrim bounds on the active sample."""
        seg = self._active_segment()
        if seg is None:
            self.wave_editor.set_preview(None)
            return
        P.ensure_source(seg)
        start, end = P.autotrim_bounds(seg.source, *self._autotrim_params())
        self.wave_editor.set_preview(start, end)

    def _apply_autotrim_cluster(self) -> None:
        """Retrim every sample on the SELECTED KEY's cluster (start→transient,
        end→next transient) with the current knobs, then audition the active one.
        Stays armed so you can select another key and judge each cluster on its
        own — nuance over a blunt whole-kit batch."""
        key = self._selected_key
        segs = self._instrument.notes.get(key) if key is not None else None
        if not segs:
            return
        sens, trim = self._autotrim_params()
        for seg in segs:
            P.ensure_source(seg)
            P.reslice_segment(seg, *P.autotrim_bounds(seg.source, sens, trim))
        # Reload so the handles snap to the committed bounds.
        self._show_active_in_editor()
        self._update_sample_selector()
        idx = self._sample_choice.get(key, 0)
        self._play_selected_sample(key, force_idx=idx)

    @Slot(int)
    def _on_note_released(self, note_index: int) -> None:
        cluster_idx = note_index + self._octave * OCTAVE_STEP
        self._engine.release_note(cluster_idx)

    # ============================================================
    # USB MIDI
    # ============================================================

    @Slot(int, int)
    def _on_midi_note_on(self, note: int, velocity: int) -> None:
        """MIDI note → cluster (note − BASE_MIDI_NOTE, matching the Logic export),
        played with the real MIDI velocity. Playing never steals the edit focus —
        it only lights the on-screen key if the note is in the visible octave."""
        cluster_idx = note - BASE_MIDI_NOTE
        if cluster_idx < 0 or not self._instrument.notes.get(cluster_idx):
            return
        note_index = cluster_idx - self._octave * OCTAVE_STEP
        if 0 <= note_index < N_VISIBLE_KEYS:
            self.piano.show_pressed(note_index)
        self._play_selected_sample(cluster_idx, velocity=velocity)

    @Slot(int)
    def _on_midi_note_off(self, note: int) -> None:
        cluster_idx = note - BASE_MIDI_NOTE
        if cluster_idx < 0:
            return
        self._engine.release_note(cluster_idx)
        note_index = cluster_idx - self._octave * OCTAVE_STEP
        if 0 <= note_index < N_VISIBLE_KEYS:
            self.piano.show_released(note_index)

    def _poll_midi(self) -> None:
        """Connect the first MIDI input, and detect plug/unplug, on a timer."""
        if not MidiListener.available():
            self.midi_lbl.setText("MIDI: (install python-rtmidi)")
            return
        if self._midi.is_connected() and not self._midi.still_present():
            self._midi.disconnect()          # device was unplugged
        if not self._midi.is_connected():
            self._midi.connect_first()
        name = self._midi.port_name
        if name:
            # The OS names Logic's virtual MIDI port after whichever Logic is
            # installed ("Logic Pro Trial Virtual Out"); show it as plain Logic Pro.
            name = name.replace("Logic Pro Trial", "Logic Pro")
        self.midi_lbl.setText(f"MIDI ● {name}" if name else "MIDI: —")

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
        self._midi_timer.stop()
        self._midi.disconnect()
        self._engine.stop()
        super().closeEvent(event)
