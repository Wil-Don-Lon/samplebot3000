"""Recategorize dialog — a single drag-and-drop list of all samples.

One flat list holds every sample, grouped under a header row per key/cluster.
Drag a sample (or a shift/ctrl-selected bunch, even from different clusters) and
drop it under another cluster's header to reassign it; drop under ✕ REMOVE to
drop it from the kit. Click a sample once to audition it. A sample's cluster is
simply whichever header it sits under — Apply walks the list top-to-bottom to
rebuild the key -> samples mapping.

Self-contained: all drag/drop, selection, and audition wiring live here; the
host GUI only hands over the instrument, an audio engine, a gain getter, an
is_classify flag, and a key-label function.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QCursor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QAbstractItemView, QWidget,
)

from pipeline import (
    Instrument, Segment, NOTE_NAMES_12, KEY_CAP_LABEL, TARGET_SR, OUTLIER_CLUSTER,
)
from audio_engine import AudioEngine

_REMOVE_KEY = -1                 # header sentinel: samples under it are dropped
_UNASSIGNED_KEY = -2             # header sentinel: samples parked, unexported

_KIND_ROLE = Qt.UserRole         # "header" | "sample"
_DATA_ROLE = Qt.UserRole + 1     # header: key int; sample: seg id int
_DISP_ROLE = Qt.UserRole + 2     # header: base display string

_HEADER_COLOR = QColor("#ffb15a")
_REMOVE_COLOR = QColor("#ff6a6a")
_UNASSIGNED_COLOR = QColor("#8a93a6")
_HEADER_BG = QColor("#1a130b")
_SAMPLE_COLOR = QColor("#d8cdbb")


class RecatList(QListWidget):
    """Single internal-move list. Headers are fixed dividers; samples drag."""

    def __init__(self, owner: "RecategorizeDialog") -> None:
        super().__init__()
        self._owner = owner
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setObjectName("recatList")
        # Pixel-granular scrolling so the drag auto-scroll below is smooth.
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        # Qt's built-in drag auto-scroll only fires in a tiny edge margin and
        # dies the moment the cursor leaves the viewport. Replace it with a timer
        # that reads the live cursor position while a drag is in flight.
        self.setAutoScroll(False)
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setInterval(30)
        self._scroll_timer.timeout.connect(self._auto_scroll_tick)
        self.itemClicked.connect(self._owner._on_item_clicked)

    def startDrag(self, supportedActions) -> None:
        # startDrag runs a blocking nested loop (QDrag.exec) until the drop, so
        # the timer ticks throughout the drag; stop it when the drag ends.
        self._scroll_timer.start()
        try:
            super().startDrag(supportedActions)
        finally:
            self._scroll_timer.stop()

    @staticmethod
    def _auto_scroll_delta(y: int, h: int) -> int:
        """Pixels to scroll for a cursor at viewport-y `y` (height `h`). Scrolls
        once the cursor enters the top/bottom eighth, ramping with distance, and
        keeps going (at full speed) when it's beyond the edge (y<0 or y>h)."""
        margin = max(24, h // 8)              # the top/bottom eighth
        if y < margin:
            over, direction = margin - y, -1
        elif y > h - margin:
            over, direction = y - (h - margin), 1
        else:
            return 0
        frac = min(1.0, over / float(margin))  # 1.0 at/past the edge
        return direction * max(2, int(frac * 36))

    def _auto_scroll_tick(self) -> None:
        vp = self.viewport()
        delta = self._auto_scroll_delta(vp.mapFromGlobal(QCursor.pos()).y(),
                                        vp.height())
        if delta:
            bar = self.verticalScrollBar()
            bar.setValue(bar.value() + delta)

    def keyPressEvent(self, event) -> None:
        # Plain Up/Down jump sample→sample (skipping the header dividers) and
        # audition whatever they land on. Shift/Ctrl+arrow fall through to the
        # default range/extend selection. Arrowing only moves the selection — it
        # never edits or renames the item text.
        plain = not (event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier))
        if plain and event.key() in (Qt.Key_Up, Qt.Key_Down):
            step = 1 if event.key() == Qt.Key_Down else -1
            target = self._next_sample_row(self.currentRow(), step)
            if target is not None:
                self.setCurrentRow(target)
                self._owner._on_item_clicked(self.currentItem())
            return
        super().keyPressEvent(event)

    def _next_sample_row(self, start: int, step: int) -> Optional[int]:
        """Row of the next sample from `start` in direction `step`, or None if
        there's no sample that way (so navigation just stops)."""
        r = start + step
        while 0 <= r < self.count():
            if self.item(r).data(_KIND_ROLE) == "sample":
                return r
            r += step
        return None

    def dropEvent(self, event) -> None:
        super().dropEvent(event)        # native internal move of selected rows
        self._owner._after_move()


class RecategorizeDialog(QDialog):
    def __init__(
        self,
        instrument: Instrument,
        engine: AudioEngine,
        get_gain: Callable[[], float],
        is_classify: bool,
        key_label: Callable[[int], str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recategorize Samples")
        self.resize(560, 720)
        self._instrument = instrument
        self._engine = engine
        self._get_gain = get_gain
        self._is_classify = is_classify
        self._key_label = key_label
        self._by_id: dict[int, Segment] = {}     # stable id -> Segment
        self._first_key: Optional[int] = None    # topmost real cluster (Apply fallback)
        # Populated on Apply; None means cancelled.
        self.new_notes: Optional[dict[int, list[Segment]]] = None
        self.new_unassigned: list[Segment] = []
        self._build_ui()

    # ---------- construction ----------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        loaded = sorted(self._instrument.loaded_notes())
        unassigned = list(self._instrument.unassigned)
        total = sum(len(self._instrument.notes[k]) for k in loaded) + len(unassigned)
        header = QLabel(f"{total} sample{'s' if total != 1 else ''} · {len(loaded)} keys")
        header.setObjectName("dialogHeader")
        root.addWidget(header)

        sub = QLabel(
            "Drag samples under another cluster's header to move them. Shift/Ctrl-"
            "click to grab several (even across clusters) and drag them together. "
            "Samples under UNASSIGNED have no key and aren't exported to Logic — "
            "drag them onto a cluster to place them. Drop under ✕ REMOVE to drop a "
            "sample. Click a sample to hear it. Nothing changes until APPLY."
        )
        sub.setWordWrap(True)
        sub.setObjectName("dialogSubtle")
        root.addWidget(sub)

        # Assign every segment a stable id.
        next_id = 0
        seg_of_key: dict[int, list[int]] = {}
        for key in loaded:
            ids: list[int] = []
            for seg in self._instrument.notes[key]:
                self._by_id[next_id] = seg
                ids.append(next_id)
                next_id += 1
            seg_of_key[key] = ids
        unassigned_ids: list[int] = []
        for seg in unassigned:
            self._by_id[next_id] = seg
            unassigned_ids.append(next_id)
            next_id += 1

        # Header keys: in classify mode offer every drum type (some may be empty
        # so you can move a sample onto them); in cluster mode only the existing
        # clusters. UNASSIGNED then REMOVE sit at the bottom.
        if self._is_classify:
            header_keys = sorted(KEY_CAP_LABEL)
        else:
            header_keys = list(loaded)
        self._first_key = header_keys[0] if header_keys else None

        self.list = RecatList(self)
        for key in header_keys:
            self.list.addItem(self._make_header(key, self._key_label(key)))
            for sid in seg_of_key.get(key, []):
                self.list.addItem(self._make_sample(sid))
        self.list.addItem(self._make_header(_UNASSIGNED_KEY, "Unassigned"))
        for sid in unassigned_ids:
            self.list.addItem(self._make_sample(sid))
        self.list.addItem(self._make_header(_REMOVE_KEY, "✕ Remove"))
        root.addWidget(self.list, 1)
        self._after_move()   # stamp initial header counts

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("APPLY")
        apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        root.addLayout(btn_row)

    def _make_header(self, key: int, disp: str) -> QListWidgetItem:
        it = QListWidgetItem()
        it.setData(_KIND_ROLE, "header")
        it.setData(_DATA_ROLE, int(key))
        it.setData(_DISP_ROLE, disp)
        # Fixed divider: not selectable, not draggable — only samples move.
        it.setFlags(Qt.ItemIsEnabled)
        font = it.font()
        font.setBold(True)
        it.setFont(font)
        color = (_REMOVE_COLOR if key == _REMOVE_KEY
                 else _UNASSIGNED_COLOR if key == _UNASSIGNED_KEY
                 else _HEADER_COLOR)
        it.setForeground(color)
        it.setBackground(_HEADER_BG)
        return it

    def _make_sample(self, sid: int) -> QListWidgetItem:
        seg = self._by_id[sid]
        it = QListWidgetItem(self._sample_text(seg, sid))
        it.setData(_KIND_ROLE, "sample")
        it.setData(_DATA_ROLE, int(sid))
        it.setForeground(_SAMPLE_COLOR)
        # Drag-enabled, selectable; NOT drop-enabled so drops land between rows
        # (never "onto" a sample).
        it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
        return it

    # ---------- rendering helpers ----------

    @staticmethod
    def _sample_text(seg: Segment, sid: int) -> str:
        tag = seg.label.upper() if getattr(seg, "label", "") else f"#{sid + 1}"
        dur = (seg.audio.size / float(TARGET_SR)) if seg.audio is not None else 0.0
        return f"      • {tag}    ·  {dur:.2f}s   ·  rms {getattr(seg, 'rms', 0.0):.3f}"

    @staticmethod
    def _header_text(disp: str, n: int) -> str:
        return f"{disp.upper()}   ·  {n} sample{'s' if n != 1 else ''}"

    def _after_move(self) -> None:
        """Re-derive each header's sample count from the current row order."""
        cur: Optional[QListWidgetItem] = None
        n = 0
        for r in range(self.list.count()):
            it = self.list.item(r)
            if it.data(_KIND_ROLE) == "header":
                if cur is not None:
                    cur.setText(self._header_text(cur.data(_DISP_ROLE), n))
                cur, n = it, 0
            else:
                n += 1
        if cur is not None:
            cur.setText(self._header_text(cur.data(_DISP_ROLE), n))

    # ---------- interaction ----------

    def _on_item_clicked(self, item: Optional[QListWidgetItem]) -> None:
        """Audition a sample — from a click or from arrow-key navigation. Headers
        (and a null current item) are ignored."""
        if item is None or item.data(_KIND_ROLE) != "sample":
            return
        seg = self._by_id.get(item.data(_DATA_ROLE))
        if seg is not None:
            self._engine.play(seg.audio, gain=self._get_gain(), note_id=-1)

    def _on_apply(self) -> None:
        new_notes: dict[int, list[Segment]] = {}
        new_unassigned: list[Segment] = []
        current: Optional[int] = None
        for r in range(self.list.count()):
            it = self.list.item(r)
            if it.data(_KIND_ROLE) == "header":
                current = it.data(_DATA_ROLE)
                continue
            # A sample dropped above the first header falls back to it.
            key = current if current is not None else self._first_key
            seg = self._by_id.get(it.data(_DATA_ROLE))
            if seg is None or key == _REMOVE_KEY:
                continue                       # removed / dropped from the kit
            if key == _UNASSIGNED_KEY or key is None:
                seg.cluster = OUTLIER_CLUSTER   # parked: no key, not exported
                new_unassigned.append(seg)
            else:
                seg.cluster = int(key)          # keep cluster consistent with placement
                new_notes.setdefault(int(key), []).append(seg)
        # Soft -> loud so velocity layers / the arrow selector stay meaningful.
        for k in new_notes:
            new_notes[k].sort(key=lambda s: getattr(s, "rms", 0.0))
        new_unassigned.sort(key=lambda s: getattr(s, "rms", 0.0))
        self.new_notes = new_notes
        self.new_unassigned = new_unassigned
        self.accept()
