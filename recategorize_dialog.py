"""Recategorize dialog — a single drag-and-drop list of all samples.

One flat list holds every sample, grouped under a header row per key/cluster.
Drag a sample (or a shift/ctrl-selected bunch, even from different clusters) and
drop it under another cluster's header to reassign it; drop under ✕ REMOVE to
drop it from the kit. Click a sample once to audition it. A sample's cluster is
simply whichever header it sits under — Apply walks the list top-to-bottom to
rebuild the key -> samples mapping.

To move a WHOLE cluster to a different key, each cluster header carries an
"Assign to Key" button: click it to arm that cluster, then press the computer
key you want it on (the same A/W/S/E/D… piano layout as the main window, with
Z/X to drop/raise the octave). Landing on an occupied key SWAPS the two
clusters. This beats a key picker because it reaches every octave with one
keystroke and mirrors how you actually play the kit.

Self-contained: all drag/drop, selection, audition, and key-capture wiring live
here; the host GUI only hands over the instrument, an audio engine, a gain
getter, an is_classify flag, and a key-label function.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QKeyEvent
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QAbstractItemView, QWidget,
)

from pipeline import (
    Instrument, Segment, KEY_CAP_LABEL, TARGET_SR, OUTLIER_CLUSTER, MAX_KEYS,
)
from audio_engine import AudioEngine

# The main window's computer-keyboard → note-index map, copied here because
# importing it from gui.py would be circular (gui imports this dialog). 17 keys
# per octave (C … E-of-next); Z/X shift the octave, so key = note + oct*12.
_KEY_TO_NOTE = {
    Qt.Key_A: 0,  Qt.Key_W: 1,  Qt.Key_S: 2,  Qt.Key_E: 3,  Qt.Key_D: 4,
    Qt.Key_F: 5,  Qt.Key_T: 6,  Qt.Key_G: 7,  Qt.Key_Y: 8,  Qt.Key_H: 9,
    Qt.Key_U: 10, Qt.Key_J: 11, Qt.Key_K: 12, Qt.Key_O: 13, Qt.Key_L: 14,
    Qt.Key_P: 15, Qt.Key_Semicolon: 16,
}
_OCTAVE_STEP = 12
_MAX_OCTAVE = (MAX_KEYS - 1) // _OCTAVE_STEP

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
        self._hdr_lbl: dict[int, QLabel] = {}    # cluster key -> its header label widget
        self._hdr_btn: dict[int, QPushButton] = {}   # cluster key -> its "Assign to Key" button
        self._armed_key: Optional[int] = None    # cluster awaiting a key-press (None = idle)
        self._assign_octave = 0                  # octave offset applied to the next key-press
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
            self._add_header(key)
            for sid in seg_of_key.get(key, []):
                self.list.addItem(self._make_sample(sid))
        self._add_header(_UNASSIGNED_KEY, "Unassigned")
        for sid in unassigned_ids:
            self.list.addItem(self._make_sample(sid))
        self._add_header(_REMOVE_KEY, "✕ Remove")
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

    def _add_header(self, key: int, disp: Optional[str] = None,
                    at: Optional[int] = None) -> QListWidgetItem:
        """Create a header row for `key`, add it (append, or insert at `at`), and
        for a real cluster key attach its 'Assign to Key' button widget."""
        if disp is None:
            disp = (self._key_label(key) if key >= 0
                    else "Unassigned" if key == _UNASSIGNED_KEY else "✕ Remove")
        it = self._make_header(key, disp)
        if at is None:
            self.list.addItem(it)
        else:
            self.list.insertItem(at, it)
        if key >= 0:                       # only cluster headers get the arm button
            w = self._make_header_widget(key, disp)
            it.setSizeHint(w.sizeHint())
            self.list.setItemWidget(it, w)
        return it

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

    def _make_header_widget(self, key: int, disp: str) -> QWidget:
        """Row widget for a cluster header: its label plus the arm button that
        starts key-capture for moving the whole cluster."""
        w = QWidget()
        w.setStyleSheet("background:#1a130b;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(8, 3, 8, 3)
        lay.setSpacing(8)
        lbl = QLabel(self._header_text(disp, 0))
        lbl.setStyleSheet("color:#ffb15a; font-weight:bold; background:transparent;")
        btn = QPushButton("Assign to Key")
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            "QPushButton { color:#ffb15a; background:#241a0e; border:1px solid #5a3f1e;"
            " border-radius:4px; padding:2px 10px; font-weight:bold; }"
            "QPushButton:hover { border-color:#ffb15a; }"
            "QPushButton:checked { color:#120c05; background:#ffb15a; border-color:#ffd089; }"
        )
        btn.clicked.connect(lambda checked, k=key: self._on_arm_clicked(k, checked))
        lay.addWidget(lbl, 1)
        lay.addWidget(btn, 0)
        self._hdr_lbl[key] = lbl
        self._hdr_btn[key] = btn
        return w

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

    def _set_header_text(self, item: QListWidgetItem, n: int) -> None:
        """Stamp a header's sample count — onto its label widget for cluster
        headers (which render via a widget), or the item text for the sentinels."""
        txt = self._header_text(item.data(_DISP_ROLE), n)
        lbl = self._hdr_lbl.get(item.data(_DATA_ROLE))
        if lbl is not None:
            lbl.setText(txt)
        else:
            item.setText(txt)

    def _after_move(self) -> None:
        """Re-derive each header's sample count from the current row order."""
        cur: Optional[QListWidgetItem] = None
        n = 0
        for r in range(self.list.count()):
            it = self.list.item(r)
            if it.data(_KIND_ROLE) == "header":
                if cur is not None:
                    self._set_header_text(cur, n)
                cur, n = it, 0
            else:
                n += 1
        if cur is not None:
            self._set_header_text(cur, n)

    # ---------- interaction ----------

    def _on_item_clicked(self, item: Optional[QListWidgetItem]) -> None:
        """Audition a sample — from a click or from arrow-key navigation. Headers
        (and a null current item) are ignored."""
        if item is None or item.data(_KIND_ROLE) != "sample":
            return
        seg = self._by_id.get(item.data(_DATA_ROLE))
        if seg is not None:
            self._engine.play(seg.audio, gain=self._get_gain(), note_id=-1)

    def _header_row_for(self, key: int) -> Optional[int]:
        for r in range(self.list.count()):
            it = self.list.item(r)
            if it.data(_KIND_ROLE) == "header" and it.data(_DATA_ROLE) == key:
                return r
        return None

    def _sids_under(self, key: int) -> list[int]:
        """Sample ids currently under `key`'s header (its cluster)."""
        hdr = self._header_row_for(key)
        if hdr is None:
            return []
        out = []
        r = hdr + 1
        while r < self.list.count() and self.list.item(r).data(_KIND_ROLE) == "sample":
            out.append(self.list.item(r).data(_DATA_ROLE))
            r += 1
        return out

    def _remove_samples(self, sids) -> None:
        sset = set(sids)
        for r in range(self.list.count() - 1, -1, -1):
            it = self.list.item(r)
            if it.data(_KIND_ROLE) == "sample" and it.data(_DATA_ROLE) in sset:
                self.list.takeItem(r)

    def _insert_under(self, key: int, sids) -> None:
        """Put `sids` under `key`'s header, creating the header if needed."""
        hdr = self._header_row_for(key)
        if hdr is None:
            insert = self._header_row_for(_UNASSIGNED_KEY)
            if insert is None:
                insert = self.list.count()
            self._add_header(key, at=insert)
            hdr = insert
        at = hdr + 1
        for sid in sids:
            self.list.insertItem(at, self._make_sample(sid))
            at += 1

    # ---------- cluster → key capture ----------

    def _on_arm_clicked(self, key: int, checked: bool) -> None:
        """A header's 'Assign to Key' button was toggled: arm (start capture) or
        disarm this cluster."""
        if checked:
            self._arm(key)
        else:
            self._disarm()

    def _arm(self, key: int) -> None:
        """Enter key-capture for the cluster on `key`: grab the keyboard so the
        next A/W/S/E/D… press (Z/X for octave) targets a destination key."""
        if self._armed_key is not None and self._armed_key != key:
            prev = self._hdr_btn.get(self._armed_key)
            if prev is not None:
                prev.setChecked(False)
        self._armed_key = key
        self._assign_octave = 0
        self.grabKeyboard()          # route every keystroke to this dialog
        self._refresh_arm_labels()

    def _disarm(self) -> None:
        if self._armed_key is None:
            return
        btn = self._hdr_btn.get(self._armed_key)
        if btn is not None:
            btn.setChecked(False)
        self._armed_key = None
        self.releaseKeyboard()
        self._refresh_arm_labels()

    def _refresh_arm_labels(self) -> None:
        """Light the armed cluster's button (showing the live octave) and reset
        every other button to its resting label."""
        for key, btn in self._hdr_btn.items():
            if key == self._armed_key:
                oct_txt = f"+{self._assign_octave}" if self._assign_octave else "0"
                btn.setText(f"⌨ press key · oct {oct_txt}")
                if not btn.isChecked():
                    btn.setChecked(True)
            else:
                btn.setText("Assign to Key")
                if btn.isChecked():
                    btn.setChecked(False)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """While a cluster is armed, capture the keystroke as its destination key
        instead of the dialog's normal handling. Z/X change octave; a note key
        commits the move (swapping on an occupied key); Esc cancels."""
        if self._armed_key is None:
            super().keyPressEvent(event)
            return
        k = event.key()
        if k == Qt.Key_Escape:
            self._disarm()
            return
        if k == Qt.Key_Z:
            self._assign_octave = max(0, self._assign_octave - 1)
            self._refresh_arm_labels()
            return
        if k == Qt.Key_X:
            self._assign_octave = min(_MAX_OCTAVE, self._assign_octave + 1)
            self._refresh_arm_labels()
            return
        note = _KEY_TO_NOTE.get(k)
        if note is None:
            return                   # swallow other keys while armed
        target = note + self._assign_octave * _OCTAVE_STEP
        src = self._armed_key
        self._disarm()
        if 0 <= target < MAX_KEYS and target != src:
            self._move_cluster(src, target)

    def _move_cluster(self, src: int, dst: int) -> None:
        """Move the WHOLE cluster on `src` to `dst`. If `dst` already holds a
        cluster, SWAP the two (dst's samples go to src)."""
        if src is None or dst is None or src == dst:
            return
        src_sids = self._sids_under(src)
        dst_sids = self._sids_under(dst)
        if not src_sids:
            return
        self._remove_samples(src_sids + dst_sids)
        self._insert_under(dst, src_sids)
        if dst_sids:
            self._insert_under(src, dst_sids)
        self._after_move()

    def done(self, result: int) -> None:
        # Never leave the app-wide keyboard grab dangling if the dialog closes
        # (Apply/Cancel) mid-capture.
        if self._armed_key is not None:
            self.releaseKeyboard()
        super().done(result)

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
