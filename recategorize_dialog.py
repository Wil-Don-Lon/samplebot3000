"""Recategorize dialog — reassign whole CLUSTERS to keys.

Each key holds one cluster (the auto-sorter places one cluster per key). This
dialog lists every cluster as a row and lets you send it to a different key via a
picker spanning all octaves. Moving a cluster onto an occupied key SWAPS the two.
Clusters can also be parked (Unassigned) or dropped (Remove). Click ▶ to audition.
Nothing changes until APPLY.

Self-contained: the host GUI hands over the instrument, an audio engine, a gain
getter, an is_classify flag, and a key-label function.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QScrollArea, QWidget, QFrame,
)

from pipeline import (
    Instrument, Segment, NOTE_NAMES_12, KEY_CAP_LABEL, OUTLIER_CLUSTER, MAX_KEYS,
)
from audio_engine import AudioEngine

_REMOVE_KEY = -1        # drop the cluster from the kit
_UNASSIGNED_KEY = -2    # park the cluster: no key, not exported


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
        self.setWindowTitle("Recategorize Clusters")
        self.resize(560, 720)
        self._instrument = instrument
        self._engine = engine
        self._get_gain = get_gain
        self._key_label = key_label
        # cid -> samples; cid -> current target key; cid -> its row combo.
        self._segs: dict[int, list[Segment]] = {}
        self._assign: dict[int, int] = {}
        self._combo: dict[int, QComboBox] = {}
        self._loading = False
        # Populated on Apply.
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
        header = QLabel(f"{len(loaded)} cluster{'s' if len(loaded) != 1 else ''} "
                        f"on keys · {len(unassigned)} unassigned")
        header.setObjectName("dialogHeader")
        root.addWidget(header)

        sub = QLabel(
            "Each row is a whole cluster. Send it to another key with its picker — "
            "moving onto an OCCUPIED key SWAPS the two clusters. Park a cluster in "
            "Unassigned (no key, not exported) or ✕ Remove it. Click ▶ to hear it. "
            "Nothing changes until APPLY."
        )
        sub.setWordWrap(True)
        sub.setObjectName("dialogSubtle")
        root.addWidget(sub)

        # Build clusters: one per loaded key + one per unassigned outlier sample.
        cid = 0
        rows: list[tuple[int, int]] = []    # (cid, current_key)
        for key in loaded:
            self._segs[cid] = list(self._instrument.notes[key])
            self._assign[cid] = key
            rows.append((cid, key))
            cid += 1
        for seg in unassigned:
            self._segs[cid] = [seg]
            self._assign[cid] = _UNASSIGNED_KEY
            rows.append((cid, _UNASSIGNED_KEY))
            cid += 1

        # Key-picker options (shared): every key across octaves, then park/remove.
        self._key_opts: list[tuple[int, str]] = [
            (k, self._key_option_label(k)) for k in range(MAX_KEYS)]
        self._key_opts.append((_UNASSIGNED_KEY, "— Unassigned —"))
        self._key_opts.append((_REMOVE_KEY, "✕ Remove"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(6)
        for c, _key in rows:
            col.addWidget(self._cluster_row(c))
        col.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QPushButton("APPLY")
        apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        root.addLayout(btn_row)

    def _cluster_row(self, cid: int) -> QWidget:
        segs = self._segs[cid]
        w = QFrame()
        w.setObjectName("clusterGroup")
        h = QHBoxLayout(w)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(8)

        play = QPushButton("▶")
        play.setObjectName("playBtn")
        play.setFixedWidth(34)
        play.clicked.connect(lambda _=False, c=cid: self._audition(c))
        h.addWidget(play)

        leaf = segs[0].label.upper() if getattr(segs[0], "label", "") else ""
        n = len(segs)
        lbl = QLabel(f"{leaf or 'CLUSTER'}   ·  {n} sample{'s' if n != 1 else ''}")
        lbl.setObjectName("clusterTitle")
        h.addWidget(lbl, 1)

        arrow = QLabel("→"); arrow.setObjectName("dialogSubtle")
        h.addWidget(arrow)

        combo = QComboBox()
        for k, disp in self._key_opts:
            combo.addItem(disp, k)
        sel = combo.findData(self._assign[cid])
        combo.setCurrentIndex(sel if sel >= 0 else 0)
        combo.setMinimumWidth(170)
        combo.currentIndexChanged.connect(
            lambda _i, c=cid: self._reassign(c, self._combo[c].currentData()))
        self._combo[cid] = combo
        h.addWidget(combo)
        return w

    # ---------- helpers ----------

    def _key_option_label(self, key: int) -> str:
        note = NOTE_NAMES_12[key % 12]
        octave = key // 12
        suffix = f"+{octave}" if octave else ""
        cap = KEY_CAP_LABEL.get(key, "")
        return f"{note}{suffix}   {cap}".rstrip()

    def _reassign(self, cid: int, new_key: int) -> None:
        """Move cluster `cid` to `new_key`. If a real key is already taken by
        another cluster, swap them (that cluster takes cid's old slot)."""
        if self._loading:
            return
        old = self._assign[cid]
        if new_key == old:
            return
        if new_key >= 0:
            occupant = next((c for c, k in self._assign.items()
                             if k == new_key and c != cid), None)
            if occupant is not None:
                self._assign[occupant] = old
                self._set_combo(occupant, old)
        self._assign[cid] = new_key

    def _set_combo(self, cid: int, key: int) -> None:
        combo = self._combo[cid]
        self._loading = True
        try:
            i = combo.findData(key)
            if i >= 0:
                combo.setCurrentIndex(i)
        finally:
            self._loading = False

    def _audition(self, cid: int) -> None:
        segs = self._segs[cid]
        if segs:
            self._engine.play(segs[len(segs) // 2].audio,
                              gain=self._get_gain(), note_id=-1)

    def _on_apply(self) -> None:
        new_notes: dict[int, list[Segment]] = {}
        new_unassigned: list[Segment] = []
        for cid, segs in self._segs.items():
            key = self._assign[cid]
            if key == _REMOVE_KEY:
                continue
            if key == _UNASSIGNED_KEY:
                for s in segs:
                    s.cluster = OUTLIER_CLUSTER
                new_unassigned.extend(segs)
            else:
                for s in segs:
                    s.cluster = key
                new_notes[int(key)] = sorted(segs, key=lambda s: getattr(s, "rms", 0.0))
        new_unassigned.sort(key=lambda s: getattr(s, "rms", 0.0))
        self.new_notes = new_notes
        self.new_unassigned = new_unassigned
        self.accept()
