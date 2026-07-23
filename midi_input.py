"""USB MIDI input → Qt signals.

Thin and optional: if python-rtmidi (or a device) isn't present, everything
degrades to a no-op so the app still runs. The rtmidi callback fires on its own
thread; emitting a Qt signal from there is delivered to the main thread via a
queued connection, so handlers run on the UI/audio thread safely.

Kept frontend-agnostic (no GUI/pipeline imports) so a future headless/Pi build
can reuse it as-is.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, Signal

try:
    import rtmidi
    _HAVE_RTMIDI = True
except Exception:  # noqa: BLE001
    _HAVE_RTMIDI = False


class MidiListener(QObject):
    """Opens a MIDI input port and emits note on/off. Poll-based hot-plug is
    handled by the caller via connect_first()/still_present()/disconnect()."""

    note_on = Signal(int, int)     # (midi_note 0..127, velocity 1..127)
    note_off = Signal(int)         # (midi_note 0..127)

    def __init__(self) -> None:
        super().__init__()
        self._midi = None
        self.port_name: Optional[str] = None

    @staticmethod
    def available() -> bool:
        return _HAVE_RTMIDI

    def ports(self) -> list[str]:
        if not _HAVE_RTMIDI:
            return []
        m = rtmidi.MidiIn()
        try:
            return list(m.get_ports())
        finally:
            m.delete()

    def is_connected(self) -> bool:
        return self._midi is not None

    def connect_first(self) -> Optional[str]:
        """Open the first available input port. No-op if already open or none."""
        if not _HAVE_RTMIDI or self._midi is not None:
            return self.port_name
        m = rtmidi.MidiIn()
        names = m.get_ports()
        if not names:
            m.delete()
            return None
        m.open_port(0)
        # Drop clock/sysex/active-sensing so the callback only sees real notes.
        m.ignore_types(sysex=True, timing=True, active_sense=True)
        m.set_callback(self._on_message)
        self._midi = m
        self.port_name = names[0]
        return self.port_name

    def still_present(self) -> bool:
        """Whether our opened port still exists (detects unplug)."""
        return self.port_name is not None and self.port_name in self.ports()

    def disconnect(self) -> None:
        if self._midi is not None:
            try:
                self._midi.cancel_callback()
                self._midi.close_port()
                self._midi.delete()
            except Exception:  # noqa: BLE001
                pass
            self._midi = None
            self.port_name = None

    # rtmidi callback (runs on the MIDI thread)
    def _on_message(self, event, data=None) -> None:
        msg, _delta = event
        if len(msg) < 3:
            return
        status = msg[0] & 0xF0
        note, vel = int(msg[1]), int(msg[2])
        if status == 0x90 and vel > 0:
            self.note_on.emit(note, vel)
        elif status == 0x80 or (status == 0x90 and vel == 0):
            self.note_off.emit(note)
