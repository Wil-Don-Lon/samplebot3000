"""Save and load named sample/drum kits inside Samplebot.

A kit is a folder under `kits/<KitName>/` holding a `kit.json` manifest plus one
WAV per sample (float32, lossless). Saving snapshots the current `Instrument`
(key -> stacked samples), each sample's playback settings, labels, and whether
the kit uses the fixed drum-machine layout. Loading rebuilds an `Instrument` you
can audition, recategorize, and export to Logic.

This module has no GUI/Qt dependency. Playback settings are reconstructed via a
`playback_factory` the caller passes in (the GUI's PlaybackSettings), so kit
storage never has to import the GUI.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from pipeline import Instrument, Segment, TARGET_SR, OUTLIER_CLUSTER
from logic_export import sanitize_name

KITS_DIR = Path(__file__).resolve().parent / "kits"

# Playback fields persisted per sample (widget-domain ints, see PlaybackSettings).
_PLAYBACK_FIELDS = ("velocity", "loop", "a", "d", "s", "r", "lpf", "volume")


def _playback_to_dict(pb) -> dict:
    if pb is None:
        return {}
    return {f: getattr(pb, f) for f in _PLAYBACK_FIELDS if hasattr(pb, f)}


def kit_dir_for(name: str, root: Optional[Path] = None) -> Path:
    root = Path(root) if root is not None else KITS_DIR
    return root / sanitize_name(name)


def save_kit(
    name: str,
    instrument: Instrument,
    is_classify: bool,
    sample_choice: Optional[dict[int, int]] = None,
    key_label: Optional[Callable[[int], str]] = None,
    root: Optional[Path] = None,
    overwrite: bool = True,
) -> Path:
    """Write `<root>/<name>/` with a manifest + WAVs. Returns the kit folder."""
    safe = sanitize_name(name)
    if not safe:
        raise ValueError("kit name is empty")
    kit_dir = kit_dir_for(safe, root)
    if kit_dir.exists():
        if not overwrite:
            raise FileExistsError(f"kit '{safe}' already exists")
        shutil.rmtree(kit_dir)
    samples_dir = kit_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    key_label = key_label or (lambda k: f"key{k}")
    sr = int(getattr(instrument, "sample_rate", TARGET_SR))
    manifest = {
        "name": safe,
        "sample_rate": sr,
        "is_classify": bool(is_classify),
        "created": time.time(),
        "keys": {},
        "unassigned": [],
        "sample_choice": {str(k): int(v) for k, v in (sample_choice or {}).items()},
    }

    def _entry(seg, fname: str) -> dict:
        audio = np.asarray(seg.audio, dtype=np.float32).ravel()
        sf.write(str(kit_dir / fname), audio, sr, subtype="FLOAT")
        return {
            "file": fname,
            "label": getattr(seg, "label", "") or "",
            "label_conf": float(getattr(seg, "label_conf", 0.0) or 0.0),
            "onset_time": float(getattr(seg, "onset_time", 0.0) or 0.0),
            "dominant_freq": float(getattr(seg, "dominant_freq", 0.0) or 0.0),
            "rms": float(getattr(seg, "rms", 0.0) or 0.0),
            "playback": _playback_to_dict(getattr(seg, "playback", None)),
        }

    def _usable(segs):
        return [s for s in segs if getattr(s, "audio", None) is not None
                and np.asarray(s.audio).size > 0]

    for key in sorted(instrument.notes):
        segs = _usable(instrument.notes[key])
        if not segs:
            continue
        label = sanitize_name(key_label(key), f"key{key}")
        manifest["keys"][str(key)] = [
            _entry(seg, f"samples/{key:02d}_{i:02d}_{label}.wav")
            for i, seg in enumerate(segs)
        ]

    # Unassigned outliers: persisted (so manual triage survives save/load) but
    # keyless — they never get a keyboard key and are never exported to Logic.
    for i, seg in enumerate(_usable(getattr(instrument, "unassigned", []))):
        manifest["unassigned"].append(
            _entry(seg, f"samples/unassigned_{i:03d}.wav"))

    (kit_dir / "kit.json").write_text(json.dumps(manifest, indent=2))
    return kit_dir


def load_kit(
    kit_dir: Path,
    playback_factory: Optional[Callable[..., object]] = None,
) -> tuple[Instrument, bool]:
    """Rebuild (Instrument, is_classify) from a kit folder.

    `playback_factory(**fields)` builds each sample's playback settings object
    (the GUI passes its PlaybackSettings). If None, playback stays unset.
    """
    kit_dir = Path(kit_dir)
    manifest = json.loads((kit_dir / "kit.json").read_text())
    sr = int(manifest.get("sample_rate", TARGET_SR))
    inst = Instrument(sample_rate=sr, notes={})

    def _seg(entry: dict, cluster: int) -> Segment:
        audio, _ = sf.read(str(kit_dir / entry["file"]), dtype="float32")
        audio = np.asarray(audio, dtype=np.float32).ravel()
        pb = None
        if playback_factory is not None and entry.get("playback"):
            pb = playback_factory(**entry["playback"])
        return Segment(
            audio=audio,
            onset_time=float(entry.get("onset_time", 0.0)),
            features=np.zeros(0, dtype=np.float32),
            rms=float(entry.get("rms", float(np.sqrt(np.mean(audio ** 2))
                      if audio.size else 0.0))),
            cluster=cluster,
            dominant_freq=float(entry.get("dominant_freq", 0.0)),
            label=entry.get("label", ""),
            label_conf=float(entry.get("label_conf", 0.0)),
            playback=pb,
        )

    for key_str, entries in manifest.get("keys", {}).items():
        key = int(key_str)
        segs = [_seg(e, key) for e in entries]
        if segs:
            inst.notes[key] = segs

    # Keyless outliers, restored to the unassigned bucket (cluster == -1).
    inst.unassigned = [_seg(e, OUTLIER_CLUSTER)
                       for e in manifest.get("unassigned", [])]

    return inst, bool(manifest.get("is_classify", False))


def load_sample_choice(kit_dir: Path) -> dict[int, int]:
    manifest = json.loads((Path(kit_dir) / "kit.json").read_text())
    return {int(k): int(v) for k, v in manifest.get("sample_choice", {}).items()}


def list_kits(root: Optional[Path] = None) -> list[dict]:
    """List saved kits, newest first: name, path, n_keys, n_samples, modified."""
    root = Path(root) if root is not None else KITS_DIR
    if not root.exists():
        return []
    kits = []
    for child in root.iterdir():
        manifest_path = child / "kit.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (ValueError, OSError):
            continue
        keys = manifest.get("keys", {})
        n_samples = sum(len(v) for v in keys.values())
        kits.append({
            "name": manifest.get("name", child.name),
            "path": child,
            "n_keys": len(keys),
            "n_samples": n_samples,
            "is_classify": bool(manifest.get("is_classify", False)),
            "modified": manifest_path.stat().st_mtime,
        })
    kits.sort(key=lambda k: k["modified"], reverse=True)
    return kits


def delete_kit(kit_dir: Path) -> None:
    kit_dir = Path(kit_dir)
    if (kit_dir / "kit.json").is_file():
        shutil.rmtree(kit_dir)
