# Samplebot-3000

Turn any audio file into a playable sampler instrument via unsupervised clustering. Point it at a recording — a drum loop, a field recording, a vocal take, an engine — and it slices the audio at every onset, fingerprints each slice by timbre, and groups the slices into clusters. Each cluster is mapped to a key, so acoustically similar sounds end up under the same note. Pick a wav/mp3/flac, hit Run, and play it with your computer keyboard. Includes an ADSR envelope, master lowpass filter, octave switching, velocity-layer crossfade, and a choice of three clustering algorithms.

> The console script and Python package are still named `audio2inst` for backward compatibility; `samplebot-3000` is provided as an alias.

## Install

Requires Python 3.10+. On macOS you also need PortAudio:

```bash
brew install portaudio
```

Then from the project directory:

```bash
uv venv && uv pip install -e .
# or:
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

## Run

```bash
samplebot-3000
# or the legacy alias:
audio2inst
# or directly:
python main.py
```

## Controls

### Analysis (left column)

- **Mode** — three clustering algorithms:
  - **Manual k (KMeans)** — you pick the cluster count (2–52). Always produces exactly k clusters; every segment belongs to one.
  - **Auto threshold (Agglomerative)** — Ward-linkage hierarchical clustering. Threshold slider controls how close two clusters must be to merge. Lower threshold → tighter, more numerous clusters.
  - **Auto density (HDBSCAN)** — density-based discovery. `min size` sets the smallest grouping considered a real cluster. Noise points get reassigned to their nearest cluster centroid so nothing is dropped.
- **Sample length** (0.1–5.0 s) — how much audio to capture starting at each onset. Default 0.5s gives drum-machine behavior; raise to 2–3s for sustained sources (vocal notes, drones, engine recordings). Changing this requires a fresh Run to apply.
- **Sensitivity** — onset detector threshold. Higher detects more transients (and more false positives). Only takes effect on the next Run.
- **Run Analysis** / **View Clusters** — pipeline runs on a worker thread; cluster viewer opens after a successful run and lets you audition every segment.

### Playback (right column)

- **Velocity** (1–127) — picks which segment plays within a cluster (low → quietest, high → loudest) **and** controls output gain. With velocity-layer crossfade, intermediate values play two adjacent segments with equal-power cos/sin weights so perceived loudness stays smooth across the velocity range.
- **Loop while held** — when checked, voices loop their audio buffer back to the start as long as the envelope hasn't entered the release phase. Combined with a long Release in ADSR, this turns short slices into pad-like sustained notes. Affects voices created after toggling; in-flight notes keep their original setting.
- **Envelope (ADSR)** — applied per-voice:
  - **A** (0–1000 ms) — fade-in from silence
  - **D** (0–1000 ms) — drop from peak to sustain level
  - **S** (0–1.00) — held level during the body of the note
  - **R** (0–2000 ms) — fade-out triggered when the key is released
- **Filter** — 4th-order Butterworth lowpass on the master mix. Slider is logarithmic, 20 Hz to 20 kHz. At max, the filter is bypassed.
- **Octave** — `Z` / `X` keys or the arrow buttons. Each octave is 13 keys; if clustering produces more than 13 clusters, the extras are reachable in higher octaves. The indicator shows current / max octave.

### Keyboard

```
  W   E       T   Y   U
A   S   D   F   G   H   J   K
C   D   E   F   G   A   B   C
```

`Z` octave down, `X` octave up. Hold multiple keys for polyphony. Releasing a key triggers the release phase of the envelope.

## How clustering works

Each onset gets a 0.5-second slice. That slice gets reduced to a 29-number fingerprint of its timbre — 13 MFCC means + 13 MFCC stds + spectral centroid (mean + std) + spectral rolloff mean. The features are z-score normalized so no single dimension dominates the geometry, then run through the chosen clustering algorithm. Finally, clusters are sorted by median spectral centroid (dark → bright) and assigned to keys in that order.

MFCC clustering groups by **acoustic similarity**, not by source object. Two different objects that produce dull thuds will land in the same cluster.

## Architecture

```
main.py
  └─ gui.py
       ├─ pipeline.py        (worker QThread — pure data transform)
       ├─ audio_engine.py    (OutputStream, ADSR voices, lowpass filter)
       ├─ piano_widget.py    (paints one octave, signals on press/release)
       ├─ cluster_viewer.py  (dialog: audition every segment in every key)
       └─ adsr_widget.py     (envelope curve visualizer)
```

- **pipeline.py** — `run_pipeline(path, mode, n_clusters | threshold | min_cluster_size, sensitivity)` → `Instrument`. Onset detect via librosa, MFCC + spectral features (29 dims), z-score normalize, cluster, sort by spectral centroid.
- **audio_engine.py** — one OutputStream. Voices carry their own ADSR state machine (Attack → Decay → Sustain → Release → Done) and a `note_id`; `release_note(note_id)` transitions matching voices to release. Master lowpass is stateful across blocks.
- **piano_widget.py** — paints exactly one octave; gui.py translates visible key → cluster index via the current octave.
- **gui.py** — integrates everything. Velocity layering uses equal-power crossfade. Octave switching releases pressed keys to prevent stuck notes.

## Roadmap / Future work

The current build is a self-contained playable instrument. The next major direction is **DAW integration** — letting Samplebot-3000 export the instrument it builds so it can be used natively inside a host, instead of only from the app's own keyboard.

- **Logic Pro / Sampler (EXS24) export.** Write the clustered slices to disk as individual `.wav` files plus a `.exs` (or modern Sampler) instrument that maps each cluster to the correct key and velocity zone. The user could then load the result as a native Logic instrument, play it from a MIDI controller, and record it into a project — no Samplebot window required.
- **General MIDI-instrument / SFZ export.** SFZ is an open, text-based sampler format supported across many DAWs and plugins (sforzando, Kontakt-adjacent tools, Decent Sampler, etc.). Emitting an `.sfz` alongside the rendered slices would make the instrument portable beyond a single host. This is likely the first export target because it is plain-text and trivial to generate from the existing key/velocity mapping.
- **Ableton / generic drum-rack export.** Map clusters onto drum-rack pads for groove-box-style workflows.
- **MIDI region export.** Optionally emit a MIDI clip that triggers the clusters in the order/timing of the original onsets, so the reconstructed performance can be dropped straight onto a track.
- **Plugin wrapper (AU/VST).** Longer term, ship the engine as an instrument plugin so analysis and playback happen inside the DAW.

The internal data model already produces exactly what these formats need — for each cluster: an ordered set of audio slices, a target key, and velocity layering — so export is primarily a serialization layer on top of `pipeline.Instrument`.

## V2 design choices

- **Velocity crossfade is mix-time, not time-domain.** When velocity lands between two layers, both segments are played simultaneously with weights `cos(frac · π/2)` and `sin(frac · π/2)`. This keeps perceived loudness constant across the velocity range. If you wanted a time-domain crossfade (sample A fades out as sample B fades in over the duration of the note), that's a different feature — let me know.
- **HDBSCAN noise points are reassigned, not dropped.** Each noise-labeled segment is attached to the nearest cluster centroid in feature space.
- **ADSR for one-shot samples is unusual.** Defaults are configured to be essentially transparent (A=0, D=0, S=1.0, R=50 ms): the sample plays normally, with a short fade-out on key release. Crank `A` for swells; crank `R` for long tails on sustained sources.
- **Lowpass state resets when cutoff changes.** Rapid slider drags may produce subtle clicks. Acceptable for live exploration.
- **Loops are naive.** When loop is on, the playhead wraps back to sample 0 with no crossfade at the seam. If your slice's first and last samples differ a lot, you'll hear a click at the loop point. Two ways to hide it: capture a longer slice (so loops happen less often), or use a long Release so the envelope is dropping while looping.
- **Memory scales with sample length.** A 5-second slice at 44.1kHz mono is 882KB per segment; with hundreds of segments across many clusters this can add up. Drop the sample length back down if you don't need long captures.
