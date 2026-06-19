# Samplebot-3000

Turn any audio file into a playable drum-machine instrument. Point it at a
recording — a drum loop, a break, a field recording — and it detects every
transient, slices a sample from each, fingerprints it by timbre, and lays the
samples out on a keyboard. Two ways to organize them:

- **Clustering** (unsupervised) — groups acoustically similar slices and maps
  each group to a key. Three algorithms (KMeans, Agglomerative, HDBSCAN), and an
  optional **CLAP** mode that clusters in a pretrained audio-embedding space and
  lays the clusters onto a real drum-kit layout by type.
- **Classification** (supervised) — a trained model labels each slice as a drum
  type (kick, snare, hat, tom, clap, fx, …) and places it on a fixed
  General-MIDI-style drum layout. Seven choices, including a **CLAP+Pitch**
  embedding model and an **Ensemble** vote across all of them.

Pick a wav/mp3/flac/aiff, choose Clustering or Classification, hit **Run**, and
play it with your computer keyboard. Each key holds a set of samples you tab
through with arrows, and every sample carries its own envelope, velocity,
loop, and filter settings. A 16-step sequencer grid is wired for a coming
step-sequencer mode.

![Reference UI](reference.png)

> The console script and Python package are still named `audio2inst` for
> backward compatibility; `samplebot-3000` is the alias.

## Install

Requires Python 3.10+. On macOS you also need PortAudio:

```bash
brew install portaudio
```

Then from the project directory:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

The app expects its dependencies (PySide6, librosa, scikit-learn, sounddevice,
scipy) in a `.venv` in the project directory.

### Optional: the CLAP models

The **CLAP+Pitch** classifier and **CLAP** clustering mode use a pretrained
audio-embedding model, which pulls in `torch` + `transformers`. These are kept
out of the core install so the clustering/hand-crafted-classifier app stays
lightweight:

```bash
pip install -e ".[clap]"
```

On first use the CLAP checkpoint (`laion/clap-htsat-unfused`, ~600 MB) downloads
once and is cached. Without these deps the app still runs — you just won't see
the CLAP options.

## Run

```bash
python main.py
```

`main.py` **re-execs itself under the project `.venv`** if launched from another
interpreter (e.g. a conda `base` env), so it always runs against the right audio
stack and library versions. If no audio device can be opened the GUI still
launches — it just runs silent so you can analyze without sound.

## Controls

### Method toggle

A segmented switch picks **CLUSTERING** or **CLASSIFICATION** — never both. The
data-cleaning pipeline (transient detection, gate, trim, normalize) is identical
for both; only the grouping step differs.

- **CLUSTERING** → a **MODE** dropdown (KMeans / Agglomerative / HDBSCAN) and its
  parameter, plus a **CLAP** selector (see below). Each cluster becomes a key.
- **CLASSIFICATION** → a **MODEL** dropdown. Each slice is labeled and placed on
  the fixed drum layout below.

### CLAP mode (clustering) — Off / Soft / Hard

The **CLAP** selector controls whether clustering uses the pretrained embedding
and how the clusters land on the keyboard:

- **Off** — cluster on the 57-dim hand-crafted features; one cluster per key,
  pitch-ordered (the original clustering behavior).
- **Soft** — cluster in CLAP-embedding space, then type each cluster with the
  CLAP classifier and drop it on the matching drum-kit key (the most kick-like
  cluster → KICK, snare-like → the SNR keys, …). When a type has more clusters
  than it has keys, the extras share that type's keys.
- **Hard** — same, but **one cluster per key, no doubling**. Clusters claim keys
  in confidence order; when a type's keys are full the overflow spills to the
  nearest empty key instead of stacking.

CLAP clustering gives tighter, more coherent groups than the hand-crafted
features (it leans on a model trained on huge audio data), and unlike the
per-slice classifier it types each cluster *as a whole* by consensus, so a few
odd slices can't flip a group's label.

### Classification models

The **MODEL** dropdown, in rough order of strength:

- **Ensemble (vote)** — every trained model classifies each slice; the
  most-voted label wins. (Note: most members share the 57-dim features, so they
  can outvote the lone CLAP model — see *Notes*.)
- **CLAP+Pitch (GB)** — the recommended default. A pretrained CLAP audio
  embedding fused with pitch/temporal scalars, fed to a Gradient-Boosting head.
  Best cross-kit accuracy.
- **SVM / Random Forest / k-NN / Gradient Boost / Neural Net** — the original
  five models over the 57-dim hand-crafted feature vector.

### Analysis (left column)

- **Sample Len** (0.1–5.0 s) — the maximum sample length captured from each
  transient. Default 0.5 s suits drums; raise it for sustained material. (For
  classification, the *feature* window is fixed independently of this — see
  *Data pipeline* — so moving the slider never changes a model's predictions.)
- **Transient Sens** — onset-detector sensitivity. Higher catches more (subtler)
  transients; lower keeps only strong hits.
- **Noise Gate** — a transient whose peak never rises above this level is
  dropped entirely.
- **Trim Threshold** — where a sample's tail ends: the sample runs from the
  transient until its envelope decays below this level.
- **Clip at Next** — when on, a sample also ends as soon as the next transient is
  detected; when off it runs to the trim-threshold decay or the sample-length
  cap, whichever comes first.
- **Run Analysis** / **View Clusters** — the pipeline runs on a worker thread;
  the cluster viewer opens after a run to audition every slice.
- **Sequencer · 16 steps** — a clickable grid. Select a key, then light up the
  steps where it should trigger. Patterns are stored per key. (Playback clock is
  not wired yet — this is state for the coming step-sequencer mode.)

### Playback — per sample (right column)

Pressing a key **selects** it (amber outline + the ACTIVE light shows its name).
The **◀ N / M ▶** selector tabs through that key's samples; the chosen sample is
what plays. **Every playback control below applies to the currently selected
sample** and is remembered per sample:

- **Velocity** (1–127) — output volume plus a slight velocity-dependent lowpass
  (softer hits are a touch darker).
- **Vel Layers** — when on, the velocity knob *selects* which of a key's stacked
  samples fires (ordered by loudness), turning a key's stack into velocity
  layers. Works in both clustering and classification modes. When off, the
  ◀ N/M ▶ arrows pick the sample and velocity is just volume.
- **Loop** — loops the sample while the key is held.
- **Envelope (ADSR)** — per-voice attack / decay / sustain / release.
- **Filter** — per-voice lowpass cutoff (20 Hz – 20 kHz, log).
- **Pitch Sort** / **Octave** — clustering-mode global controls (ordering and
  multi-octave reach); not used by the fixed classification layout.

### Keyboard

Computer keys `A W S E D F T G Y H U J K O L P ;` map to the visible octave.

In **classification** mode (and CLAP soft/hard clustering) the layout is a fixed
drum kit:

```
key:  C   C#   D   D#   E    F   F#   G   G#   A   A#   B   C   C#   D    D#   E
       Kick Snr1 Snr2 Clap Snr3 LoT HH1 MidT HH2 HiT HH3 FX1 FX2 Crash FX3 Ride FX4
```

Drum types with several keys (snare, hats, fx) spread their samples across those
keys.

## Data pipeline

```
raw file
  → detect transients above the noise gate
  → for each: grab from the transient start until the signal falls below the
    trim threshold, OR hits the sample-length cap, OR the next transient
    (only if "Clip at Next" is on)
  → normalize each sample (peak)
  → cluster (KMeans/Agglomerative/HDBSCAN, optionally in CLAP space)
    OR classify (trained model / ensemble)
```

There is exactly **one** "audio slice → features" implementation
(`pipeline.segment_features` / `features_from_oneshot`), shared by training and
serving, so a model is trained on the same representation it sees live. A
regression test (`tests/test_train_serve_skew.py`) locks this in.

For classification, the feature vector is computed over a **fixed window**
(`CLASSIFY_FEATURE_LEN_S`, 0.5 s) that is decoupled from the user's *Sample Len*
slider — the slider controls how much audio is captured for playback, but a
trained model always sees the same window, so the knob can't drag a model
out-of-distribution.

### Feature representations

- **Hand-crafted (57 dims)** — used by clustering and the five classic models:
  13 MFCC means + 13 MFCC stds + 13 MFCC-delta means + 12 chroma + spectral
  centroid (mean/std), rolloff, bandwidth, flatness, and zero-crossing rate.
- **CLAP+Pitch (517 dims)** — used by the CLAP classifier and CLAP clustering: a
  512-dim LAION-CLAP audio embedding (L2-normalized) fused with 5 pitch/temporal
  scalars (low-band fundamental `f0`, `f0_midi`, log-attack-time, temporal
  centroid, decay slope). The pitch scalars target the hi/mid/lo-tom distinction
  that timbre embeddings alone don't resolve.

Features are computed on a peak-normalized copy so loudness doesn't influence
them.

## Supervised classifier

`classifier.py` trains/evaluates the drum-type models from the sorted
`training data/` bins (kept out of the repo). Models are saved to
`models/*.joblib` and selected live in the GUI.

```bash
python classifier.py --extract            # build feature_cache.npz from the bins
python classifier.py --compare            # 5-fold CV accuracy for all 57-dim models
python classifier.py --train --model gb   # train + save one 57-dim model
python classifier.py --train --model clap # train + save the CLAP+Pitch model
python classifier.py --predict file.wav --model gb
```

The **Ensemble** model is assembled on the fly from whatever base models are
trained — it has no file of its own.

## Evaluation harnesses

Honest evaluation is grouped by source drum machine (**leave-one-machine-out**,
LOMO), so near-identical siblings from the same kit can't leak across the
train/test split and inflate the score. Macro-F1 under LOMO is the primary
metric throughout.

```bash
python model_analysis.py            # LOMO for the 57-dim models; random-vs-grouped gap
python embed_eval.py --backend clap # LOMO for CLAP embeddings (emb / scalars / fused)
python fx_reject.py --method calibrated   # FX-as-open-set-rejection experiment
python acoustic_eval.py transfer    # synth→acoustic generalization on IDMT-SMT-Drums
```

Supporting data tools (operate on `training data/`, not needed to run the app):
`sort_samples.py` (name-based bin sorter), `normalize_names.py` (rename/repack
bins), `datasets_loader.py` (ingest external datasets into LOMO-splittable rows).

## Architecture

```
main.py                    (re-exec into .venv, then launch)
  └─ gui.py
       ├─ pipeline.py        (worker QThread — extract, cluster/classify, build)
       ├─ embedders.py       (swappable pretrained audio embedders; CLAP backend)
       ├─ audio_engine.py    (OutputStream; per-voice ADSR + per-voice lowpass)
       ├─ piano_widget.py    (paints the octave; selection + key labels)
       ├─ moog_widgets.py    (Knob, ToggleSwitch, LED, StepGrid)
       ├─ cluster_viewer.py  (dialog: audition every slice in every key)
       ├─ adsr_widget.py     (envelope curve visualizer)
       └─ classifier.py      (load/train models; assemble the ensemble)
```

- **pipeline.py** — `run_pipeline(path, mode, …)` → `Instrument`. Transient
  detect (librosa) → gate/trim/normalize → features → cluster or classify → key
  assignment. Hosts the shared feature path, CLAP clustering, consensus typing,
  and the drum-kit layouts.
- **embedders.py** — an `Embedder` interface (`embed(audio, sr) -> vector`) with
  a CLAP backend (and room for PaSST/OpenL3/BEATs). Heavy deps import lazily.
- **audio_engine.py** — one OutputStream. Each voice captures its own ADSR and
  lowpass cutoff at play time and is filtered independently, so per-sample
  settings don't bleed across overlapping voices.
- **gui.py** — per-sample `PlaybackSettings` are stored on each slice; selecting
  a sample loads its settings into the controls, editing a control writes back,
  and playing applies them.

## Roadmap / Future work

- **Step-sequencer playback.** The 16-step grid already stores a pattern per
  key; next is a transport/clock that triggers each key on its lit steps, with
  tempo and swing.
- **Manual cluster editing.** Drag samples between keys/clusters, split or merge
  clusters, and re-assign a mislabeled slice by hand — turning the auto layout
  into a starting point you can correct.
- **Reinforcement from your corrections.** Treat every manual re-categorization
  as a labeled example and **recalibrate the models on your own taste** — an
  active-learning loop where moving a sample to the "right" key feeds back into
  training, so the classifier adapts to how *you* hear drums.
- **A real Moog-style retro-future GUI.** Lean fully into the skeuomorphic
  analog look — warm panel textures, real rotary knobs with inertia, glowing
  VU/LED metering, patch-cable routing — so it feels like a piece of hardware,
  not a settings dialog. The `moog_widgets` are the seed of this.
- **Save / load custom drum kits + Logic Pro export.** Serialize a built
  instrument (per key: ordered slices + mappings + per-sample settings) to a
  project format you can reopen, and **export to Logic Pro** — native Sampler
  (EXS24) / Drum Machine Designer kits so it plays in a DAW without the Samplebot
  window. SFZ for everything else.
- **Ingest Logic Pro's stock drum one-shots as training data.** Logic ships a
  large, cleanly-labeled library of kit pieces — a high-quality, consistent
  source to broaden the classifier (especially acoustic kits) beyond the
  drum-machine bins.
- **Per-note pitch control.** A pitch knob beside the filter that transposes each
  note independently (resample / phase-vocoder), so a single slice can be tuned
  across keys — useful for melodic toms, tuned percussion, and bass hits.
- **More embedding backends.** PaSST / BEATs / OpenL3 behind the same `Embedder`
  interface, chosen by LOMO macro-F1; a lighter backend (EfficientAT) if live
  latency matters.
- **Better toms & FX.** A pitch-aware tom path and a cleaner FX treatment
  (open-set rejection or its own coherent sub-classes) — the two classes the
  honest evaluation still flags as weak.

## Notes

- **CLAP+Pitch is the strongest classifier** in honest leave-one-machine-out
  evaluation; it's the default. The **Ensemble** votes across every trained
  model, but most members share the 57-dim features and so are correlated — they
  can outvote the lone CLAP member, so the ensemble doesn't automatically beat
  CLAP alone. Treat it as experimental until measured on your material.
- **CLAP already generalizes synth→acoustic well** — a model trained only on
  drum-machine samples classifies unseen acoustic kits at ~0.97 macro-F1 on
  kick/snare/hat — so more clean kick/snare/hat data isn't the bottleneck; toms
  (pitch) and FX (taxonomy) are.
- **Velocity no longer selects the sample by default.** Sample choice is the
  arrow-tab selector; turn on **Vel Layers** to make velocity pick the sample.
- **Per-sample settings live on the slice**, so they survive pitch-sort
  re-ordering in clustering mode.
- **Loops are naive** (no crossfade at the seam) — capture a longer slice or use
  a long release to hide the click.
- **Memory scales with sample length** — a 5 s slice at 44.1 kHz mono is ~882 KB.
```
