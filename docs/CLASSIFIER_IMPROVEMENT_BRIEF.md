# Samplebot-3000 — Classifier Improvement Brief

> **Historical — archived 2026-10-01.** This brief drove the supervised-classifier
> work (§1, train/serve skew, is done and locked by `tests/test_train_serve_skew.py`).
> The app has since moved to a cluster-first design: HDBSCAN groups the slices and
> CLAP labels the finished clusters, so the five-model supervised classifier it
> describes as "current" is no longer in the live path. Kept for the reasoning and
> the evaluation methodology, which still hold.

**For:** the coding agent working in this repo (`audio2inst` / `samplebot-3000`)
**Goal:** raise supervised drum-type classification accuracy and cross-kit generalization, fix the tom and FX problems, and eliminate train/serve skew.

This brief is opinionated about *ordering*. Do the steps in sequence; several early steps are cheap and likely account for most of the current accuracy gap. Validate after each step before moving on.

---

## 0. Context you need

- Current classifier: a 57-dim hand-crafted feature vector (13 MFCC mean + 13 MFCC std + 13 MFCC-delta mean + 12 chroma + spectral centroid mean/std + rolloff + bandwidth + flatness + ZCR) fed to one of five sklearn models (SVM / RandomForest / kNN / GradientBoost / MLP). See `classifier.py`, `pipeline.py`.
- Live serving path (`pipeline.py` → `run_pipeline`): transient detect (librosa) → noise gate → trim at envelope threshold → optional clip-at-next-onset → peak-normalize → feature extraction → classify.
- Known issues (confirmed): (a) **train/serve skew** — training features are likely extracted from clean full-length bin samples, not from slices processed by the live path; (b) **toms** collapse because MFCCs discard pitch; (c) the **FX** class is an incoherent catch-all.
- Evaluation harness exists: `model_analysis.py` does leave-one-machine-out (LOMO). Treat LOMO macro-F1 as the primary metric throughout — NOT random-split accuracy.

---

## 1. Eliminate train/serve skew (do this first — it's free)

The model must be trained on features extracted by the **exact same** code path that runs at inference.

- Refactor feature extraction into a single function used by BOTH training and serving. There must be exactly one implementation of "audio slice → feature vector."
- Before extracting training features, push every training sample through the live segmentation logic (gate / trim / normalize / length-cap / clip-at-next where applicable). If a training bin sample is a clean full-length one-shot, it must still be trimmed/normalized the same way a live slice would be.
- Add a regression test asserting that a known sample produces identical features through the training entry point and the serving entry point.

**Acceptance:** identical-feature test passes; re-run LOMO and record the new baseline.

---

## 2. Replace the hand-crafted vector with pretrained audio embeddings

This is the single biggest expected accuracy gain. Keep the existing sklearn classifiers as the head; only swap the features. Implement an `Embedder` interface (`embed(audio, sr) -> np.ndarray`) with swappable backends so we can A/B them.

Backends to implement, in priority order:

1. **PaSST** (`pip install hear21passt`) — default recommendation. AST-class model, strong on instrument tasks, works well on short clips.
2. **OpenL3** (`pip install openl3`, `content_type="music"`, 512-dim) — easiest; implement first as the smoke-test backend.
3. **BEATs** (Microsoft `unilm` repo / HF) — highest raw quality (768-dim, 50 Hz temporal resolution, good for transients). Heavier; implement if PaSST isn't enough.
4. **CLAP** (`laion/larger_clap_general` or `microsoft/msclap` via HF) — needed for the zero-shot path in §5. Implement alongside the others.
5. **EfficientAT** (`fschmid56/EfficientAT`, MobileNetV3) — implement only if in-app latency/model size becomes a constraint; near-transformer quality at much lower cost.

Implementation notes:
- These models expect specific sample rates (e.g. VGGish/CLAP 16 kHz, OpenL3 48 kHz). Resample inside each backend; do NOT change the app's audio engine SR.
- Drum one-shots are short (<1 s). Pad/center to the model's minimum temporal support; average frame embeddings over time to one vector per slice.
- Cache embeddings to disk (extend the existing `feature_cache.npz` pattern) — extraction is the slow part.
- First experiment: **frozen embedding + existing SVM/GB head.** This isolates whether the representation was the bottleneck. Do not fine-tune yet.

**Acceptance:** LOMO macro-F1 with frozen embeddings beats the §1 baseline. Compare PaSST vs OpenL3 vs BEATs head-to-head in the harness and keep the best.

---

## 3. Fix toms with a pitch feature; strengthen temporal features

Do this whether or not embeddings already solved toms — a low-band pitch scalar is cheap and directly targets hi/mid/lo separation. If staying partly hand-crafted, also add temporal-envelope descriptors (percussion is defined by its attack/decay, which the current vector barely captures).

- **Low-band fundamental:** estimate the dominant low-frequency spectral peak of the resonant body (peak-pick the magnitude spectrum below ~400–500 Hz, or run pYIN / subharmonic-summation restricted to the low band). Emit `f0_hz` (and optionally `f0_midi`) as a feature.
- **Temporal-envelope descriptors:** log-attack-time, temporal centroid, decay slope, envelope peak count (the last separates claps — multiple transients — from single hits), and a couple of autocorrelation features.
- If using embeddings, **concatenate** these scalars onto the embedding vector (standard-scale the scalars first). If hand-crafted only, add them to the 57-dim vector and run feature selection (RFECV with GradientBoost, or forward selection) — more dims is not automatically better.

**Acceptance:** per-class recall for high/mid/low tom improves in the confusion matrix without regressing other classes.

---

## 4. Bring in acoustic + diverse training data

Current data is drum-machine-only. Add acoustic and real-world sources. Write loaders that normalize each into `(audio_path, label, source_kit)` rows so LOMO can hold out by `source_kit`.

| Dataset | Get it from | Why | Notes |
|---|---|---|---|
| **StemGMD** (single-hits partition) | `zenodo.org/records/7860223` (CC-BY 4.0) | Isolated, labeled acoustic one-shots; **3 separate tom classes** (high / mid-low / floor) + open/closed hat, ride, crash | Large — pull single hits, not the 1224 h of mixtures. Ref: `github.com/polimi-ispl/larsnet` |
| **Freesound One-Shot Percussive** | `zenodo.org/records/3665275`, or `mirdata` loader | Real-world recordings; diversity; raw material for FX | 10,254 clips, 1 s, 16 kHz |
| **E-GMD** | `magenta.tensorflow.org/datasets/e-gmd` | 43 kits (808/909 → acoustic) for cross-kit generalization | ~90 GB; loops — slice one-shots using the aligned MIDI onsets |
| **IDMT-SMT-Drums** | `zenodo.org/record/7544164` | Acoustic + sample-lib + synth, kick/snare/hat, onset-annotated | Small; has per-instrument training files |
| **ENST-Drums** | Télécom Paris ADASP (request access) | High-quality acoustic; sticks/rods/brushes/mallets; isolated hits | Multi-mic |

Data hygiene:
- **De-duplicate** across sources using embedding distance (cosine on the §2 embeddings, threshold near-duplicates). Duplicates leak across splits and inflate accuracy.
- Class taxonomy is skewed toward kick/snare/hat — oversample or rely on augmentation (§6) for toms/rides/crashes.
- Map every source's labels onto the app's fixed GM-style layout; keep tom subtypes distinct (don't merge high/mid/low — merging coherent-but-distinct classes is known to hurt, e.g. open vs closed hat).

---

## 5. Redesign FX as open-set rejection (per the requested approach)

FX is not a class; it's "none of the above." Model it as a confidence threshold.

- Classify into real drum types only. Then if `max_class_probability < τ`, label the slice **FX**.
- **Calibrate first.** Raw SVM/GB/MLP scores are not probabilities. Wrap the classifier in `sklearn`'s `CalibratedClassifierCV` (isotonic or Platt) so `τ` is meaningful. Tune `τ` on a validation split to trade off FX precision/recall.
- **CLAP fast path:** if the CLAP backend is active, run zero-shot — embed text prompts for each drum type, take cosine similarity to the audio embedding; `max_similarity < τ → FX`. This needs no training and the similarity *is* the confidence. Worth wiring as an optional mode.
- Expose `τ` as a tunable (it maps naturally to the app's existing classification flow).

**Acceptance:** held-out non-drum / weird sounds land in FX; real drums do not get over-rejected. Report FX precision/recall separately.

---

## 6. Augmentation (label-preserving, pipeline-consistent)

Essential given modest per-class data. Use `audiomentations` (waveform) and add SpecAugment only if/when a spectrogram-CNN path is built.

- Waveform: pitch shift (small range), time-stretch, gain, add noise. Pitch shift is the most informative for music — but **keep ranges label-preserving** (don't shift a high tom down into kick range, or a tom into snare territory).
- Apply augmentation **through the same segmentation/normalize path** as everything else (§1) so you don't reintroduce skew.
- Augment the training split only; never the validation/test split.

---

## 7. Evaluation protocol (use throughout)

- Primary metric: **macro-F1 under leave-one-machine/kit-out** (`source_kit` held out), via `model_analysis.py`. Macro, not accuracy — accuracy hides minority-class failure.
- Always emit a **confusion matrix** per run (watch tom↔tom, snare↔clap, hat↔ride).
- Track **calibration** (reliability curve) since FX depends on it.
- Splits are by `source_kit`/track, never random — random splits leak near-duplicates.
- Keep a frozen test set untouched until the end.

---

## 8. Suggested library set

`torch`, `transformers` (AST/CLAP/BEATs), `hear21passt` (PaSST), `openl3`, `librosa` (existing — onset/feature/pitch), `scikit-learn` (heads + `CalibratedClassifierCV` + RFECV), `audiomentations` (augmentation), `mirdata` (dataset loaders), `numpy`/`scipy` (existing).

---

## 9. Definition of done

1. One shared feature/embedding path for train and serve; skew regression test green.
2. Best embedding backend chosen by LOMO macro-F1, beating the hand-crafted baseline.
3. Tom subclasses separated (pitch feature + StemGMD tom labels); confusion matrix confirms.
4. FX implemented as calibrated-confidence rejection with a tunable threshold.
5. Acoustic datasets ingested, de-duplicated, LOMO-splittable by kit.
6. Augmentation pipeline in place, train-split only.
7. Final LOMO macro-F1 + confusion matrix + calibration curve reported vs the original baseline.
