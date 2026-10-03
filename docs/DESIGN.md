# Design

## Decision log

| Date | Decision | Options considered | Why |
|---|---|---|---|
| 2026-10-02 | `Protocol` validates disjointness in its constructor | Check in a test only; check in the training script | Leakage should be impossible to build, not just unlikely. A test then proves the constructor rejects it. |
| 2026-10-02 | Splits stored as subject-level manifests with a SHA-256 | Random seed only; file lists | Auditable, diff-able, tiny, independent of local paths; hash goes in every results file. |
| 2026-10-02 | Subject identity = `(dataset, subject_id)` | Raw `subject_id` | IDs like `001` repeat across datasets and would raise false leakage in cross-dataset protocols. |
| 2026-10-02 | Protocol rejects splits missing a class | Allow and warn | APCER/BPCER are undefined with one class. Failing early beats a NaN in a results table. |
| 2026-10-02 | `uv` + `pyproject.toml`, hatchling, `src/` layout | pip + requirements.txt | Lockfile reproducibility. CLAUDE.md default. |
| 2026-10-02 | Real dataset loaders are stubs until data is inspected | Write loaders from memory of the layouts | Guessing a layout is inventing facts. |
| 2026-10-02 | Weights not published | MIT weights | Training data is likely non-commercial. See DATA.md. |
| 2026-10-02 | Dev environment is CPU-only (4 cores, 15 GB) | n/a | CPU-first design; Colab/Kaggle heavy-path docs arrive in M2. |
| 2026-10-02 | torch stack in optional `train` extra, not core deps | Core dep | CPU wheel index was unreachable from the sandbox; PyPI Linux torch wheel is 555 MB (> 500 MB ask-first rule) plus CUDA libs. Everything else was built and tested torch-free. |
| 2026-10-02 | Pin `opencv-python-headless<5` | Allow 5.x | OpenCV 5.0.0 shipped no `CascadeClassifier`/Haar files, so face detection silently degraded. |
| 2026-10-02 | Score = higher means more bona fide everywhere; ACER uses worst-type APCER (mean variant also reported) | Mean-only APCER | Worst-type is the conservative reading of ISO/IEC 30107-3 per-attack reporting. |
| 2026-10-02 | Bootstrap CIs resample subjects, not samples | Sample-level bootstrap | Samples from one subject are correlated; sample-level CIs would be too narrow. |
| 2026-10-02 | Cross-domain threshold = source-val threshold, applied unchanged to the target | Re-tune on target | Re-tuning on the target test set would be leakage. The target-EER threshold is reported only as a diagnostic of calibration shift. |
| 2026-10-02 | Insufficient quality / evidence are explicit decisions | Force LIVE/SPOOF | An unusable frame is not evidence of an attack; forcing SPOOF would also hide quality problems. |
| 2026-10-02 | Active challenges: `secrets`-based randomness, nonce one-time use, short expiry | Fixed challenge set | Unpredictability is the only defence against replayed recordings of a previous challenge. |
| 2026-10-03 | Licensed datasets enter through a normalized, checksummed manifest | Guess each proprietary archive layout | The adapter is testable without redistributing data and forces subject/split decisions to be explicit. |
| 2026-10-03 | CNN early stopping uses validation ACER; CLI requires at least three seeds | Stop on training loss or report one lucky seed | This matches the target metric and exposes variance without looking at test labels. |
