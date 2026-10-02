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
