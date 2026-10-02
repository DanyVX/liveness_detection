# Evaluation protocols

Written before any training (M0). Changing a protocol after seeing results requires a
new protocol name and a Decision-log entry in `DESIGN.md`.

## Rules

1. **Subject-disjoint, always.** Splits are made by subject, never by file or frame.
   `Protocol` refuses to construct if a subject (or file) is in two splits
   (`tests/test_protocols.py`, including a hypothesis property test).
2. **Identity key = `(dataset, subject_id)`.** The same person under different IDs in two
   datasets cannot be detected from metadata. Cross-dataset results assume the datasets'
   subject pools are different people. This is unverifiable and listed in `LIMITATIONS.md`.
3. **Every split has both classes**, or the protocol is rejected (metrics undefined).
4. **Thresholds and hyperparameters are chosen on `val` only.** `test` is touched once for
   the final number per protocol.
5. **Manifests are subject-level JSON** (`Protocol.write_manifest`), hashed with SHA-256, and
   the hash is stored in every results file.
6. **Frames from one video stay in one split** (follows from rule 1).

## Protocols

| Name | Train | Val | Test | Status |
|---|---|---|---|---|
| `synthetic-v0` | 60% of subjects | 20% | 20% | CI only. Never reported as a result. |
| `replay_attack-official` | official train subjects | official devel subjects | official test subjects | Needs access (see DATA.md) |
| `casia_fasd-intra` | official train subjects minus a held-out val subject group | held-out subjects from train | official test subjects | Needs access. The dataset has no dev set, so val is carved from train subjects by seeded subject split. |
| `cross: casia->replay` | CASIA train | CASIA val | Replay test | Needs access |
| `cross: replay->casia` | Replay train | Replay devel | CASIA test | Needs access |

Cross-dataset rule: the **threshold is picked on the source-domain val split** and applied
unchanged to the target test split. The threshold shift between domains is reported, not hidden.

## Attack types

`print`, `replay` (labels from `AttackType`). Per-type APCER is reported, not just the
average. Source-dataset attack subtypes (e.g. warped/cut photo, phone vs tablet replay) are
mapped to these two and the mapping is recorded in `DATA.md` when each loader is written.
