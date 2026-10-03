# Model card

> **Status: no real-data trained model or performance claim exists yet.** Synthetic training
> only verifies the pipeline. Every TBD metric is filled from signed result JSON, never by hand.

## Overview
- **Task:** face presentation-attack detection (bona fide vs print / screen replay),
  passive RGB, single image and short-clip fusion; plus a landmark-based active challenge mode.
- **Intended use:** research and portfolio study of PAD evaluation methodology; a gate signal
  that a larger identity-verification flow may consume.
- **Not intended for:** sole-factor authentication, access to high-value assets, any use that
  needs a guarantee against 3D masks, deepfakes or injection attacks.

## Models
| Model | Purpose | Status |
|---|---|---|
| LBP + FFT features + SVM | Sanity floor | Implemented; real-data result TBD (not yet measured) |
| Small pretrained CNN (MobileNetV3-small / ResNet18) | Main passive model | Training/evaluation complete in code and synthetic smoke; real-data weights not produced |

## Training data
TBD. Planned sources are Replay-Attack and CASIA-FASD (restricted research data; see DATA.md).
Models trained on them inherit the applicable agreement restrictions;
**weights are not published.**

## Evaluation protocol
Subject-disjoint splits (docs/PROTOCOLS.md); thresholds chosen on validation only at a target
BPCER; APCER reported per attack type; cross-dataset train-A/test-B reported alongside
intra-dataset. Metrics: APCER, BPCER, ACER, EER, AUC with subject-level bootstrap CIs.

| Metric | Value |
|---|---|
| Intra-dataset ACER | TBD (not yet measured) |
| Cross-dataset ACER | TBD (not yet measured) |
| APCER @ BPCER 1% / 5% | TBD (not yet measured) |

## Known failure modes
To be filled from `docs/FAILURES.md` after real-data evaluation. Expected (hypotheses, not
findings): a large cross-dataset gap, sensitivity to unseen display types and print finishes,
and dataset shortcuts (compression, resolution, background).

## Fairness and bias caveats
Candidate datasets are small, recorded in a few labs, and skewed in demographics, cameras and
lighting. Per-skin-tone performance is **unmeasured** unless a dataset carries the attribute.
Do not assume parity across skin tones, lighting or camera quality.

## Out-of-scope attacks
3D / silicone masks, real-time deepfakes, camera/API injection (docs/THREAT_MODEL.md).

## Ethics and privacy
Faces are sensitive biometric data. Only licensed or consented data is used; nothing is
committed; the service does not store uploads; logs carry no images or embeddings.
