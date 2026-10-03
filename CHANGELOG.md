# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: SemVer.

## [Unreleased]

### Added
- M2: MobileNetV3/ResNet-compatible CNN training with balanced sampling, JPEG/color/blur/crop
  augmentation, validation-ACER early stopping, exact single-worker resume, AMP safeguards,
  three-seed aggregation, crop-margin ablation and intra/cross-dataset reporting.
- Licensed-data `manifest.csv` adapter with path, checksum, class and subject-leakage validation.
- M1: LBP + FFT features, JPEG-bias detector, classical SVM baseline.
- M3 core: ISO/IEC 30107-3 metrics, subject-level bootstrap CIs, val-only thresholds, cross-dataset report, results rendering, synthetic smoke pipeline.
- M4: active liveness (landmark geometry, nonce-protected random challenges, session state machine).
- M5/M6: temporal fusion with hysteresis, ONNX inference, `LivenessGate`, FastAPI service, latency bench, Dockerfile.
- Threat model, model card (no measured claims), edge-case matrix.
- M0: project scaffold, unified `Sample` interface, subject-disjoint `Protocol` with leakage
  checks, synthetic CI fixture, settings and logging, CI workflow, protocol docs.
