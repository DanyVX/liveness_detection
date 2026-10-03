# liveness-detection

Face presentation-attack detection (print / screen replay) built around honest evaluation:
ISO/IEC 30107-3 metrics, subject-disjoint protocols, and cross-dataset generalization.

**Status: pipeline, metrics, classical baseline, CNN training path, active-challenge logic and serving layer are built and tested on a synthetic fixture. No real dataset has been used and no real-performance number exists yet. Real-data CNN evaluation remains blocked on dataset access.**

## Key results

TBD (not yet measured). See [docs/RESULTS.md](docs/RESULTS.md).

## Why this exists

Most student anti-spoofing projects report intra-dataset numbers on random splits, which leak
subjects and hide the cross-dataset collapse. This repo fixes the protocol first and reports
the drop plainly.

## Quickstart

```bash
uv sync
uv run pytest
make lint typecheck
```

Tests run on a generated synthetic fixture, with no real data, network, or GPU needed.

## Usage

```bash
# synthetic smoke pipeline (labelled "not a real result")
PYTHONPATH=src uv run python bench/smoke_pipeline.py --out-dir /tmp/smoke
PYTHONPATH=src uv run python scripts/render_results.py --results-dir /tmp/smoke \
    --out /tmp/smoke/RESULTS.generated.md --plots-dir /tmp/smoke/plots

# serve an ONNX model (contract: input "input" [N,3,H,W], output "logit" [N,1])
LIVENESS_API_MODEL_PATH=model.onnx LIVENESS_API_DECISION_THRESHOLD=0.5 \
LIVENESS_API_CLIP_ACCEPT_THRESHOLD=0.6 LIVENESS_API_CLIP_REJECT_THRESHOLD=0.4 \
uv run uvicorn --factory liveness.api.app:create_app
```

Endpoints: `GET /healthz`, `GET /v1/model`, `POST /v1/score`, `POST /v1/decide`. Every response
carries `score`, `decision`, `reason_code`, `model_version`. Unusable input yields
`INSUFFICIENT_QUALITY`, never `SPOOF`.

## Architecture

```mermaid
flowchart LR
  D[Dataset loaders] --> P[Subject-disjoint Protocol]
  P --> M[Model: LBP/FFT+SVM or CNN]
  M --> S[Scores]
  S --> E[eval: val-only thresholds, APCER/BPCER/ACER, CIs]
  E --> R[results/*.json + env info]
  R --> T[Rendered tables and plots]
  I[ONNX scorer] --> F[Temporal fusion] --> G[LivenessGate / FastAPI]
  A[Active challenge: landmarks + nonce] --> G
```

## Design

Protocols: [docs/PROTOCOLS.md](docs/PROTOCOLS.md). Decisions: [docs/DESIGN.md](docs/DESIGN.md).

## Data and licensing

Real datasets are research / non-commercial and are **never** committed or redistributed
([DATA.md](DATA.md)). Code is MIT. **Trained weights will not be published** because they derive
from non-commercial data.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). 3D masks and injection attacks are out of scope.

## Roadmap

Done (synthetic only): M0 protocols, M1 classical baseline, M3 eval core, M4 active challenge,
M5 fusion, M6 serving, M7 docs drafts, and the M2 CNN training path.

Blocked: dataset access (Replay-Attack, CASIA-FASD), real-data cross-dataset results, latency
on a real model, and demo GIF. Install the optional `train` extra for CNN training:
`uv sync --extra train --group dev`.
