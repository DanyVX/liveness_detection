# liveness-detection

Face presentation-attack detection (print / screen replay) built around honest evaluation:
ISO/IEC 30107-3 metrics, subject-disjoint protocols, and cross-dataset generalization.

**Status: M0 (setup + protocol design). No model exists yet and nothing has been measured.**

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

## Design

Protocols: [docs/PROTOCOLS.md](docs/PROTOCOLS.md). Decisions: [docs/DESIGN.md](docs/DESIGN.md).

## Data and licensing

Real datasets are research / non-commercial and are **never** committed or redistributed
([DATA.md](DATA.md)). Code is MIT. **Trained weights will not be published** because they derive
from non-commercial data.

## Limitations

See [docs/LIMITATIONS.md](docs/LIMITATIONS.md). 3D masks and injection attacks are out of scope.

## Roadmap

M1 classical baseline, M2 CNN, M3 cross-dataset eval, M4 active challenge, M5 temporal fusion,
M6 ONNX + API, M7 docs and release.
