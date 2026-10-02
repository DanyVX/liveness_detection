# Contributing

- Setup: `make setup`. Checks: `make lint typecheck test`.
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`).
- Every bug fix needs a regression test. Tests must be deterministic and need no network or GPU.
- Never commit datasets, weights, personal images or video, or `.env`.
- Any number in the README or RESULTS.md must come from a script writing `results/*.json`.
