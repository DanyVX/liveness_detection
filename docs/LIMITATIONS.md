# Limitations

Updated every milestone. Negative results go here.

## Scope (non-goals)
- 3D silicone masks and deepfake / virtual-camera injection attacks are out of scope.
- Passive RGB liveness is not a solved problem. Nothing here is claimed to be production-ready.

## Known limitations so far (M0)
- No real dataset has been used yet; no measurement exists. See `RESULTS.md`.
- Cross-dataset identity overlap (same person under different IDs) cannot be detected from
  metadata; cross-dataset protocols assume disjoint people.
- Edge cases from the spec not yet covered by tests are tracked in `docs/EDGE_CASES.md`.
