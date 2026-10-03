# Limitations

Updated every milestone. Negative results go here.

## Scope (non-goals)
- 3D silicone masks and deepfake / virtual-camera injection attacks are out of scope.
- Passive RGB liveness is not a solved problem. Nothing here is claimed to be production-ready.

## What is NOT done yet (honest status)
- **No real-data result exists.** Datasets need signed access requests; every number in
  RESULTS.md is TBD. Synthetic smoke results are labelled and are not performance claims.
- **No real-data CNN weights or metric.** CNN training, ONNX parity and INT8 comparison are
  implemented and synthetic-tested; only licensed-data execution is pending.
- **No latency numbers for a real model.** `bench/latency.py` exists; only a labelled
  synthetic-smoke mode has been exercised.
- **No demo GIF.** It needs the owner's own face and device.
- Real datasets use a strict normalized manifest adapter. Raw archive-to-manifest conversion
  still requires access to, and inspection of, the licensed release.

## Known technical limitations
- Cross-dataset identity overlap (same person, different IDs) cannot be detected from
  metadata; cross-dataset protocols assume disjoint people.
- Haar face detection is weak on tilted, occluded or small faces; unmeasured on real data.
  When no face is found, the classical path falls back to a LOGGED centre crop and counts it.
- MediaPipe landmark indices in `active/landmarks.py` were written from memory and are
  untested; verify against the MediaPipe docs before real use.
- `NonceStore` is in-memory and per process: multi-instance deployments need a shared store.
- Active liveness can be defeated by a real-time deepfake or injection, and by a replayed video
  only if the challenge is predictable (see ACTIVE_LIVENESS.md, THREAT_MODEL.md).
- JPEG quality estimate is approximate (quantisation-table based); PNG gives n/a.
- `joblib` model files are pickles: load only trusted files.
- Fairness (skin tone, lighting, camera) is unmeasured.

## Findings during development
- OpenCV 5.0.0 shipped no Haar cascades, silently disabling face detection; pinned `<5`.
- A first `.gitignore` rule (`data/`) hid `src/liveness/data`; caught by a clean-clone check.
