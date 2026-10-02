# Threat model

Purpose is strictly defensive: decide whether a face in front of a camera is a live person
or a presentation attack. No attack-generation tooling is included. The synthetic fixture
is generic image degradation used for tests only.

Terminology follows ISO/IEC 30107-3 (APCER, BPCER, ACER).

## Attack taxonomy

| Attack | Description | Passive CNN / classical | Active challenge | Status |
|---|---|---|---|---|
| Printed photo | Paper print held to the camera (matte or glossy, flat or bent) | Target | Defeats it (no blink or head turn) | In scope. Measured only on real data, see RESULTS.md |
| Screen replay (photo) | Photo shown on a phone or tablet | Target. Moire and screen-glare cues | Defeats it | In scope |
| Screen replay (video) | Recorded video of the victim, played on a screen | Target (passive); weak if video is high quality | **Can defeat a fixed challenge.** Random challenge + nonce + short expiry mitigates a *previously recorded* video | In scope, partial |
| Cut-out / partial face | Print with eye holes, a real eye behind it | Not specifically trained | Blink check helps | Not measured |
| 3D / silicone mask | Worn mask | **Out of scope** | Likely defeats blink and turn | Out of scope |
| Real-time deepfake / face swap on a live feed | Attacker performs the challenge, face is swapped live | **Out of scope** | **Defeats it** | Out of scope |
| Injection (virtual camera, hooked API) | Frames never come from a physical camera | **Out of scope** (needs device attestation) | Defeats it | Out of scope |
| Challenge replay | Resubmitting a captured successful session | n/a | Nonce is one-time and expires | Mitigated (see ACTIVE_LIVENESS.md) |

## What this does NOT defend against

- Masks, deepfakes and camera injection (above).
- Attacks the training data did not contain: cross-dataset results (M3) quantify how large
  the generalization gap is; expect it to be large.
- A targeted adaptive attacker who has the model and can optimize inputs against it.
- Compromise of the capture device or of this service.

## Assets and abuse considerations

- Inputs are biometric data. The API does not persist uploads, logs contain no images,
  embeddings or subject identifiers, and model weights are not published (derived from
  non-commercial datasets).
- A liveness score is one signal. It should not be the sole authentication factor, and
  `INSUFFICIENT_QUALITY` / `INSUFFICIENT_EVIDENCE` must route to retry or another method,
  never to a silent accept.
- Fairness: skin tone, lighting and camera skew are unmeasured unless a dataset provides
  the attribute; see MODEL_CARD.md.
