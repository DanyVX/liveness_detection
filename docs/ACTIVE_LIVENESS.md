# Active liveness (challenge-response)

Defensive purpose: check that a live person responds to a randomly chosen, short-lived
challenge ("blink twice", "turn to your left", "open your mouth") at the time of the request.
It is a second signal next to the passive print/replay model, not a replacement for it.

## How it works

1. **Issue.** `generate_challenge(allowed_kinds, length)` draws a random sequence of steps
   (`BLINK_N` with a count, `TURN_LEFT`, `TURN_RIGHT`, `OPEN_MOUTH`) using `secrets`, plus a
   256-bit nonce, `issued_at` and `expires_at` (default TTL 30 s). Consecutive identical kinds
   are avoided when more than one kind is allowed. The server registers it in a `NonceStore`.
2. **Respond.** The client streams per-frame `FrameObservation`s (timestamp in ms, 21 compact
   landmarks or `None`, number of faces, landmark confidence) to an `ActiveLivenessSession`
   built from the nonce. The session takes the sequence from the server-side copy held by the
   store, never from the client.
3. **Decide.** `Decision` is `PASS`, `FAIL`, `INSUFFICIENT_EVIDENCE` or `IN_PROGRESS`, with a
   `ReasonCode`. All thresholds live in `SessionConfig` / `BlinkConfig` / `TurnConfig` /
   `MouthConfig`.

Steps must be completed in the issued order. A completed turn or mouth action that belongs to a
different step gives `WRONG_CHALLENGE_ORDER` (`strict_order`). Natural blinks during other steps are ignored.
Each step has a timeout (`CHALLENGE_NOT_PERFORMED`); the whole session has a hard cap on duration and frames
(`SESSION_TOO_LONG`). No frames are stored, so memory is bounded however long the stream runs.

### Detectors

- **Blink:** mean six-point EAR of both eyes. Per-user baseline from the first
  `baseline_frames` valid frames (75th percentile, so a blink during calibration does not
  poison it). Closed below 0.70 x baseline, reopened above 0.85 x baseline (hysteresis). A
  blink counts if the closed span is 50 ms to 1000 ms, measured from frame timestamps, so it is
  frame-rate independent. A 100 ms blink at 30 fps and a 400 ms slow blink both count (tested);
  one-frame glitches and eyes held shut do not.
- **Head turn:** normalised nose offset between the face-edge landmarks (not an angle). Needs a
  neutral pose first, then the threshold held for 150 ms.
- **Mouth open:** mouth aspect ratio above a threshold for 100 ms after a closed-mouth state.

### Mirroring

Selfie previews are usually flipped. `mirrored` is explicit everywhere. `TURN_LEFT` and
`TURN_RIGHT` always mean the subject's own left and right; `yaw_estimate` is positive for the
subject's left. With a wrong `mirrored` flag the user's correct turn is read as the opposite
direction and fails with `WRONG_CHALLENGE_ORDER` (tested). The flag is a property of the
client camera pipeline and must be set by whoever knows it.

### Replay protection

Each nonce is one-time: the session consumes it on construction. A used, expired or unknown
nonce yields `NONCE_REUSED`, `CHALLENGE_EXPIRED`, `NONCE_UNKNOWN`. Expiry is also re-checked on
every frame against the injected wall clock. The store is in-memory and per-process; a
multi-instance deployment needs a shared store with atomic compare-and-set (not provided).

## Minimum frame rate

Default minimum is 15 fps, measured as the median inter-frame interval over a 30-interval window
(after 10 intervals). Below it the session returns `INSUFFICIENT_EVIDENCE` / `FPS_TOO_LOW` instead
of silently missing blinks. `BlinkDetector.check()` reports the same. Duplicate or backwards
timestamps fail with `TIMESTAMPS_OUT_OF_ORDER` (`max_bad_timestamps` can tolerate a few).

## Glasses, low light, occlusion

When landmark confidence is low, landmarks are non-finite, or EAR is implausible (calibration EARs
with a large upward spread, baseline outside 0.12-0.60), the blink detector reports
`eyes_occluded`. A blink step that times out in that state ends as `INSUFFICIENT_EVIDENCE` /
`EYES_OCCLUDED`, not `FAIL`, so the caller can offer a non-blink challenge. Thick frames, glare
and low light can still cause misses; the defaults have not been tuned on real users.
Low confidence also blanks the turn/mouth detectors.

## Accessibility

`allowed_kinds` is configurable per user: someone who cannot blink on demand can be given turns
and mouth-open only, and so on. Limits: fewer kinds means a smaller challenge space (a single
kind allowed gives only counts and repeats), so compensate with more steps or a shorter TTL;
users who cannot turn their head or open their mouth on cue may have no workable active
challenge and need a different channel (human review, other factors). Do not offer the
alternative automatically on failure in a way an attacker can exploit by always requesting the
weakest set.

## Limitations (honest)

- A recording of a real person performing the *requested* actions defeats this check if the
  recording matches a challenge. Defence rests on the sequence being unpredictable and the TTL
  short; it is not cryptographic. With a small allowed set the odds of a pre-recorded clip
  matching are not negligible.
- Real-time deepfakes, face puppeteering and virtual-camera / injection attacks that react to the
  challenge can pass. Nothing here detects injection or verifies the camera.
- The client supplies timestamps and landmarks. If the landmark extraction runs on an untrusted
  client, it can submit fabricated observations; run extraction server-side on the raw video.
- No identity binding: it does not check the person responding is the account holder.
- Not covered: 3D depth, rPPG, audio, challenge UI/accessibility of instructions, rate limiting of
  challenge issuance, shared nonce stores, adaptive per-user thresholds beyond the blink baseline.
- `FaceMeshAdapter` is untested because mediapipe is unavailable in CI; its Face Mesh index lists
  and API usage are unverified and must be checked against the MediaPipe documentation.
- Synthetic landmark tests prove the logic, not real-world accuracy; no real-user evaluation exists.
