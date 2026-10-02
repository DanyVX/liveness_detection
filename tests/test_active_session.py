from dataclasses import replace

import numpy as np
import pytest

from active_helpers import Script, make_challenge, make_landmarks
from liveness.active.challenge import ChallengeKind as K
from liveness.active.challenge import ChallengeStep, NonceStore
from liveness.active.session import (
    ActiveLivenessSession,
    Decision,
    FrameObservation,
    ReasonCode,
    Result,
    SessionConfig,
)

NEUTRAL_THEN_LEFT = [(0.0, 0.0), (3000.0, 0.0), (3400.0, 0.6), (4200.0, 0.6), (4600.0, 0.0)]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def setup(
    steps: list[K] | list[ChallengeStep],
    *,
    mirrored: bool = False,
    config: SessionConfig | None = None,
) -> tuple[ActiveLivenessSession, NonceStore, Clock]:
    clock = Clock()
    store = NonceStore(clock)
    ch = make_challenge(steps)
    store.register(ch)
    return (
        ActiveLivenessSession(ch.nonce, store, mirrored=mirrored, config=config, clock=clock),
        (store),
        clock,
    )


def feed(session: ActiveLivenessSession, frames: list[FrameObservation]) -> Result:
    res = session.result
    for f in frames:
        res = session.update(f)
        if res.decision is not Decision.IN_PROGRESS:
            return res
    return session.finish()


def full_script(mirrored: bool = False, **kw: object) -> Script:
    return Script(
        duration_ms=8000,
        blinks=[(1500.0, 150.0), (2300.0, 150.0)],
        yaw_keys=NEUTRAL_THEN_LEFT,
        mouth_keys=[(0.0, 0.0), (5000.0, 0.0), (5200.0, 0.7), (5800.0, 0.7), (6000.0, 0.0)],
        mirrored=mirrored,
        **kw,  # type: ignore[arg-type]
    )


CHALLENGE = [ChallengeStep(K.BLINK_N, 2), ChallengeStep(K.TURN_LEFT), ChallengeStep(K.OPEN_MOUTH)]


def test_happy_path_passes() -> None:
    s, _, _ = setup(CHALLENGE)
    res = feed(s, full_script().frames())
    assert (res.decision, res.reason) == (Decision.PASS, ReasonCode.OK)
    assert res.steps_completed == res.total_steps == 3
    assert s.update(full_script().frames()[0]) is res  # terminal: later frames ignored


def test_in_progress_before_completion() -> None:
    s, _, _ = setup(CHALLENGE)
    res = s.update(full_script().frames()[0])
    assert res.decision is Decision.IN_PROGRESS
    assert res.reason is ReasonCode.IN_PROGRESS
    assert res.total_steps == 3
    assert s.challenge is not None
    assert s.poll().decision is Decision.IN_PROGRESS


def test_mirrored_camera_left_turn() -> None:
    s, _, _ = setup([K.TURN_LEFT], mirrored=True)
    res = feed(s, full_script(mirrored=True).frames())
    assert res.decision is Decision.PASS


def test_mirror_flag_wrong_means_wrong_direction() -> None:
    s, _, _ = setup([K.TURN_LEFT], mirrored=False)
    res = feed(s, full_script(mirrored=True).frames())
    assert (res.decision, res.reason) == (Decision.FAIL, ReasonCode.WRONG_CHALLENGE_ORDER)


def test_steps_must_follow_issued_order() -> None:
    s, _, _ = setup([K.OPEN_MOUTH, K.TURN_LEFT])
    script = Script(
        duration_ms=6000,
        yaw_keys=[(0.0, 0.0), (1500.0, 0.0), (1900.0, 0.6), (3000.0, 0.6)],
        mouth_keys=[(0.0, 0.0), (4000.0, 0.0), (4200.0, 0.7), (5000.0, 0.7)],
    )
    res = feed(s, script.frames())
    assert (res.decision, res.reason) == (Decision.FAIL, ReasonCode.WRONG_CHALLENGE_ORDER)
    assert "turn_left" in res.detail


def test_natural_blinks_during_pose_step_are_ignored() -> None:
    s, _, _ = setup([K.TURN_LEFT])
    script = Script(
        duration_ms=6000,
        blinks=[(1200.0, 150.0), (2000.0, 150.0), (3000.0, 150.0)],
        yaw_keys=[(0.0, 0.0), (3500.0, 0.0), (3900.0, 0.6), (5000.0, 0.6)],
    )
    assert feed(s, script.frames()).decision is Decision.PASS


def test_strict_order_can_be_disabled() -> None:
    cfg = SessionConfig(strict_order=False, step_timeout_ms=2000)
    s, _, _ = setup([K.OPEN_MOUTH], config=cfg)
    script = Script(duration_ms=6000, yaw_keys=[(0.0, 0.0), (1200.0, 0.0), (1600.0, 0.6)])
    res = feed(s, script.frames())
    assert res.reason is ReasonCode.CHALLENGE_NOT_PERFORMED


def test_challenge_not_performed_times_out() -> None:
    s, _, _ = setup([K.OPEN_MOUTH])
    res = feed(s, Script(duration_ms=12000).frames())
    assert (res.decision, res.reason) == (Decision.FAIL, ReasonCode.CHALLENGE_NOT_PERFORMED)


def test_stream_end_before_completion_is_not_a_pass() -> None:
    s, _, _ = setup([K.OPEN_MOUTH])
    res = feed(s, Script(duration_ms=3000).frames())
    assert res.reason is ReasonCode.CHALLENGE_NOT_PERFORMED
    assert s.finish() is res


def test_replayed_challenge_is_rejected() -> None:
    clock = Clock()
    store = NonceStore(clock)
    ch = make_challenge(CHALLENGE)
    store.register(ch)
    first = ActiveLivenessSession(ch.nonce, store, clock=clock)
    assert feed(first, full_script().frames()).decision is Decision.PASS
    # A recording of that same performance, presented for the same nonce.
    replay = ActiveLivenessSession(ch.nonce, store, clock=clock)
    assert replay.result.decision is Decision.FAIL
    assert replay.result.reason is ReasonCode.NONCE_REUSED
    assert feed(replay, full_script().frames()).reason is ReasonCode.NONCE_REUSED


def test_unknown_nonce() -> None:
    s = ActiveLivenessSession("0" * 64, NonceStore())
    assert s.result.reason is ReasonCode.NONCE_UNKNOWN
    assert s.challenge is None


def test_expired_at_start_and_mid_session() -> None:
    clock = Clock()
    store = NonceStore(clock)
    ch = make_challenge(CHALLENGE, ttl_s=5)
    store.register(ch)
    clock.now = ch.expires_at + 1
    assert ActiveLivenessSession(ch.nonce, store, clock=clock).result.reason is (
        ReasonCode.CHALLENGE_EXPIRED
    )
    ch2 = make_challenge(CHALLENGE, nonce="b" * 64, ttl_s=5)
    clock.now = ch2.issued_at
    store.register(ch2)
    s = ActiveLivenessSession(ch2.nonce, store, clock=clock)
    frames = full_script().frames()
    s.update(frames[0])
    clock.now = ch2.expires_at + 0.1
    assert s.update(frames[1]).reason is ReasonCode.CHALLENGE_EXPIRED


def test_poll_detects_expiry_without_frames() -> None:
    s, _, clock = setup(CHALLENGE)
    clock.now += 100
    assert s.poll().reason is ReasonCode.CHALLENGE_EXPIRED


def test_multiple_faces_fail_after_tolerance() -> None:
    s, _, _ = setup(CHALLENGE)
    frames = [replace(f, num_faces=2) for f in full_script().frames()[:30]]
    res = feed(s, frames)
    assert (res.decision, res.reason) == (Decision.FAIL, ReasonCode.MULTIPLE_FACES)


def test_transient_second_face_is_tolerated() -> None:
    s, _, _ = setup([K.OPEN_MOUTH])
    script = Script(
        duration_ms=7000, mouth_keys=[(0.0, 0.0), (3000.0, 0.0), (3200.0, 0.7), (4000.0, 0.7)]
    )
    frames = script.frames()
    frames[40] = replace(frames[40], num_faces=2)
    assert feed(s, frames).decision is Decision.PASS


def test_face_lost_mid_challenge_times_out() -> None:
    s, _, _ = setup(CHALLENGE)
    frames = full_script().frames()
    for i in range(60, 130):
        frames[i] = FrameObservation(frames[i].timestamp_ms, None, 0)
    res = feed(s, frames)
    assert (res.decision, res.reason) == (Decision.FAIL, ReasonCode.FACE_LOST_TIMEOUT)


def test_brief_face_loss_is_tolerated() -> None:
    s, _, _ = setup([K.OPEN_MOUTH])
    script = Script(
        duration_ms=7000, mouth_keys=[(0.0, 0.0), (3000.0, 0.0), (3200.0, 0.7), (4500.0, 0.7)]
    )
    frames = script.frames()
    for i in range(30, 45):
        frames[i] = FrameObservation(frames[i].timestamp_ms, None, 0)
    assert feed(s, frames).decision is Decision.PASS


def test_duplicate_and_backwards_timestamps_fail() -> None:
    frames = full_script().frames()
    s, _, _ = setup(CHALLENGE)
    dup = [*frames[:10], frames[9]]
    assert feed(s, dup).reason is ReasonCode.TIMESTAMPS_OUT_OF_ORDER
    s2, _, _ = setup(CHALLENGE)
    back = [*frames[:10], replace(frames[3])]
    assert feed(s2, back).reason is ReasonCode.TIMESTAMPS_OUT_OF_ORDER
    s3, _, _ = setup(CHALLENGE)
    nan = [*frames[:5], replace(frames[5], timestamp_ms=float("nan"))]
    assert feed(s3, nan).reason is ReasonCode.TIMESTAMPS_OUT_OF_ORDER


def test_bad_timestamps_tolerated_when_configured() -> None:
    cfg = SessionConfig(max_bad_timestamps=1)
    s, _, _ = setup([K.OPEN_MOUTH], config=cfg)
    script = Script(
        duration_ms=7000, mouth_keys=[(0.0, 0.0), (3000.0, 0.0), (3200.0, 0.7), (4500.0, 0.7)]
    )
    frames = script.frames()
    frames.insert(20, frames[19])
    assert feed(s, frames).decision is Decision.PASS


def test_session_hard_cap_duration_and_frames() -> None:
    s, _, _ = setup(CHALLENGE, config=SessionConfig(max_duration_ms=3000))
    res = feed(s, Script(duration_ms=9000).frames())
    assert res.reason is ReasonCode.SESSION_TOO_LONG
    s2, _, _ = setup(CHALLENGE, config=SessionConfig(max_frames=50))
    assert feed(s2, Script(duration_ms=9000).frames()).reason is ReasonCode.SESSION_TOO_LONG


def test_long_session_memory_is_bounded() -> None:
    cfg = SessionConfig(max_duration_ms=1e9, max_frames=10**9, step_timeout_ms=1e9)
    s, _, _ = setup([K.OPEN_MOUTH], config=cfg)
    lm = make_landmarks()
    for i in range(20000):
        s.update(FrameObservation(i * 33.0, lm, 1))
    assert s.result.decision is Decision.IN_PROGRESS
    assert not any(isinstance(v, list) and len(v) > 100 for v in vars(s).values())


def test_low_fps_is_insufficient_evidence() -> None:
    s, _, _ = setup([ChallengeStep(K.BLINK_N, 1)])
    res = feed(s, Script(duration_ms=6000, fps=5.0, blinks=[(2000.0, 150.0)]).frames())
    assert (res.decision, res.reason) == (Decision.INSUFFICIENT_EVIDENCE, ReasonCode.FPS_TOO_LOW)
    assert "fps=" in res.detail


def test_occluded_eyes_are_insufficient_not_fail() -> None:
    s, _, _ = setup([ChallengeStep(K.BLINK_N, 1)])
    frames = [replace(f, landmark_confidence=0.1) for f in Script(duration_ms=12000).frames()]
    res = feed(s, frames)
    assert (res.decision, res.reason) == (
        Decision.INSUFFICIENT_EVIDENCE,
        ReasonCode.EYES_OCCLUDED,
    )


def test_non_blink_alternative_passes_without_blinking() -> None:
    s, _, _ = setup([K.TURN_RIGHT])
    script = Script(
        duration_ms=6000, yaw_keys=[(0.0, 0.0), (2000.0, 0.0), (2400.0, -0.6), (4000.0, -0.6)]
    )
    assert feed(s, script.frames()).decision is Decision.PASS


def test_observation_rejects_bad_landmark_shape() -> None:
    with pytest.raises(ValueError, match="landmarks"):
        FrameObservation(0.0, np.zeros((5, 2)), 1)
