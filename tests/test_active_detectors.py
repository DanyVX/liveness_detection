import numpy as np
import pytest

from active_helpers import Script, make_landmarks
from liveness.active.detectors import (
    BlinkConfig,
    BlinkDetector,
    DetectorStatus,
    FpsMonitor,
    HeadTurnDetector,
    MouthOpenDetector,
    Outcome,
    TurnDirection,
)


def run_blink(script: Script, **kw: float) -> BlinkDetector:
    det = BlinkDetector(BlinkConfig(**kw) if kw else None)  # type: ignore[arg-type]
    for t, lm in script.landmark_frames():
        det.update(t, lm)
    return det


@pytest.mark.parametrize("duration", [100.0, 400.0])
def test_fast_and_slow_blinks_counted(duration: float) -> None:
    det = run_blink(Script(duration_ms=3000, blinks=[(1500.0, duration)]))
    assert det.blink_count == 1
    assert det.status is DetectorStatus.OK
    assert det.check(1).outcome is Outcome.SATISFIED


def test_fast_blink_counted_at_any_phase() -> None:
    for t0 in np.linspace(0, 33, 8):
        det = run_blink(Script(duration_ms=3000, blinks=[(1500.0, 100.0)], t0=float(t0)))
        assert det.blink_count == 1


def test_three_blinks_and_baseline_adapts_to_user() -> None:
    det = run_blink(Script(duration_ms=5000, blinks=[(1500, 150), (2500, 150), (3500, 150)]))
    assert det.blink_count == 3
    assert det.baseline == pytest.approx(0.3, abs=0.01)


def test_no_blink_and_single_frame_glitch_rejected() -> None:
    det = BlinkDetector()
    open_lm = make_landmarks()
    for i in range(40):
        t = i * 1000 / 30
        det.update(t, make_landmarks(eye_open=0.05) if i == 25 else open_lm)
    assert det.blink_count == 0
    assert det.check(1).outcome is Outcome.PENDING


def test_eyes_held_shut_is_not_a_blink() -> None:
    det = run_blink(Script(duration_ms=5000, blinks=[(1500.0, 2500.0)]))
    assert det.blink_count == 0


def test_works_per_user_with_small_eyes() -> None:
    det = BlinkDetector()
    for i in range(90):
        t = i * 1000 / 30
        openness = 0.5 if not 50 <= i <= 54 else 0.02  # baseline EAR 0.15
        det.update(t, make_landmarks(eye_open=openness))
    assert det.blink_count == 1


def test_low_fps_reported_not_silent() -> None:
    det = run_blink(Script(duration_ms=4000, fps=5.0, blinks=[(2000.0, 150.0)]))
    assert det.status is DetectorStatus.FPS_TOO_LOW
    assert det.fps == pytest.approx(5.0)
    verdict = det.check(1)
    assert verdict.outcome is Outcome.INSUFFICIENT_EVIDENCE
    assert verdict.reason is DetectorStatus.FPS_TOO_LOW


def test_fps_monitor_unknown_until_enough_intervals() -> None:
    m = FpsMonitor(15.0, min_intervals=5)
    for t in (0.0, 33.0, 66.0):
        m.observe(t)
    assert m.fps is None
    assert not m.too_low


def test_low_confidence_reports_eyes_occluded() -> None:
    det = BlinkDetector()
    for i in range(60):
        det.update(i * 33.3, make_landmarks(), confidence=0.2)
    assert det.status is DetectorStatus.EYES_OCCLUDED
    assert det.check(1).reason is DetectorStatus.EYES_OCCLUDED
    for i in range(60, 66):
        det.update(i * 33.3, make_landmarks(), confidence=0.9)
    assert det.status is DetectorStatus.CALIBRATING
    for i in range(66, 80):
        det.update(i * 33.3, make_landmarks(), confidence=0.9)
    assert det.status is DetectorStatus.OK


def test_implausible_ear_variance_reports_eyes_occluded() -> None:
    det = BlinkDetector()
    for i in range(24):
        det.update(i * 33.3, make_landmarks(eye_open=1.3 if i % 2 else 0.1))
    assert det.status is DetectorStatus.EYES_OCCLUDED
    for i in range(24, 60):  # a clean recalibration recovers
        det.update(i * 33.3, make_landmarks())
    assert det.status is DetectorStatus.OK


def test_nonfinite_landmarks_and_missing_face() -> None:
    det = BlinkDetector()
    bad = make_landmarks()
    bad[2, 1] = np.inf
    det.update(0.0, bad)
    det.update(33.0, None)
    assert det.blink_count == 0


def test_gap_aborts_closure() -> None:
    det = BlinkDetector()
    for i in range(20):
        det.update(i * 33.3, make_landmarks())
    det.update(700.0, make_landmarks(eye_open=0.05))
    det.update(1700.0, make_landmarks())  # 1 s gap while "closed": not a blink
    assert det.blink_count == 0


def test_blink_memory_bounded() -> None:
    cfg = BlinkConfig(max_events=5)
    det = BlinkDetector(cfg)
    s = Script(duration_ms=40000, blinks=[(1500.0 + 1000 * k, 150.0) for k in range(35)])
    for t, lm in s.landmark_frames():
        det.update(t, lm)
    assert det.blink_count == 35
    assert len(det.blinks) == 5


def turn_run(direction: TurnDirection, script: Script, mirrored: bool) -> bool:
    det = HeadTurnDetector(direction, mirrored=mirrored)
    for t, lm in script.landmark_frames():
        det.update(t, lm)
    return det.completed


@pytest.mark.parametrize("mirrored", [False, True])
def test_turn_directions_respect_mirroring(mirrored: bool) -> None:
    sweep = [(0.0, 0.0), (500.0, 0.0), (900.0, 0.6), (2000.0, 0.6)]
    left = Script(duration_ms=2000, yaw_keys=sweep, mirrored=mirrored)
    right = Script(duration_ms=2000, yaw_keys=[(t, -y) for t, y in sweep], mirrored=mirrored)
    assert turn_run(TurnDirection.LEFT, left, mirrored)
    assert not turn_run(TurnDirection.RIGHT, left, mirrored)
    assert turn_run(TurnDirection.RIGHT, right, mirrored)
    assert not turn_run(TurnDirection.LEFT, right, mirrored)
    # Wrong mirror flag swaps the meaning.
    assert turn_run(TurnDirection.RIGHT, left, not mirrored)


def test_turn_requires_neutral_first_and_hold() -> None:
    det = HeadTurnDetector(TurnDirection.LEFT, mirrored=False)
    for i in range(30):  # already turned from the start: never armed
        det.update(i * 33.3, make_landmarks(yaw=0.6))
    assert not det.completed
    short = HeadTurnDetector(TurnDirection.LEFT, mirrored=False)
    short.update(0.0, make_landmarks())
    short.update(33.0, make_landmarks(yaw=0.6))
    short.update(66.0, make_landmarks())  # 33 ms flick is below hold_ms
    assert not short.completed
    short.update(100.0, None)
    short.reset()
    assert short.completed_at is None


def test_turn_ignores_low_confidence_and_degenerate() -> None:
    det = HeadTurnDetector(TurnDirection.LEFT, mirrored=False)
    det.update(0.0, make_landmarks())
    for i in range(1, 20):
        det.update(i * 33.3, make_landmarks(yaw=0.6), confidence=0.1)
    assert not det.completed
    flat = make_landmarks()
    flat[19:21] = flat[19]
    det.update(700.0, flat)
    assert not det.completed


def test_mouth_open() -> None:
    s = Script(duration_ms=2000, mouth_keys=[(0.0, 0.0), (500.0, 0.0), (700.0, 0.7), (2000.0, 0.7)])
    det = MouthOpenDetector()
    for t, lm in s.landmark_frames():
        det.update(t, lm)
    assert det.completed
    assert det.completed_at is not None
    det.reset()
    assert not det.completed
    det.update(0.0, make_landmarks(mouth_open=0.7))  # already open: not armed
    det.update(300.0, make_landmarks(mouth_open=0.7))
    assert not det.completed
    det.update(400.0, None)
    flat = make_landmarks()
    flat[12] = flat[15]
    det.update(500.0, flat)
    assert not det.completed
