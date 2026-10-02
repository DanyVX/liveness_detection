"""Challenge-response state machine. Consumes frame observations, emits a Result.

Memory is bounded: no frames are stored, only detector state and small capped deques.
Timestamps are capture times in ms (monotonic per session); expiry uses the injected wall clock.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from liveness.active.challenge import (
    Challenge,
    ChallengeKind,
    ChallengeStep,
    ConsumeStatus,
    NonceStore,
)
from liveness.active.detectors import (
    BlinkConfig,
    BlinkDetector,
    DetectorStatus,
    FpsMonitor,
    HeadTurnDetector,
    MouthConfig,
    MouthOpenDetector,
    Outcome,
    TurnConfig,
    TurnDirection,
)
from liveness.active.geometry import N_LANDMARKS


class Decision(StrEnum):
    PASS = "PASS"  # noqa: S105
    FAIL = "FAIL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    IN_PROGRESS = "IN_PROGRESS"


class ReasonCode(StrEnum):
    OK = "OK"
    IN_PROGRESS = "IN_PROGRESS"
    CHALLENGE_EXPIRED = "CHALLENGE_EXPIRED"
    NONCE_REUSED = "NONCE_REUSED"
    NONCE_UNKNOWN = "NONCE_UNKNOWN"
    FACE_LOST_TIMEOUT = "FACE_LOST_TIMEOUT"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    TIMESTAMPS_OUT_OF_ORDER = "TIMESTAMPS_OUT_OF_ORDER"
    FPS_TOO_LOW = "FPS_TOO_LOW"
    EYES_OCCLUDED = "EYES_OCCLUDED"
    SESSION_TOO_LONG = "SESSION_TOO_LONG"
    WRONG_CHALLENGE_ORDER = "WRONG_CHALLENGE_ORDER"
    CHALLENGE_NOT_PERFORMED = "CHALLENGE_NOT_PERFORMED"


@dataclass(frozen=True)
class FrameObservation:
    timestamp_ms: float
    landmarks: NDArray[np.float64] | None
    num_faces: int
    landmark_confidence: float = 1.0

    def __post_init__(self) -> None:
        lm = self.landmarks
        if lm is not None and (
            lm.ndim != 2 or lm.shape[0] != N_LANDMARKS or lm.shape[1] not in (2, 3)
        ):
            raise ValueError(f"landmarks must be ({N_LANDMARKS}, 2|3), got {lm.shape}")


@dataclass(frozen=True)
class Result:
    decision: Decision
    reason: ReasonCode
    steps_completed: int = 0
    total_steps: int = 0
    detail: str = ""


@dataclass(frozen=True)
class SessionConfig:
    warmup_ms: float = 1000.0  # neutral period for blink baseline before step 0 starts
    step_timeout_ms: float = 8000.0
    max_duration_ms: float = 25000.0  # hard cap on session length
    max_frames: int = 3000  # hard cap on frames processed
    face_lost_timeout_ms: float = 1500.0
    multi_face_tolerance_ms: float = 200.0
    max_bad_timestamps: int = 0  # duplicate / backwards frames tolerated (dropped) before FAIL
    min_fps: float = 15.0
    fps_window: int = 30
    fps_min_intervals: int = 10
    strict_order: bool = True  # an action belonging to another step fails the session
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    mouth: MouthConfig = field(default_factory=MouthConfig)


_CONSUME_REASON = {
    ConsumeStatus.NONCE_UNKNOWN: ReasonCode.NONCE_UNKNOWN,
    ConsumeStatus.NONCE_REUSED: ReasonCode.NONCE_REUSED,
    ConsumeStatus.CHALLENGE_EXPIRED: ReasonCode.CHALLENGE_EXPIRED,
}


class ActiveLivenessSession:
    """One use per challenge: the nonce is consumed at construction.

    `mirrored` states whether the delivered frames are horizontally flipped (selfie preview).
    TURN_LEFT / TURN_RIGHT always mean the subject's own left / right.
    """

    def __init__(
        self,
        nonce: str,
        store: NonceStore,
        *,
        mirrored: bool = False,
        config: SessionConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cfg = config or SessionConfig()
        self._clock = clock
        self._final: Result | None = None
        self._steps: tuple[ChallengeStep, ...] = ()
        self._challenge: Challenge | None = None
        self._idx = 0
        self._first_t: float | None = None
        self._last_t: float | None = None
        self._frames = 0
        self._bad_ts = 0
        self._step_start: float | None = None
        self._lost_since: float | None = None
        self._multi_since: float | None = None
        self._fps = FpsMonitor(self._cfg.min_fps, self._cfg.fps_window, self._cfg.fps_min_intervals)
        self._blink = BlinkDetector(self._cfg.blink)
        self._turn_left = HeadTurnDetector(
            TurnDirection.LEFT, mirrored=mirrored, config=self._cfg.turn
        )
        self._turn_right = HeadTurnDetector(
            TurnDirection.RIGHT, mirrored=mirrored, config=self._cfg.turn
        )
        self._mouth = MouthOpenDetector(self._cfg.mouth)
        consumed = store.consume(nonce)
        if consumed.challenge is None:
            self._final = Result(Decision.FAIL, _CONSUME_REASON[consumed.status])
        else:
            self._challenge = consumed.challenge
            self._steps = consumed.challenge.sequence

    @property
    def challenge(self) -> Challenge | None:
        return self._challenge

    @property
    def result(self) -> Result:
        return self._final or self._progress()

    def _progress(self) -> Result:
        return Result(Decision.IN_PROGRESS, ReasonCode.IN_PROGRESS, self._idx, len(self._steps))

    def _end(self, decision: Decision, reason: ReasonCode, detail: str = "") -> Result:
        self._final = Result(decision, reason, self._idx, len(self._steps), detail)
        return self._final

    def _pose_detectors(self) -> tuple[HeadTurnDetector | MouthOpenDetector, ...]:
        return (self._turn_left, self._turn_right, self._mouth)

    def _reset_pose(self) -> None:
        for d in self._pose_detectors():
            d.reset()

    def poll(self) -> Result:
        """Re-check wall-clock expiry without a frame (e.g. camera stalled)."""
        if self._final is None and self._challenge and self._clock() > self._challenge.expires_at:
            return self._end(Decision.FAIL, ReasonCode.CHALLENGE_EXPIRED)
        return self.result

    def finish(self) -> Result:
        """Call when the stream ends; an unfinished challenge is not a pass."""
        if self._final is None:
            return self._unfinished()
        return self._final

    def _unfinished(self) -> Result:
        step = self._steps[self._idx]
        if step.kind is ChallengeKind.BLINK_N:
            start = self._step_start if self._step_start is not None else 0.0
            verdict = self._blink.check(step.count, start)
            if verdict.outcome is Outcome.INSUFFICIENT_EVIDENCE:
                reason = (
                    ReasonCode.FPS_TOO_LOW
                    if verdict.reason is DetectorStatus.FPS_TOO_LOW
                    else ReasonCode.EYES_OCCLUDED
                )
                return self._end(Decision.INSUFFICIENT_EVIDENCE, reason)
        return self._end(Decision.FAIL, ReasonCode.CHALLENGE_NOT_PERFORMED, step.kind.value)

    def update(self, obs: FrameObservation) -> Result:
        if self._final is not None:
            return self._final
        cfg = self._cfg
        if self._challenge is None or self._clock() > self._challenge.expires_at:
            return self._end(Decision.FAIL, ReasonCode.CHALLENGE_EXPIRED)
        t = obs.timestamp_ms
        if not math.isfinite(t) or (self._last_t is not None and t <= self._last_t):
            self._bad_ts += 1
            if self._bad_ts > cfg.max_bad_timestamps:
                return self._end(Decision.FAIL, ReasonCode.TIMESTAMPS_OUT_OF_ORDER)
            return self._progress()
        if self._first_t is None:
            self._first_t = t
        self._last_t = t
        self._frames += 1
        if self._frames > cfg.max_frames or t - self._first_t > cfg.max_duration_ms:
            return self._end(Decision.FAIL, ReasonCode.SESSION_TOO_LONG)
        self._fps.observe(t)
        if self._fps.too_low:
            return self._end(
                Decision.INSUFFICIENT_EVIDENCE, ReasonCode.FPS_TOO_LOW, f"fps={self._fps.fps:.1f}"
            )

        if obs.num_faces > 1:
            if self._multi_since is None:
                self._multi_since = t
            if t - self._multi_since >= cfg.multi_face_tolerance_ms:
                return self._end(Decision.FAIL, ReasonCode.MULTIPLE_FACES)
        else:
            self._multi_since = None
        usable = obs.num_faces == 1 and obs.landmarks is not None
        if usable:
            self._lost_since = None
        else:
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since >= cfg.face_lost_timeout_ms:
                return self._end(Decision.FAIL, ReasonCode.FACE_LOST_TIMEOUT)
        lm = obs.landmarks if usable else None
        conf = obs.landmark_confidence

        self._blink.update(t, lm, conf)
        for d in self._pose_detectors():
            d.update(t, lm, conf)
        return self._step_logic(t)

    def _step_logic(self, t: float) -> Result:
        cfg = self._cfg
        if self._step_start is None:
            if self._first_t is None or t - self._first_t < cfg.warmup_ms:
                return self._progress()
            self._step_start = t
            self._reset_pose()
        while self._idx < len(self._steps):
            step = self._steps[self._idx]
            if self._step_done(step, self._step_start):
                self._idx += 1
                self._step_start = t
                self._reset_pose()
                continue
            wrong = self._wrong_action(step)
            if wrong is not None and cfg.strict_order:
                return self._end(Decision.FAIL, ReasonCode.WRONG_CHALLENGE_ORDER, wrong)
            if t - self._step_start > cfg.step_timeout_ms:
                return self._unfinished()
            return self._progress()
        return self._end(Decision.PASS, ReasonCode.OK)

    def _step_done(self, step: ChallengeStep, start: float) -> bool:
        if step.kind is ChallengeKind.BLINK_N:
            return self._blink.check(step.count, start).outcome is Outcome.SATISFIED
        return self._detector_for(step.kind).completed

    def _detector_for(self, kind: ChallengeKind) -> HeadTurnDetector | MouthOpenDetector:
        if kind is ChallengeKind.TURN_LEFT:
            return self._turn_left
        if kind is ChallengeKind.TURN_RIGHT:
            return self._turn_right
        return self._mouth

    def _wrong_action(self, step: ChallengeStep) -> str | None:
        """Natural blinks are ignored; any other completed action must belong to this step."""
        for kind in (ChallengeKind.TURN_LEFT, ChallengeKind.TURN_RIGHT, ChallengeKind.OPEN_MOUTH):
            if kind is not step.kind and self._detector_for(kind).completed:
                return f"expected {step.kind.value}, saw {kind.value}"
        return None
