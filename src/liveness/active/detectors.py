"""Per-action detectors: blink, head turn, mouth open.

All durations use frame timestamps (ms), never frame counts, so the same thresholds hold at any
frame rate at or above the enforced minimum.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from liveness.active.geometry import (
    ear_from_landmarks,
    landmarks_valid,
    mar_from_landmarks,
    yaw_from_landmarks,
)

Landmarks = NDArray[np.float64]


class DetectorStatus(StrEnum):
    OK = "ok"
    CALIBRATING = "calibrating"
    FPS_TOO_LOW = "fps_too_low"
    EYES_OCCLUDED = "eyes_occluded"


class Outcome(StrEnum):
    SATISFIED = "satisfied"
    PENDING = "pending"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class Verdict:
    outcome: Outcome
    reason: DetectorStatus | None = None


class TurnDirection(StrEnum):
    LEFT = "left"  # the subject's own left
    RIGHT = "right"


class FpsMonitor:
    """Median-interval frame rate over a short window; unknown until enough intervals."""

    def __init__(self, min_fps: float, window: int = 30, min_intervals: int = 10) -> None:
        self._min_fps = min_fps
        self._min_intervals = min_intervals
        self._intervals: deque[float] = deque(maxlen=window)
        self._last: float | None = None

    def observe(self, t_ms: float) -> None:
        if self._last is not None and t_ms > self._last:
            self._intervals.append(t_ms - self._last)
        self._last = t_ms

    @property
    def fps(self) -> float | None:
        if len(self._intervals) < self._min_intervals:
            return None
        return 1000.0 / statistics.median(self._intervals)

    @property
    def too_low(self) -> bool:
        fps = self.fps
        return fps is not None and fps < self._min_fps


@dataclass(frozen=True)
class BlinkConfig:
    min_fps: float = 15.0
    fps_window: int = 30
    fps_min_intervals: int = 10
    baseline_frames: int = 12
    baseline_ear_range: tuple[float, float] = (0.12, 0.60)
    max_upper_spread: float = 0.5  # (p95 - p50) / p50 of calibration EARs; larger = implausible
    close_ratio: float = 0.70  # closed when EAR < baseline * close_ratio
    open_ratio: float = 0.85  # reopened when EAR > baseline * open_ratio (hysteresis)
    min_closed_ms: float = 50.0
    max_closed_ms: float = 1000.0  # longer is eyes held shut, not a blink
    max_gap_ms: float = 400.0  # a bigger inter-frame gap aborts an in-progress closure
    min_confidence: float = 0.5
    frame_ear_max: float = 0.8
    occlusion_report_ms: float = 600.0
    baseline_ema: float = 0.02
    max_events: int = 32


@dataclass(frozen=True)
class Blink:
    start_ms: float
    end_ms: float


class BlinkDetector:
    def __init__(self, config: BlinkConfig | None = None) -> None:
        self._cfg = config or BlinkConfig()
        self._fps = FpsMonitor(self._cfg.min_fps, self._cfg.fps_window, self._cfg.fps_min_intervals)
        self._calib: list[float] = []
        self._baseline: float | None = None
        self._baseline_bad = False
        self._closed_at: float | None = None
        self._last_t: float | None = None
        self._occluded_since: float | None = None
        self._blinks: deque[Blink] = deque(maxlen=self._cfg.max_events)
        self._count = 0

    @property
    def baseline(self) -> float | None:
        return self._baseline

    @property
    def fps(self) -> float | None:
        return self._fps.fps

    @property
    def blink_count(self) -> int:
        return self._count

    @property
    def blinks(self) -> tuple[Blink, ...]:
        return tuple(self._blinks)

    def count_since(self, t_ms: float) -> int:
        return sum(1 for b in self._blinks if b.start_ms >= t_ms)

    @property
    def status(self) -> DetectorStatus:
        if self._fps.too_low:
            return DetectorStatus.FPS_TOO_LOW
        occluded_for = (
            0.0
            if self._occluded_since is None or self._last_t is None
            else self._last_t - self._occluded_since
        )
        if self._baseline_bad or occluded_for >= self._cfg.occlusion_report_ms:
            return DetectorStatus.EYES_OCCLUDED
        if self._baseline is None:
            return DetectorStatus.CALIBRATING
        return DetectorStatus.OK

    def check(self, required: int, since_ms: float | None = None) -> Verdict:
        """Insufficient evidence (with the reason) instead of silently reporting no blinks."""
        status = self.status
        if status is DetectorStatus.FPS_TOO_LOW:
            return Verdict(Outcome.INSUFFICIENT_EVIDENCE, status)
        n = self._count if since_ms is None else self.count_since(since_ms)
        if n >= required:
            return Verdict(Outcome.SATISFIED)
        if status is DetectorStatus.EYES_OCCLUDED:
            return Verdict(Outcome.INSUFFICIENT_EVIDENCE, status)
        return Verdict(Outcome.PENDING)

    def _frame_ear(
        self, landmarks: Landmarks | None, confidence: float
    ) -> tuple[float | None, bool]:
        if landmarks is None:
            return None, False
        if not landmarks_valid(landmarks) or confidence < self._cfg.min_confidence:
            return None, True
        try:
            ear = ear_from_landmarks(landmarks)
        except ValueError:
            return None, True
        if not 0.0 <= ear <= self._cfg.frame_ear_max:
            return None, True
        return ear, False

    def _abort_closure(self) -> None:
        self._closed_at = None

    def update(self, t_ms: float, landmarks: Landmarks | None, confidence: float = 1.0) -> None:
        cfg = self._cfg
        self._fps.observe(t_ms)
        if self._last_t is not None and t_ms - self._last_t > cfg.max_gap_ms:
            self._abort_closure()
        self._last_t = t_ms
        ear, occluded = self._frame_ear(landmarks, confidence)
        if ear is None:
            self._abort_closure()
            if not occluded:
                self._occluded_since = None
            elif self._occluded_since is None:
                self._occluded_since = t_ms
            return
        self._occluded_since = None
        if self._baseline is None:
            self._calibrate(ear)
            return
        base = self._baseline
        if self._closed_at is None:
            if ear < base * cfg.close_ratio:
                self._closed_at = t_ms
            else:
                self._baseline = base + cfg.baseline_ema * (ear - base)
        elif ear > base * cfg.open_ratio:
            duration = t_ms - self._closed_at
            if cfg.min_closed_ms <= duration <= cfg.max_closed_ms:
                self._blinks.append(Blink(self._closed_at, t_ms))
                self._count += 1
            self._closed_at = None

    def _calibrate(self, ear: float) -> None:
        cfg = self._cfg
        self._calib.append(ear)
        if len(self._calib) < cfg.baseline_frames:
            return
        arr = np.asarray(self._calib)
        self._calib.clear()
        median = float(np.percentile(arr, 50))
        base = float(np.percentile(arr, 75))
        spread = (float(np.percentile(arr, 95)) - median) / max(median, 1e-9)
        lo, hi = cfg.baseline_ear_range
        if lo <= base <= hi and spread <= cfg.max_upper_spread:
            self._baseline = base
            self._baseline_bad = False
        else:
            self._baseline_bad = True


class _ArmedHold:
    """Fires once a condition holds for `hold_ms`, but only after a neutral state was seen."""

    def __init__(self, hold_ms: float) -> None:
        self._hold_ms = hold_ms
        self.reset()

    def reset(self) -> None:
        self.armed = False
        self.completed = False
        self.completed_at: float | None = None
        self._since: float | None = None

    def lost(self) -> None:
        self._since = None

    def feed(self, t_ms: float, arm_ok: bool, fire_ok: bool) -> None:
        if self.completed:
            return
        if not self.armed:
            self.armed = arm_ok
            return
        if not fire_ok:
            self._since = None
            return
        if self._since is None:
            self._since = t_ms
        if t_ms - self._since >= self._hold_ms:
            self.completed = True
            self.completed_at = t_ms


@dataclass(frozen=True)
class TurnConfig:
    turn_threshold: float = 0.35
    neutral_threshold: float = 0.15
    hold_ms: float = 150.0
    min_confidence: float = 0.5


class HeadTurnDetector:
    """Direction is the subject's own left/right; `mirrored` says the image is flipped."""

    def __init__(
        self, direction: TurnDirection, *, mirrored: bool, config: TurnConfig | None = None
    ) -> None:
        self._cfg = config or TurnConfig()
        self._sign = 1.0 if direction is TurnDirection.LEFT else -1.0
        self._mirrored = mirrored
        self._hold = _ArmedHold(self._cfg.hold_ms)

    @property
    def completed(self) -> bool:
        return self._hold.completed

    @property
    def completed_at(self) -> float | None:
        return self._hold.completed_at

    def reset(self) -> None:
        self._hold.reset()

    def update(self, t_ms: float, landmarks: Landmarks | None, confidence: float = 1.0) -> bool:
        if (
            landmarks is None
            or not landmarks_valid(landmarks)
            or confidence < self._cfg.min_confidence
        ):
            self._hold.lost()
            return self.completed
        try:
            yaw = yaw_from_landmarks(landmarks, mirrored=self._mirrored)
        except ValueError:
            self._hold.lost()
            return self.completed
        self._hold.feed(
            t_ms,
            arm_ok=abs(yaw) < self._cfg.neutral_threshold,
            fire_ok=yaw * self._sign >= self._cfg.turn_threshold,
        )
        return self.completed


@dataclass(frozen=True)
class MouthConfig:
    open_threshold: float = 0.5
    closed_threshold: float = 0.3
    hold_ms: float = 100.0
    min_confidence: float = 0.5


class MouthOpenDetector:
    def __init__(self, config: MouthConfig | None = None) -> None:
        self._cfg = config or MouthConfig()
        self._hold = _ArmedHold(self._cfg.hold_ms)

    @property
    def completed(self) -> bool:
        return self._hold.completed

    @property
    def completed_at(self) -> float | None:
        return self._hold.completed_at

    def reset(self) -> None:
        self._hold.reset()

    def update(self, t_ms: float, landmarks: Landmarks | None, confidence: float = 1.0) -> bool:
        if (
            landmarks is None
            or not landmarks_valid(landmarks)
            or confidence < self._cfg.min_confidence
        ):
            self._hold.lost()
            return self.completed
        try:
            mar = mar_from_landmarks(landmarks)
        except ValueError:
            self._hold.lost()
            return self.completed
        self._hold.feed(
            t_ms,
            arm_ok=mar <= self._cfg.closed_threshold,
            fire_ok=mar >= self._cfg.open_threshold,
        )
        return self.completed
