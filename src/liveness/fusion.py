"""Temporal fusion of per-frame liveness scores (higher = more bona fide).

A bounded sliding window feeds an aggregate (mean or median) into a two-threshold hysteresis
state machine. Too little valid evidence yields INSUFFICIENT_EVIDENCE, never a forced decision.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

Aggregation = Literal["mean", "median"]


class Decision(StrEnum):
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class FusionResult:
    decision: Decision
    aggregate: float | None
    n_valid: int
    n_frames: int


class TemporalFuser:
    """Sliding-window fuser with hysteresis.

    The aggregate must reach ``accept_threshold`` to enter LIVE and fall to
    ``reject_threshold`` or below to enter SPOOF. In between, the previous decision persists
    (or INSUFFICIENT_EVIDENCE if none yet), so noise around one threshold cannot flap the output.
    """

    def __init__(
        self,
        *,
        accept_threshold: float,
        reject_threshold: float,
        window: int,
        min_valid_frames: int,
        aggregation: Aggregation = "mean",
    ) -> None:
        for name, value in (
            ("accept_threshold", accept_threshold),
            ("reject_threshold", reject_threshold),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if reject_threshold > accept_threshold:
            raise ValueError("reject_threshold must be <= accept_threshold")
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= min_valid_frames <= window:
            raise ValueError("min_valid_frames must be in [1, window]")
        if aggregation not in ("mean", "median"):
            raise ValueError(f"unknown aggregation: {aggregation!r}")
        self._accept = accept_threshold
        self._reject = reject_threshold
        self._min_valid = min_valid_frames
        self._aggregation: Aggregation = aggregation
        self._window: deque[float | None] = deque(maxlen=window)
        self._state = Decision.INSUFFICIENT_EVIDENCE

    @property
    def state(self) -> Decision:
        return self._state

    @property
    def window_size(self) -> int:
        return len(self._window)

    def reset(self) -> None:
        self._window.clear()
        self._state = Decision.INSUFFICIENT_EVIDENCE

    def update(self, score: float | None) -> FusionResult:
        """Add one frame. ``None`` marks a frame without a usable face (or a dropped frame)."""
        if score is not None:
            if isinstance(score, bool) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"score must be finite and within [0, 1], got {score!r}")
            score = float(score)
        self._window.append(score)
        valid = [s for s in self._window if s is not None]
        if len(valid) < self._min_valid:
            self._state = Decision.INSUFFICIENT_EVIDENCE
            return FusionResult(self._state, None, len(valid), len(self._window))
        agg = statistics.fmean(valid) if self._aggregation == "mean" else statistics.median(valid)
        if agg >= self._accept:
            self._state = Decision.LIVE
        elif agg <= self._reject:
            self._state = Decision.SPOOF
        return FusionResult(self._state, agg, len(valid), len(self._window))

    def fuse(self, scores: Iterable[float | None]) -> FusionResult:
        """Fuse a whole clip from a clean state."""
        self.reset()
        result = FusionResult(Decision.INSUFFICIENT_EVIDENCE, None, 0, 0)
        for s in scores:
            result = self.update(s)
        return result
