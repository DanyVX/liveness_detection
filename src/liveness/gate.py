"""Integration surface: a Protocol other repos can type against without importing this package's
internals, plus an ONNX-backed implementation. Nothing is persisted; inputs live only in memory."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from liveness.fusion import Aggregation, Decision, TemporalFuser
from liveness.infer import (
    FrameScorer,
    InvalidImageError,
    ReasonCode,
    Verdict,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    score: float | None
    decision: str
    reason_code: str
    model_version: str


@runtime_checkable
class LivenessGate(Protocol):
    def check_image(self, data: bytes) -> GateResult: ...

    def check_clip(self, frames: list[bytes]) -> GateResult: ...


@dataclass(frozen=True)
class GateConfig:
    decision_threshold: float
    clip_accept_threshold: float
    clip_reject_threshold: float
    clip_min_valid_frames: int
    clip_aggregation: Aggregation = "mean"
    max_clip_frames: int = 32


class OnnxLivenessGate:
    """Stateless per call: each clip gets a fresh TemporalFuser."""

    def __init__(self, frame_scorer: FrameScorer, config: GateConfig) -> None:
        self._fs = frame_scorer
        self._cfg = config
        # Validates the clip config at construction time.
        self._new_fuser(max(config.max_clip_frames, config.clip_min_valid_frames))

    @property
    def model_version(self) -> str:
        return self._fs.scorer.model_version

    def _result(self, score: float | None, decision: Verdict, reason: ReasonCode) -> GateResult:
        return GateResult(score, decision.value, reason.value, self.model_version)

    def _new_fuser(self, window: int) -> TemporalFuser:
        c = self._cfg
        return TemporalFuser(
            accept_threshold=c.clip_accept_threshold,
            reject_threshold=c.clip_reject_threshold,
            window=window,
            min_valid_frames=c.clip_min_valid_frames,
            aggregation=c.clip_aggregation,
        )

    def check_image(self, data: bytes) -> GateResult:
        try:
            fr = self._fs.analyze(data)
        except InvalidImageError:
            return self._result(None, Verdict.ERROR, ReasonCode.INVALID_INPUT)
        if fr.score is None:
            return self._result(None, Verdict.INSUFFICIENT_QUALITY, fr.reason)
        if fr.score >= self._cfg.decision_threshold:
            return self._result(fr.score, Verdict.LIVE, ReasonCode.SCORE_ABOVE_THRESHOLD)
        return self._result(fr.score, Verdict.SPOOF, ReasonCode.SCORE_BELOW_THRESHOLD)

    def check_clip(self, frames: Sequence[bytes]) -> GateResult:
        if not frames or len(frames) > self._cfg.max_clip_frames:
            return self._result(None, Verdict.ERROR, ReasonCode.INVALID_INPUT)
        scores: list[float | None] = []
        for data in frames:
            try:
                scores.append(self._fs.analyze(data).score)
            except InvalidImageError:
                return self._result(None, Verdict.ERROR, ReasonCode.INVALID_INPUT)
        fused = self._new_fuser(max(len(frames), self._cfg.clip_min_valid_frames)).fuse(scores)
        if fused.decision is Decision.LIVE:
            return self._result(fused.aggregate, Verdict.LIVE, ReasonCode.SCORE_ABOVE_THRESHOLD)
        if fused.decision is Decision.SPOOF:
            return self._result(fused.aggregate, Verdict.SPOOF, ReasonCode.SCORE_BELOW_THRESHOLD)
        return self._result(
            fused.aggregate, Verdict.INSUFFICIENT_EVIDENCE, ReasonCode.INSUFFICIENT_EVIDENCE
        )
