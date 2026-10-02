"""Response models. score, decision, reason_code and model_version are present on every
response, including errors."""

from __future__ import annotations

from pydantic import BaseModel


class DecisionResponse(BaseModel):
    score: float | None
    decision: str
    reason_code: str
    model_version: str
    n_frames: int | None = None
    detail: str | None = None


class ModelInfo(BaseModel):
    model_version: str
    input_size: list[int]
    providers: list[str]
    decision_threshold: float
    clip_accept_threshold: float
    clip_reject_threshold: float
    max_batch_frames: int
    max_image_bytes: int


class Health(BaseModel):
    status: str
    model_version: str
