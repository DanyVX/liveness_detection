"""API settings (env prefix LIVENESS_API_). Thresholds are configuration, never code."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from liveness.fusion import Aggregation


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIVENESS_API_", extra="ignore")

    model_path: Path
    providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    decision_threshold: float = Field(gt=0.0, lt=1.0)
    clip_accept_threshold: float = Field(gt=0.0, lt=1.0)
    clip_reject_threshold: float = Field(gt=0.0, lt=1.0)
    clip_min_valid_frames: int = Field(default=3, ge=1)
    clip_aggregation: Aggregation = "mean"
    max_batch_frames: int = Field(default=16, ge=1, le=256)
    max_image_bytes: int = Field(default=2_000_000, ge=1)
    min_face_px: int = Field(default=64, ge=1)
    margin: float = Field(default=0.2, ge=0.0, le=2.0)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_clip_thresholds(self) -> Self:
        if self.clip_reject_threshold > self.clip_accept_threshold:
            raise ValueError("clip_reject_threshold must be <= clip_accept_threshold")
        return self
