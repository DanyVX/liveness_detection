"""Runtime settings. Validated at startup; a bad value fails fast with a clear message."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIVENESS_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    seed: int = 1337
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False


def load_settings() -> Settings:
    """Load settings or exit with a readable error (no stack trace for a config typo)."""
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"LIVENESS_{'.'.join(str(p) for p in e['loc']).upper()}: {e['msg']}"
            for e in exc.errors()
        )
        sys.exit(f"Invalid configuration: {problems}. See .env.example.")
