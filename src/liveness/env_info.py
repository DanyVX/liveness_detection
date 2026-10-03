"""Environment fingerprint embedded in every results file."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_LIBS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "opencv": "opencv-python-headless",
    "onnxruntime": "onnxruntime",
    "pillow": "pillow",
}


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ram_bytes() -> int | None:
    try:
        sysconf = os.sysconf  # type: ignore[attr-defined]
        return int(sysconf("SC_PAGE_SIZE")) * int(sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _gpu() -> str:
    """Best-effort GPU name without importing torch; "none" when nothing is found."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "none"
    names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
    return "; ".join(names) if names else "none"


def collect_environment(repo_dir: Path | None = None) -> dict[str, object]:
    cwd = repo_dir or Path(__file__).resolve().parent
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": _ram_bytes(),
        "gpu": _gpu(),
        "libraries": {name: _version(dist) for name, dist in _LIBS.items()},
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
