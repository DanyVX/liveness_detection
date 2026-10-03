"""Dataset loaders by name, backed by audited normalized manifests for licensed data."""

from __future__ import annotations

from pathlib import Path

from liveness.data.base import Sample
from liveness.data.manifest import ManifestError, load_manifest_dataset
from liveness.data.synthetic import DATASET_NAME as SYNTHETIC
from liveness.data.synthetic import DATASET_NAME_B as SYNTHETIC_B
from liveness.data.synthetic import load_synthetic

REAL_DATASETS = ("replay_attack", "casia_fasd", "oulu_npu")


class DatasetNotFoundError(FileNotFoundError):
    """The dataset folder is missing; the message says where to get it."""


def load_dataset(name: str, root: Path) -> list[Sample]:
    if name in (SYNTHETIC, SYNTHETIC_B):
        return load_synthetic(root)
    if name in REAL_DATASETS:
        if not root.is_dir():
            raise DatasetNotFoundError(
                f"{name}: {root} not found. Request access and follow DATA.md; "
                "datasets are never downloaded or committed automatically."
            )
        try:
            return list(load_manifest_dataset(name, root).samples)
        except ManifestError as exc:
            raise ManifestError(f"{name}: {exc}") from exc
    raise KeyError(f"unknown dataset {name!r}; known: {[SYNTHETIC, SYNTHETIC_B, *REAL_DATASETS]}")
