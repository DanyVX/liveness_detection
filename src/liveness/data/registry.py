"""Dataset loaders by name.

Real loaders are intentionally stubs until access is granted and the on-disk layout has
been checked against DATA.md. They fail loudly rather than guessing a layout.
"""

from __future__ import annotations

from pathlib import Path

from liveness.data.base import Sample
from liveness.data.synthetic import DATASET_NAME as SYNTHETIC
from liveness.data.synthetic import load_synthetic

REAL_DATASETS = ("replay_attack", "casia_fasd", "oulu_npu")


class DatasetNotFoundError(FileNotFoundError):
    """The dataset folder is missing; the message says where to get it."""


def load_dataset(name: str, root: Path) -> list[Sample]:
    if name == SYNTHETIC:
        return load_synthetic(root)
    if name in REAL_DATASETS:
        if not root.is_dir():
            raise DatasetNotFoundError(
                f"{name}: {root} not found. Request access and follow DATA.md; "
                "datasets are never downloaded or committed automatically."
            )
        raise NotImplementedError(
            f"{name}: loader not implemented yet (needs the real data to verify the layout)."
        )
    raise KeyError(f"unknown dataset {name!r}; known: {[SYNTHETIC, *REAL_DATASETS]}")
