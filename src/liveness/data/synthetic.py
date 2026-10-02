"""Tiny synthetic fixture so CI and unit tests run without any real dataset.

The images are cartoon "faces" with generic image degradations (blur, colour shift,
interference pattern). They exist only to exercise the pipeline; models trained on them
say nothing about real-world liveness, and no result from them may appear in RESULTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFilter

from liveness.data.base import AttackType, Label, Sample

DATASET_NAME = "synthetic"
_Array = NDArray[np.float64]


def _face(rng: np.random.Generator, size: int) -> _Array:
    yy, xx = np.mgrid[0:size, 0:size] / size
    background = rng.uniform(0.2, 0.8, size=3)
    img = np.broadcast_to(background, (size, size, 3)).copy()
    img += 0.15 * (xx + yy)[..., None] * rng.uniform(-1, 1, size=3)
    skin = rng.uniform(0.35, 0.85, size=3)
    face = ((xx - 0.5) / 0.32) ** 2 + ((yy - 0.5) / 0.42) ** 2 < 1
    img[face] = skin
    for ex in (0.4, 0.6):
        eye = ((xx - ex) / 0.04) ** 2 + ((yy - 0.42) / 0.025) ** 2 < 1
        img[eye] = 0.1
    mouth = ((xx - 0.5) / 0.1) ** 2 + ((yy - 0.66) / 0.02) ** 2 < 1
    img[mouth] = skin * 0.5
    img += rng.normal(0, 0.01, img.shape)
    return np.clip(img, 0, 1)


def _to_image(arr: _Array) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def render_bona_fide(rng: np.random.Generator, size: int = 64) -> Image.Image:
    return _to_image(_face(rng, size))


def render_print(rng: np.random.Generator, size: int = 64) -> Image.Image:
    """Blur, washed-out contrast and a colour cast, like a re-captured paper print."""
    img = _to_image(_face(rng, size)).filter(ImageFilter.GaussianBlur(radius=1.2))
    arr = np.asarray(img, dtype=np.float64) / 255
    arr = 0.75 * arr + 0.15 + np.array([0.05, 0.0, -0.05])
    return _to_image(arr)


def render_replay(rng: np.random.Generator, size: int = 64) -> Image.Image:
    """Two slightly rotated gratings (moire-like) and a cool tint, like a screen re-capture."""
    arr = _face(rng, size)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    period = rng.uniform(3.0, 4.5)
    angle = rng.uniform(0.05, 0.2)
    g1 = np.sin(2 * np.pi * xx / period)
    g2 = np.sin(2 * np.pi * (xx * np.cos(angle) + yy * np.sin(angle)) / (period * 1.05))
    arr = arr + 0.06 * (g1 * g2)[..., None] + np.array([-0.03, 0.0, 0.06])
    return _to_image(arr)


def generate_synthetic_dataset(
    root: Path, n_subjects: int = 12, seed: int = 0, size: int = 64
) -> list[Sample]:
    """Write PNGs plus manifest.json under `root`; return the samples. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    plan = [
        (Label.BONA_FIDE, AttackType.NONE, "bf0", render_bona_fide),
        (Label.BONA_FIDE, AttackType.NONE, "bf1", render_bona_fide),
        (Label.ATTACK, AttackType.PRINT, "print0", render_print),
        (Label.ATTACK, AttackType.REPLAY, "replay0", render_replay),
    ]
    samples: list[Sample] = []
    for i in range(n_subjects):
        subject = f"s{i:03d}"
        for label, attack, tag, render in plan:
            path = root / f"{subject}_{tag}.png"
            render(rng, size).save(path)
            samples.append(Sample(DATASET_NAME, subject, path, label, attack))
    index = [
        {
            "subject_id": s.subject_id,
            "file": s.path.name,
            "label": s.label.name,
            "attack_type": s.attack_type.value,
        }
        for s in samples
    ]
    (root / "manifest.json").write_text(json.dumps(index, indent=2) + "\n")
    return samples


def load_synthetic(root: Path) -> list[Sample]:
    index = json.loads((root / "manifest.json").read_text())
    return [
        Sample(
            DATASET_NAME,
            row["subject_id"],
            root / row["file"],
            Label[row["label"]],
            AttackType(row["attack_type"]),
        )
        for row in index
    ]
