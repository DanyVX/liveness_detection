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
DATASET_NAME_B = "synthetic_b"
DOMAINS = {"A": DATASET_NAME, "B": DATASET_NAME_B}
_Array = NDArray[np.float64]
_B_BG = (0.0, 0.45)
_B_NOISE = 0.03
_A_REPLAY_TINT = np.array([-0.03, 0.0, 0.06])


def _face(
    rng: np.random.Generator,
    size: int,
    bg_range: tuple[float, float] = (0.2, 0.8),
    noise: float = 0.01,
) -> _Array:
    yy, xx = np.mgrid[0:size, 0:size] / size
    background = rng.uniform(bg_range[0], bg_range[1], size=3)
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
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1)


def _to_image(arr: _Array) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def render_bona_fide(rng: np.random.Generator, size: int = 64, domain: str = "A") -> Image.Image:
    if domain == "B":
        return _to_image(_face(rng, size, _B_BG, _B_NOISE))
    return _to_image(_face(rng, size))


def render_print(rng: np.random.Generator, size: int = 64, domain: str = "A") -> Image.Image:
    """Blur, washed-out contrast and a colour cast, like a re-captured paper print."""
    if domain == "B":
        base = _to_image(_face(rng, size, _B_BG, _B_NOISE))
        img = base.filter(ImageFilter.GaussianBlur(radius=0.6))
        gain, offset, cast = 0.9, 0.05, np.array([-0.04, 0.03, 0.05])
    else:
        img = _to_image(_face(rng, size)).filter(ImageFilter.GaussianBlur(radius=1.2))
        gain, offset, cast = 0.75, 0.15, np.array([0.05, 0.0, -0.05])
    arr = np.asarray(img, dtype=np.float64) / 255
    arr = gain * arr + offset + cast
    return _to_image(arr)


def render_replay(rng: np.random.Generator, size: int = 64, domain: str = "A") -> Image.Image:
    """Two slightly rotated gratings (moire-like) and a cool tint, like a screen re-capture."""
    b = domain == "B"
    arr = _face(rng, size, _B_BG, _B_NOISE) if b else _face(rng, size)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    period = rng.uniform(5.0, 7.0) if b else rng.uniform(3.0, 4.5)
    angle = rng.uniform(0.05, 0.2)
    g1 = np.sin(2 * np.pi * xx / period)
    g2 = np.sin(2 * np.pi * (xx * np.cos(angle) + yy * np.sin(angle)) / (period * 1.05))
    strength, tint = (0.03, np.array([0.05, 0.03, -0.02])) if b else (0.06, _A_REPLAY_TINT)
    arr = arr + strength * (g1 * g2)[..., None] + tint
    return _to_image(arr)


def generate_synthetic_dataset(
    root: Path,
    n_subjects: int = 12,
    seed: int = 0,
    size: int | None = None,
    domain: str = "A",
) -> list[Sample]:
    """Write PNGs plus manifest.json under `root`; return the samples. Deterministic per seed.

    Domain "A" is the original fixture (default size 64). Domain "B" shifts the background
    distribution, noise level, attack artifact strengths and colour casts (default size 80),
    giving a systematic covariate shift for cross-domain smoke tests.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown synthetic domain {domain!r}; known: {sorted(DOMAINS)}")
    if size is None:
        size = 80 if domain == "B" else 64
    dataset = DOMAINS[domain]
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
            render(rng, size, domain).save(path)
            samples.append(Sample(dataset, subject, path, label, attack))
    index = [
        {
            "subject_id": s.subject_id,
            "file": s.path.name,
            "label": s.label.name,
            "attack_type": s.attack_type.value,
        }
        for s in samples
    ]
    manifest = {"domain": domain, "samples": index}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return samples


def load_synthetic(root: Path) -> list[Sample]:
    raw = json.loads((root / "manifest.json").read_text())
    # Old manifests are a bare list of rows (implicitly domain A).
    domain, index = (
        ("A", raw) if isinstance(raw, list) else (raw.get("domain", "A"), raw["samples"])
    )
    dataset = DOMAINS[domain]
    return [
        Sample(
            dataset,
            row["subject_id"],
            root / row["file"],
            Label[row["label"]],
            AttackType(row["attack_type"]),
        )
        for row in index
    ]
