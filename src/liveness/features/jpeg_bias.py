"""Dataset-bias detector: do bona fide and attack files differ in compression or size?

If attacks were saved at a different JPEG quality, resolution or bytes-per-pixel than
bona fide images, a classifier can score well by reading those artefacts instead of
liveness cues. This reports per-class means and Cohen's d for each such property.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from liveness.data.base import Label, Sample
from liveness.features.preprocess import to_gray

# Standard libjpeg luminance quantisation table (quality 50); used to estimate quality.
_STD_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)  # fmt: skip
DEFAULT_THRESHOLD = 0.8  # |Cohen's d| conventionally read as "large"
METRICS = ("blockiness", "jpeg_quality", "bytes_per_pixel", "width", "height")


def blockiness(gray: np.ndarray[Any, Any]) -> float:
    """Mean absolute step across 8-pixel block edges over the step elsewhere (1.0 = none)."""
    g = gray.astype(np.float64)
    h, w = g.shape
    if h < 16 or w < 16:
        return float("nan")
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    on_x = np.arange(dx.shape[1]) % 8 == 7
    on_y = np.arange(dy.shape[0]) % 8 == 7
    edge = np.concatenate([dx[:, on_x].ravel(), dy[on_y, :].ravel()])
    rest = np.concatenate([dx[:, ~on_x].ravel(), dy[~on_y, :].ravel()])
    return float(edge.mean() / (rest.mean() + 1e-9))


def estimate_jpeg_quality(im: Image.Image) -> float | None:
    """Quality estimate from the luminance quantisation table; None if not a JPEG.

    Inverts the libjpeg scaling of the standard table. Approximate: encoders with custom
    tables will not map exactly, and values are clamped by table entries >= 1.
    """
    tables = getattr(im, "quantization", None)
    if not tables or 0 not in tables:
        return None
    scale = 100.0 * sum(tables[0]) / sum(_STD_LUMA)
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return float(min(100.0, max(1.0, quality)))


def _measure_one(sample: Sample) -> dict[str, float | None]:
    with Image.open(sample.path) as im:
        width, height = im.size
        quality = estimate_jpeg_quality(im)
        gray = to_gray(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return {
        "blockiness": blockiness(gray),
        "jpeg_quality": quality,
        "bytes_per_pixel": sample.path.stat().st_size / (width * height),
        "width": float(width),
        "height": float(height),
    }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Cohen's d (bona fide minus attack, pooled SD); None if undefined (n < 2 per group).

    Returns +/-inf when both groups have zero variance but different means.
    """
    if len(a) < 2 or len(b) < 2:
        return None
    xa, xb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    pooled = math.sqrt(
        ((len(xa) - 1) * xa.var(ddof=1) + (len(xb) - 1) * xb.var(ddof=1)) / (len(xa) + len(xb) - 2)
    )
    diff = float(xa.mean() - xb.mean())
    if pooled == 0.0:
        return 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
    return diff / pooled


def measure_jpeg_bias(
    samples: Sequence[Sample], threshold: float = DEFAULT_THRESHOLD
) -> dict[str, Any]:
    """Per-class means and effect sizes of compression/size properties.

    jpeg_quality is None ("n/a") for classes with no JPEG files (e.g. PNG). A metric is
    flagged, and a warning emitted, when |d| >= threshold.
    """
    values: dict[Label, dict[str, list[float]]] = {
        label: {m: [] for m in METRICS} for label in Label
    }
    for s in samples:
        for metric, value in _measure_one(s).items():
            if value is not None and not math.isnan(value):
                values[s.label][metric].append(value)

    def cls(label: Label) -> dict[str, Any]:
        out: dict[str, Any] = {"n_samples": sum(1 for s in samples if s.label is label)}
        for m in METRICS:
            v = values[label][m]
            out[m] = {"mean": float(np.mean(v)) if v else None, "n": len(v)}
        return out

    effect: dict[str, float | None] = {}
    warnings: list[str] = []
    for m in METRICS:
        d = cohens_d(values[Label.BONA_FIDE][m], values[Label.ATTACK][m])
        effect[m] = d
        if d is not None and abs(d) >= threshold:
            warnings.append(
                f"{m}: bona fide and attack classes differ strongly (Cohen's d={d:.2f}); "
                "a classifier may exploit this instead of liveness cues"
            )
    return {
        "classes": {"bona_fide": cls(Label.BONA_FIDE), "attack": cls(Label.ATTACK)},
        "cohens_d": effect,
        "threshold": threshold,
        "flagged": [w.split(":")[0] for w in warnings],
        "warnings": warnings,
    }
