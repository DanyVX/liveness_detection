import math
from pathlib import Path

import numpy as np
from PIL import Image

from liveness.data import AttackType, Label, Sample
from liveness.data.synthetic import generate_synthetic_dataset, render_bona_fide
from liveness.features.jpeg_bias import (
    blockiness,
    cohens_d,
    estimate_jpeg_quality,
    measure_jpeg_bias,
)


def _jpeg_set(root: Path, q_bona: int, q_attack: int, n: int = 10) -> list[Sample]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    out: list[Sample] = []
    for i in range(n):
        for label, attack, q, tag in (
            (Label.BONA_FIDE, AttackType.NONE, q_bona, "bf"),
            (Label.ATTACK, AttackType.PRINT, q_attack, "at"),
        ):
            p = root / f"s{i}_{tag}.jpg"
            render_bona_fide(rng, 64).save(p, quality=q)
            out.append(Sample("d", f"s{i}", p, label, attack))
    return out


def test_flags_constructed_bias(tmp_path: Path) -> None:
    report = measure_jpeg_bias(_jpeg_set(tmp_path, 95, 30))
    assert "jpeg_quality" in report["flagged"]
    assert "bytes_per_pixel" in report["flagged"]
    assert report["warnings"]
    q = report["classes"]
    assert q["bona_fide"]["jpeg_quality"]["mean"] > q["attack"]["jpeg_quality"]["mean"]
    assert abs(q["bona_fide"]["jpeg_quality"]["mean"] - 95) < 3
    assert abs(q["attack"]["jpeg_quality"]["mean"] - 30) < 3


def test_unbiased_set_not_flagged(tmp_path: Path) -> None:
    report = measure_jpeg_bias(_jpeg_set(tmp_path, 85, 85))
    assert report["flagged"] == []
    assert report["warnings"] == []


def test_png_marks_quality_not_available(tmp_path: Path) -> None:
    samples = generate_synthetic_dataset(tmp_path, n_subjects=4, seed=0)
    report = measure_jpeg_bias(samples)
    assert report["classes"]["bona_fide"]["jpeg_quality"] == {"mean": None, "n": 0}
    assert report["cohens_d"]["jpeg_quality"] is None
    assert report["classes"]["attack"]["width"]["mean"] == 64.0


def test_cohens_d_edge_cases() -> None:
    assert cohens_d([1.0], [2.0, 3.0]) is None
    assert cohens_d([1.0, 1.0], [1.0, 1.0]) == 0.0
    assert cohens_d([2.0, 2.0], [1.0, 1.0]) == math.inf
    d = cohens_d([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert d is not None and abs(d + 1.0) < 1e-9


def test_blockiness_detects_block_edges() -> None:
    rng = np.random.default_rng(0)
    blocks = np.kron(rng.integers(0, 255, (8, 8)), np.ones((8, 8)))
    assert blockiness(blocks) > 5
    assert blockiness(np.zeros((8, 8))) != blockiness(np.zeros((8, 8)))  # NaN: too small


def test_quality_none_for_png(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(p)
    with Image.open(p) as im:
        assert estimate_jpeg_quality(im) is None
