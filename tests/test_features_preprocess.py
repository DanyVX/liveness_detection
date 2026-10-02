import logging
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from liveness.features import preprocess as pp
from liveness.features.preprocess import (
    crop_face,
    crop_with_margin,
    detect_face,
    expand_box,
    load_image,
    normalize_resolution,
    to_gray,
)


def _rgb(h: int = 40, w: int = 50) -> pp.ImageU8:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_load_image_modes(tmp_path: Path) -> None:
    rgb = _rgb()
    Image.fromarray(rgb[..., 0]).save(tmp_path / "g.png")
    Image.fromarray(np.dstack([rgb, rgb[..., :1]]), "RGBA").save(tmp_path / "a.png")
    Image.fromarray(rgb).convert("P").save(tmp_path / "p.png")
    gray = load_image(tmp_path / "g.png")
    assert gray.shape == (40, 50, 3) and gray.dtype == np.uint8
    assert np.array_equal(gray[..., 0], gray[..., 2])
    for name in ("a.png", "p.png"):
        assert load_image(tmp_path / name).shape == (40, 50, 3)


def test_to_gray_shapes_and_error() -> None:
    assert to_gray(_rgb()).shape == (40, 50)
    assert to_gray(_rgb()[..., 0]).shape == (40, 50)
    with pytest.raises(ValueError, match="expected"):
        to_gray(np.zeros((4, 4, 2), dtype=np.uint8))


def test_normalize_resolution_fixed_size_both_directions() -> None:
    assert normalize_resolution(_rgb(200, 100), 64).shape == (64, 64, 3)
    assert normalize_resolution(_rgb(20, 30), 64).shape == (64, 64, 3)
    with pytest.raises(ValueError, match="size"):
        normalize_resolution(_rgb(), 2)
    with pytest.raises(ValueError, match="empty"):
        normalize_resolution(np.zeros((0, 5, 3), dtype=np.uint8), 16)


def test_expand_box_errors() -> None:
    with pytest.raises(ValueError, match="margin"):
        expand_box((0, 0, 5, 5), -0.1, (10, 10, 3))
    with pytest.raises(ValueError, match="positive"):
        expand_box((0, 0, 0, 5), 0.1, (10, 10, 3))
    with pytest.raises(ValueError, match="outside"):
        expand_box((50, 50, 5, 5), 0.0, (10, 10, 3))


@given(
    h=st.integers(1, 80),
    w=st.integers(1, 80),
    x=st.integers(-20, 100),
    y=st.integers(-20, 100),
    bw=st.integers(1, 120),
    bh=st.integers(1, 120),
    margin=st.floats(0, 2, allow_nan=False),
)
def test_crop_with_margin_inside_bounds_and_nonempty(
    h: int, w: int, x: int, y: int, bw: int, bh: int, margin: float
) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    try:
        ex, ey, ew, eh = expand_box((x, y, bw, bh), margin, img.shape)
    except ValueError:
        return  # box entirely outside the image is rejected, not silently emptied
    crop = crop_with_margin(img, (x, y, bw, bh), margin)
    assert crop.size > 0
    assert ex >= 0 and ey >= 0 and ex + ew <= w and ey + eh <= h
    assert crop.shape[:2] == (eh, ew)


@given(
    h=st.integers(8, 60),
    w=st.integers(8, 60),
    data=st.data(),
    margin=st.floats(0, 1, allow_nan=False),
)
def test_crop_with_margin_always_succeeds_for_inside_boxes(
    h: int, w: int, data: st.DataObject, margin: float
) -> None:
    x = data.draw(st.integers(0, w - 1))
    y = data.draw(st.integers(0, h - 1))
    bw = data.draw(st.integers(1, w - x))
    bh = data.draw(st.integers(1, h - y))
    crop = crop_with_margin(np.zeros((h, w, 3), dtype=np.uint8), (x, y, bw, bh), margin)
    assert crop.shape[0] >= bh and crop.shape[1] >= bw
    assert crop.shape[0] <= h and crop.shape[1] <= w


def test_crop_face_fallback_is_logged_and_flagged(caplog: pytest.LogCaptureFixture) -> None:
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    with caplog.at_level(logging.WARNING, logger="liveness.features.preprocess"):
        result = crop_face(flat, margin=0.1, source="flat.png")
    assert result.used_fallback
    assert any(
        "falling back to centre crop" in r.message and "flat.png" in r.message
        for r in caplog.records
    )
    assert result.image.size > 0


def test_crop_face_error_mode() -> None:
    with pytest.raises(LookupError, match="no face"):
        crop_face(np.full((64, 64, 3), 128, dtype=np.uint8), fallback="error")


class _FakeCascade:
    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self.boxes = boxes

    def empty(self) -> bool:
        return False

    def detectMultiScale(self, *_: object, **__: object) -> np.ndarray:
        return np.array(self.boxes, dtype=np.int32).reshape(-1, 4)


def test_detect_face_picks_largest_and_crops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "_cascade", lambda: _FakeCascade([(1, 1, 10, 10), (5, 5, 20, 20)]))
    img = _rgb(64, 64)
    assert detect_face(img) == (5, 5, 20, 20)
    result = crop_face(img, margin=0.0)
    assert not result.used_fallback and result.image.shape[:2] == (20, 20)


def test_detect_face_no_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "_cascade", lambda: _FakeCascade([]))
    assert detect_face(_rgb()) is None


def test_cascade_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    pp._cascade.cache_clear()
    monkeypatch.delattr(pp.cv2, "CascadeClassifier", raising=False)
    try:
        assert pp._cascade() is None
        assert detect_face(_rgb()) is None
    finally:
        pp._cascade.cache_clear()


def test_cascade_loads_when_available() -> None:
    pp._cascade.cache_clear()
    if not hasattr(pp.cv2, "CascadeClassifier"):
        pytest.skip("this OpenCV build ships no Haar cascade")
    assert pp._cascade() is not None  # pragma: no cover
    pp._cascade.cache_clear()  # pragma: no cover
