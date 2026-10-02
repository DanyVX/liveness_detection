"""Image loading, resolution normalisation and face cropping.

All hand-crafted features are computed on a fixed-size image so that the native
resolution of a dataset cannot become a shortcut for the classifier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)

ImageU8 = NDArray[np.uint8]
Box = tuple[int, int, int, int]  # x, y, w, h in pixels

_FALLBACK_SIDE_FRACTION = 0.6


def load_image(path: Path | str) -> ImageU8:
    """Read any PIL-readable image as an (H, W, 3) uint8 RGB array.

    Grayscale is replicated to three channels; RGBA and palette images are converted
    (alpha is dropped, not composited).
    """
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def to_gray(img: NDArray[np.generic]) -> NDArray[np.float32]:
    """Luma image in [0, 255] as float32; accepts (H, W) or (H, W, 3) input."""
    if img.ndim == 2:
        return img.astype(np.float32)
    if img.ndim == 3 and img.shape[2] == 3:
        return np.asarray(
            cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY), dtype=np.float32
        )
    raise ValueError(f"expected (H, W) or (H, W, 3) image, got shape {img.shape}")


def normalize_resolution(img: ImageU8, size: int) -> ImageU8:
    """Resize to a square `size` x `size` image (area interpolation when shrinking)."""
    if size < 8:
        raise ValueError("size must be >= 8")
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty image")
    shrinking = size <= min(h, w)
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return np.asarray(cv2.resize(img, (size, size), interpolation=interp), dtype=np.uint8)


def expand_box(bbox: Box, margin: float, shape: tuple[int, ...]) -> Box:
    """Grow `bbox` by `margin` * (w, h) on every side and clip to the image."""
    if margin < 0:
        raise ValueError("margin must be >= 0")
    img_h, img_w = shape[:2]
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        raise ValueError("bbox must have positive width and height")
    x0 = max(0, int(np.floor(x - margin * w)))
    y0 = max(0, int(np.floor(y - margin * h)))
    x1 = min(img_w, int(np.ceil(x + w + margin * w)))
    y1 = min(img_h, int(np.ceil(y + h + margin * h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox {bbox} lies outside the {img_w}x{img_h} image")
    return (x0, y0, x1 - x0, y1 - y0)


def crop_with_margin(img: ImageU8, bbox: Box, margin: float) -> ImageU8:
    """Crop `bbox` enlarged by `margin` (fraction of the box size), clipped to the image."""
    x, y, w, h = expand_box(bbox, margin, img.shape)
    return img[y : y + h, x : x + w]


@lru_cache(maxsize=1)
def _cascade() -> Any | None:
    """Bundled Haar cascade, or None when this OpenCV build ships none (e.g. OpenCV 5)."""
    factory = getattr(cv2, "CascadeClassifier", None)
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if factory is None or data_dir is None:
        logger.warning("this OpenCV build has no Haar cascade; face detection is unavailable")
        return None
    path = Path(data_dir) / "haarcascade_frontalface_default.xml"
    cascade = factory(str(path))
    if cascade.empty():
        logger.warning("Haar cascade %s could not be loaded; face detection is unavailable", path)
        return None
    return cascade


def detect_face(img: ImageU8) -> Box | None:
    """Largest frontal face found by OpenCV's bundled Haar cascade, or None.

    None also means "detector unavailable" (logged once); callers cannot tell the two
    apart, and crop_face treats both as a logged fallback.
    """
    cascade = _cascade()
    if cascade is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    if len(found) == 0:
        return None
    x, y, w, h = max(found, key=lambda b: int(b[2]) * int(b[3]))
    return (int(x), int(y), int(w), int(h))


@dataclass(frozen=True)
class FaceCrop:
    image: ImageU8
    bbox: Box
    used_fallback: bool


def crop_face(
    img: ImageU8,
    margin: float = 0.2,
    fallback: Literal["center", "error"] = "center",
    source: str = "<array>",
) -> FaceCrop:
    """Detect and crop the face. A failed detection is logged and flagged, never silent.

    With fallback="center" a centred square is used and `used_fallback` is True; with
    fallback="error" a LookupError is raised instead.
    """
    bbox = detect_face(img)
    if bbox is not None:
        return FaceCrop(crop_with_margin(img, bbox, margin), bbox, False)
    if fallback == "error":
        raise LookupError(f"no face detected in {source}")
    h, w = img.shape[:2]
    side = max(1, round(_FALLBACK_SIDE_FRACTION * min(h, w)))
    bbox = ((w - side) // 2, (h - side) // 2, side, side)
    logger.warning("no face detected in %s; falling back to centre crop %s", source, bbox)
    return FaceCrop(crop_with_margin(img, bbox, margin), bbox, True)
