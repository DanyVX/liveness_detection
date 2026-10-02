"""ONNX inference: image decoding, quality checks, deterministic preprocessing, scoring.

Model contract: input "input" float32 [N,3,H,W], RGB in 0..1 then ImageNet-normalised; output
"logit" float32 [N,1]; score = sigmoid(logit), higher = more bona fide.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = 128
MODEL_VERSION_KEY = "model_version"

UInt8Image = npt.NDArray[np.uint8]


class Verdict(StrEnum):
    LIVE = "LIVE"
    SPOOF = "SPOOF"
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ERROR = "ERROR"


class ReasonCode(StrEnum):
    OK = "OK"
    INSUFFICIENT_QUALITY_TOO_SMALL = "INSUFFICIENT_QUALITY_TOO_SMALL"
    INSUFFICIENT_QUALITY_NO_FACE = "INSUFFICIENT_QUALITY_NO_FACE"
    INSUFFICIENT_QUALITY_MULTIPLE_FACES = "INSUFFICIENT_QUALITY_MULTIPLE_FACES"
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    SCORE_BELOW_THRESHOLD = "SCORE_BELOW_THRESHOLD"
    SCORE_ABOVE_THRESHOLD = "SCORE_ABOVE_THRESHOLD"


class InvalidImageError(ValueError):
    """Bytes could not be decoded as an image."""


class ScoringError(RuntimeError):
    """The model produced an unusable output."""


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int


class HaarUnavailableError(RuntimeError):
    """The bundled Haar cascade cannot be used with the installed OpenCV."""


class FaceDetector(Protocol):
    def detect(self, rgb: UInt8Image) -> list[FaceBox]: ...


class HaarFaceDetector:
    """OpenCV's bundled frontal-face Haar cascade (cheap, weak; swap via FaceDetector)."""

    def __init__(self, min_size: int = 24, scale_factor: float = 1.1, min_neighbors: int = 5):
        # OpenCV 5.x wheels ship neither CascadeClassifier nor the cascade XML files.
        if not hasattr(cv2, "CascadeClassifier"):
            raise HaarUnavailableError(
                f"OpenCV {cv2.__version__} has no CascadeClassifier; "
                "install opencv-python-headless<5 or inject another FaceDetector"
            )
        cascade_dir = Path(cv2.data.haarcascades)  # type: ignore[attr-defined]
        path = cascade_dir / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(path))
        if not path.is_file() or cascade.empty():
            raise HaarUnavailableError(f"could not load Haar cascade at {path}")
        self._cascade = cascade
        self._min_size = min_size
        self._scale = scale_factor
        self._neighbors = min_neighbors

    def detect(self, rgb: UInt8Image) -> list[FaceBox]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale,
            minNeighbors=self._neighbors,
            minSize=(self._min_size, self._min_size),
        )
        return [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in found]


def decode_image(data: bytes) -> UInt8Image:
    """Decode to an RGB uint8 array (EXIF orientation applied). Raises InvalidImageError."""
    if not data:
        raise InvalidImageError("empty input")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            oriented = ImageOps.exif_transpose(opened)
            rgb = np.asarray(oriented.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise InvalidImageError("could not decode image") from None
    if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise InvalidImageError("degenerate image")
    return rgb


def crop_face(rgb: UInt8Image, box: FaceBox, margin: float, size: tuple[int, int]) -> UInt8Image:
    """Square crop centred on the box, side = max(w, h) * (1 + 2 * margin), edge-padded."""
    side = max(1, round(max(box.w, box.h) * (1.0 + 2.0 * margin)))
    cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
    x0, y0 = round(cx - side / 2.0), round(cy - side / 2.0)
    h, w = rgb.shape[:2]
    pad = ((max(0, -y0), max(0, y0 + side - h)), (max(0, -x0), max(0, x0 + side - w)), (0, 0))
    padded = np.pad(rgb, pad, mode="edge")
    px0, py0 = x0 + pad[1][0], y0 + pad[0][0]
    crop = padded[py0 : py0 + side, px0 : px0 + side]
    return np.ascontiguousarray(cv2.resize(crop, (size[1], size[0]), interpolation=cv2.INTER_AREA))


def normalize(chip: UInt8Image) -> npt.NDArray[np.float32]:
    """HxWx3 uint8 RGB -> 3xHxW float32, scaled to 0..1 then ImageNet-normalised."""
    scaled = chip.astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(((scaled - mean) / std).transpose(2, 0, 1), dtype=np.float32)


def _sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    z = np.clip(x.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


class OnnxScorer:
    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        default_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        available = ort.get_available_providers()
        requested = list(providers) if providers else ["CPUExecutionProvider"]
        chosen = [p for p in requested if p in available]
        if not chosen:
            raise ValueError(f"none of the providers {requested} are available: {available}")
        if chosen != requested:
            logger.warning("providers unavailable, using %s instead of %s", chosen, requested)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(str(path), sess_options=opts, providers=chosen)
        self._providers = list(self._session.get_providers())
        logger.info("onnx providers in use: %s", self._providers)
        inp = self._session.get_inputs()[0]
        if inp.name != "input":
            raise ValueError(f"model input must be named 'input', got {inp.name!r}")
        dims = inp.shape
        self._size = (
            dims[2] if isinstance(dims[2], int) and dims[2] > 0 else default_size,
            dims[3] if isinstance(dims[3], int) and dims[3] > 0 else default_size,
        )
        meta = self._session.get_modelmeta().custom_metadata_map
        self._version = (
            meta.get(MODEL_VERSION_KEY) or hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        )

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def input_size(self) -> tuple[int, int]:
        return self._size

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    def score_chips(self, chips: Sequence[UInt8Image]) -> list[float]:
        """Score already-cropped RGB uint8 chips of exactly input_size."""
        if not chips:
            return []
        batch = np.stack([normalize(c) for c in chips]).astype(np.float32)
        out = self._session.run(["logit"], {"input": batch})[0]
        logits = np.asarray(out, dtype=np.float32).reshape(len(chips), -1)
        if logits.shape[1] != 1 or not np.all(np.isfinite(logits)):
            raise ScoringError("model returned a non-finite or mis-shaped logit")
        return [float(s) for s in _sigmoid(logits[:, 0])]


@dataclass(frozen=True)
class FrameResult:
    score: float | None
    reason: ReasonCode


class FrameScorer:
    """decode -> quality checks -> crop -> score. Quality failures never become SPOOF."""

    def __init__(
        self, scorer: OnnxScorer, detector: FaceDetector, *, min_face_px: int, margin: float
    ) -> None:
        self.scorer = scorer
        self._detector = detector
        self._min_face_px = min_face_px
        self._margin = margin

    def analyze(self, data: bytes) -> FrameResult:
        """Raises InvalidImageError for undecodable bytes."""
        rgb = decode_image(data)
        if min(rgb.shape[:2]) < self._min_face_px:
            return FrameResult(None, ReasonCode.INSUFFICIENT_QUALITY_TOO_SMALL)
        faces = self._detector.detect(rgb)
        if not faces:
            return FrameResult(None, ReasonCode.INSUFFICIENT_QUALITY_NO_FACE)
        if len(faces) > 1:
            return FrameResult(None, ReasonCode.INSUFFICIENT_QUALITY_MULTIPLE_FACES)
        box = faces[0]
        if min(box.w, box.h) < self._min_face_px:
            return FrameResult(None, ReasonCode.INSUFFICIENT_QUALITY_TOO_SMALL)
        chip = crop_face(rgb, box, self._margin, self.scorer.input_size)
        return FrameResult(self.scorer.score_chips([chip])[0], ReasonCode.OK)


def build_tiny_onnx_model(
    path: str | Path,
    *,
    size: int | None = DEFAULT_INPUT_SIZE,
    weights: Sequence[float] = (1.0, 1.0, 1.0),
    bias: float = 0.0,
    model_version: str | None = "tiny-test",
) -> Path:
    """Hand-built model (global-average-pool + fixed linear layer) for tests and smoke benches.

    NOT a real detector. ``size=None`` makes H/W dynamic. Needs the dev-only ``onnx`` package.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    dim: int | str = "H" if size is None else size
    dim_w: int | str = "W" if size is None else size
    w = numpy_helper.from_array(np.asarray(weights, np.float32).reshape(3, 1), "w")
    b = numpy_helper.from_array(np.asarray([bias], np.float32), "b")
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "w"], ["mm"]),
        helper.make_node("Add", ["mm", "b"], ["logit"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_liveness",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, dim, dim_w])],
        [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["N", 1])],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    if model_version is not None:
        entry = model.metadata_props.add()
        entry.key, entry.value = MODEL_VERSION_KEY, model_version
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    return out
