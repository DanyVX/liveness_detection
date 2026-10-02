import io
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from liveness.gate import GateConfig, LivenessGate, OnnxLivenessGate
from liveness.infer import (
    FaceBox,
    FrameScorer,
    HaarFaceDetector,
    HaarUnavailableError,
    InvalidImageError,
    OnnxScorer,
    ReasonCode,
    Verdict,
    build_tiny_onnx_model,
    crop_face,
    decode_image,
    normalize,
)


class StubDetector:
    def __init__(self, boxes: list[FaceBox]) -> None:
        self.boxes = boxes

    def detect(self, rgb: np.ndarray) -> list[FaceBox]:  # type: ignore[type-arg]
        return list(self.boxes)


def png(value: int | tuple[int, int, int], size: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (value,) * 3 if isinstance(value, int) else value).save(
        buf, format="PNG"
    )
    return buf.getvalue()


FACE = FaceBox(50, 50, 100, 100)


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_onnx_model(tmp_path_factory.mktemp("m") / "tiny.onnx")


@pytest.fixture(scope="module")
def scorer(model_path: Path) -> OnnxScorer:
    return OnnxScorer(model_path)


def frame_scorer(scorer: OnnxScorer, boxes: list[FaceBox]) -> FrameScorer:
    return FrameScorer(scorer, StubDetector(boxes), min_face_px=64, margin=0.2)


def gate(scorer: OnnxScorer, boxes: list[FaceBox] | None = None, **kw: object) -> OnnxLivenessGate:
    cfg: dict[str, object] = {
        "decision_threshold": 0.5,
        "clip_accept_threshold": 0.6,
        "clip_reject_threshold": 0.4,
        "clip_min_valid_frames": 3,
    }
    cfg.update(kw)
    return OnnxLivenessGate(
        frame_scorer(scorer, [FACE] if boxes is None else boxes),
        GateConfig(**cfg),  # type: ignore[arg-type]
    )


def test_corrupt_bytes_raise_typed_error_and_gate_reports_invalid_input(scorer: OnnxScorer) -> None:
    for bad in (b"", b"not an image", png(128)[:40]):
        with pytest.raises(InvalidImageError):
            decode_image(bad)
    r = gate(scorer).check_image(b"\x00garbage")
    assert (r.decision, r.reason_code) == (Verdict.ERROR, ReasonCode.INVALID_INPUT)
    assert r.score is None
    assert r.model_version == "tiny-test"


def test_tiny_image_is_insufficient_quality_not_spoof(scorer: OnnxScorer) -> None:
    r = gate(scorer).check_image(png(0, size=16))
    assert r.decision == Verdict.INSUFFICIENT_QUALITY
    assert r.reason_code == ReasonCode.INSUFFICIENT_QUALITY_TOO_SMALL


def test_small_face_box_is_too_small(scorer: OnnxScorer) -> None:
    r = gate(scorer, [FaceBox(10, 10, 30, 30)]).check_image(png(0))
    assert r.reason_code == ReasonCode.INSUFFICIENT_QUALITY_TOO_SMALL
    assert r.decision == Verdict.INSUFFICIENT_QUALITY


def test_no_face_is_insufficient_quality(scorer: OnnxScorer) -> None:
    r = gate(scorer, []).check_image(png(0))
    assert r.decision == Verdict.INSUFFICIENT_QUALITY
    assert r.reason_code == ReasonCode.INSUFFICIENT_QUALITY_NO_FACE


def test_multiple_faces_is_insufficient_quality(scorer: OnnxScorer) -> None:
    r = gate(scorer, [FACE, FaceBox(0, 0, 80, 80)]).check_image(png(255))
    assert r.decision == Verdict.INSUFFICIENT_QUALITY
    assert r.reason_code == ReasonCode.INSUFFICIENT_QUALITY_MULTIPLE_FACES


def test_scores_follow_contract_direction(scorer: OnnxScorer) -> None:
    live = gate(scorer).check_image(png(255))
    spoof = gate(scorer).check_image(png(0))
    assert live.decision == Verdict.LIVE and live.reason_code == ReasonCode.SCORE_ABOVE_THRESHOLD
    assert spoof.decision == Verdict.SPOOF and spoof.reason_code == ReasonCode.SCORE_BELOW_THRESHOLD
    assert live.score is not None and spoof.score is not None and live.score > spoof.score


def test_threshold_change_flips_decision_without_code_change(scorer: OnnxScorer) -> None:
    img = png(128)
    assert gate(scorer, decision_threshold=0.3).check_image(img).decision == Verdict.LIVE
    assert gate(scorer, decision_threshold=0.95).check_image(img).decision == Verdict.SPOOF


def test_gate_satisfies_protocol(scorer: OnnxScorer) -> None:
    assert isinstance(gate(scorer), LivenessGate)


def test_clip_decisions_and_insufficient_evidence(scorer: OnnxScorer) -> None:
    g = gate(scorer)
    assert g.check_clip([png(255)] * 5).decision == Verdict.LIVE
    assert g.check_clip([png(0)] * 5).decision == Verdict.SPOOF
    short = g.check_clip([png(255)] * 2)
    assert short.decision == Verdict.INSUFFICIENT_EVIDENCE
    assert short.reason_code == ReasonCode.INSUFFICIENT_EVIDENCE


def test_clip_with_no_face_frames_is_insufficient_evidence(scorer: OnnxScorer) -> None:
    r = gate(scorer, []).check_clip([png(255)] * 6)
    assert r.decision == Verdict.INSUFFICIENT_EVIDENCE


def test_clip_invalid_inputs(scorer: OnnxScorer) -> None:
    g = gate(scorer, max_clip_frames=4)
    for clip in ([], [png(255)] * 5, [png(255), b"junk", png(255)]):
        r = g.check_clip(clip)
        assert (r.decision, r.reason_code) == (Verdict.ERROR, ReasonCode.INVALID_INPUT)


def test_preprocessing_is_deterministic_and_normalised() -> None:
    chip = np.full((8, 8, 3), 255, np.uint8)
    out = normalize(chip)
    assert out.shape == (3, 8, 8) and out.dtype == np.float32
    assert out[0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-5)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)
    a = crop_face(img, FaceBox(10, 10, 40, 40), 0.2, (32, 32))
    b = crop_face(img, FaceBox(10, 10, 40, 40), 0.2, (32, 32))
    assert np.array_equal(a, b) and a.shape == (32, 32, 3)


@given(
    x=st.integers(-50, 150),
    y=st.integers(-50, 150),
    w=st.integers(1, 120),
    h=st.integers(1, 120),
    margin=st.floats(0.0, 1.0),
)
def test_crop_always_returns_requested_size(x: int, y: int, w: int, h: int, margin: float) -> None:
    img = np.zeros((60, 80, 3), np.uint8)
    assert crop_face(img, FaceBox(x, y, w, h), margin, (16, 24)).shape == (16, 24, 3)


def test_model_version_from_metadata_or_hash(tmp_path: Path) -> None:
    named = OnnxScorer(build_tiny_onnx_model(tmp_path / "a.onnx", model_version="v7"))
    assert named.model_version == "v7"
    anon = OnnxScorer(build_tiny_onnx_model(tmp_path / "b.onnx", model_version=None))
    assert len(anon.model_version) == 12
    int(anon.model_version, 16)


def test_dynamic_input_shape_defaults_to_128(tmp_path: Path) -> None:
    s = OnnxScorer(build_tiny_onnx_model(tmp_path / "d.onnx", size=None))
    assert s.input_size == (128, 128)
    fixed = OnnxScorer(build_tiny_onnx_model(tmp_path / "f.onnx", size=64))
    assert fixed.input_size == (64, 64)
    assert len(fixed.score_chips([np.zeros((64, 64, 3), np.uint8)] * 2)) == 2
    assert fixed.score_chips([]) == []


def test_providers_configurable_and_unavailable_ones_dropped(
    model_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger="liveness.infer"):
        s = OnnxScorer(model_path, providers=["TensorrtExecutionProvider", "CPUExecutionProvider"])
    assert s.providers == ["CPUExecutionProvider"]
    assert "CPUExecutionProvider" in caplog.text
    with pytest.raises(ValueError, match="providers"):
        OnnxScorer(model_path, providers=["NopeExecutionProvider"])


def test_missing_model_file_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OnnxScorer(tmp_path / "missing.onnx")


def test_non_finite_logit_raises(tmp_path: Path) -> None:
    from liveness.infer import ScoringError

    s = OnnxScorer(build_tiny_onnx_model(tmp_path / "n.onnx", bias=float("nan")))
    with pytest.raises(ScoringError):
        s.score_chips([np.zeros((128, 128, 3), np.uint8)])


def test_haar_detector_runs_or_fails_with_clear_error() -> None:
    try:
        det = HaarFaceDetector()
    except HaarUnavailableError as exc:  # OpenCV 5 wheels dropped the cascade
        assert "FaceDetector" in str(exc)
        return
    assert det.detect(np.full((200, 200, 3), 128, np.uint8)) == []


def test_exif_orientation_applied() -> None:
    im = Image.new("RGB", (40, 20), (10, 20, 30))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    assert decode_image(buf.getvalue()).shape[:2] == (40, 20)
