import io
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from liveness.api.app import create_app
from liveness.api.settings import ApiSettings
from liveness.infer import FaceBox, build_tiny_onnx_model

VERSION = "tiny-test"


class StubDetector:
    """No face when the top-left pixel is the (1, 2, 3) marker, else one fixed face."""

    def detect(self, rgb: np.ndarray) -> list[FaceBox]:  # type: ignore[type-arg]
        if tuple(rgb[0, 0]) == (1, 2, 3):
            return []
        return [FaceBox(50, 50, 100, 100)]


def png(value: int, size: int = 200, marker: bool = False) -> bytes:
    im = Image.new("RGB", (size, size), (value,) * 3)
    if marker:
        im.putpixel((0, 0), (1, 2, 3))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_onnx_model(tmp_path_factory.mktemp("api") / "tiny.onnx")


def make_settings(model_path: Path, **kw: object) -> ApiSettings:
    base: dict[str, object] = {
        "model_path": model_path,
        "decision_threshold": 0.5,
        "clip_accept_threshold": 0.6,
        "clip_reject_threshold": 0.4,
        "clip_min_valid_frames": 3,
        "max_batch_frames": 5,
        "max_image_bytes": 50_000,
        "min_face_px": 64,
    }
    base.update(kw)
    return ApiSettings(**base)  # type: ignore[arg-type]


def client(model_path: Path, **kw: object) -> TestClient:
    return TestClient(create_app(make_settings(model_path, **kw), detector=StubDetector()))


def one(data: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {"image": ("a.png", data, "image/png")}


def many(blobs: list[bytes]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("frames", (f"{i}.png", b, "image/png")) for i, b in enumerate(blobs)]


def assert_contract(body: dict[str, object]) -> None:
    for key in ("score", "decision", "reason_code", "model_version"):
        assert key in body
    assert body["model_version"] == VERSION


def test_healthz_and_model_info(model_path: Path) -> None:
    c = client(model_path)
    h = c.get("/healthz")
    assert h.status_code == 200 and h.json() == {"status": "ok", "model_version": VERSION}
    m = c.get("/v1/model").json()
    assert m["model_version"] == VERSION
    assert m["input_size"] == [128, 128]
    assert m["providers"] == ["CPUExecutionProvider"]
    assert m["decision_threshold"] == 0.5


def test_score_live_and_spoof(model_path: Path) -> None:
    c = client(model_path)
    live = c.post("/v1/score", files=one(png(255)))
    spoof = c.post("/v1/score", files=one(png(0)))
    assert live.status_code == 200 and spoof.status_code == 200
    assert live.json()["decision"] == "LIVE"
    assert spoof.json()["decision"] == "SPOOF"
    assert_contract(live.json())
    assert_contract(spoof.json())


def test_corrupt_image_is_invalid_input_with_model_version(model_path: Path) -> None:
    r = client(model_path).post("/v1/score", files=one(b"definitely not an image"))
    assert r.status_code == 422
    assert r.json()["reason_code"] == "INVALID_INPUT"
    assert r.json()["decision"] == "ERROR"
    assert_contract(r.json())


def test_tiny_image_and_no_face_are_insufficient_quality_not_spoof(model_path: Path) -> None:
    c = client(model_path)
    small = c.post("/v1/score", files=one(png(0, size=16))).json()
    noface = c.post("/v1/score", files=one(png(0, marker=True))).json()
    assert small["decision"] == "INSUFFICIENT_QUALITY"
    assert small["reason_code"] == "INSUFFICIENT_QUALITY_TOO_SMALL"
    assert noface["decision"] == "INSUFFICIENT_QUALITY"
    assert noface["reason_code"] == "INSUFFICIENT_QUALITY_NO_FACE"
    assert_contract(small)
    assert_contract(noface)


def test_decide_fuses_clip(model_path: Path) -> None:
    c = client(model_path)
    live = c.post("/v1/decide", files=many([png(255)] * 4))
    assert live.status_code == 200
    assert live.json()["decision"] == "LIVE" and live.json()["n_frames"] == 4
    spoof = c.post("/v1/decide", files=many([png(0)] * 4)).json()
    assert spoof["decision"] == "SPOOF"


def test_decide_short_clip_and_faceless_frames_are_insufficient_evidence(model_path: Path) -> None:
    c = client(model_path)
    short = c.post("/v1/decide", files=many([png(255)] * 2)).json()
    assert short["decision"] == "INSUFFICIENT_EVIDENCE"
    assert short["reason_code"] == "INSUFFICIENT_EVIDENCE"
    mixed = [
        png(255),
        png(255, marker=True),
        png(255, marker=True),
        png(255),
        png(255, marker=True),
    ]
    r = c.post("/v1/decide", files=many(mixed)).json()
    assert r["decision"] == "INSUFFICIENT_EVIDENCE"
    assert_contract(r)


def test_batch_limit_exceeded_returns_413_with_model_version(model_path: Path) -> None:
    r = client(model_path).post("/v1/decide", files=many([png(255)] * 6))
    assert r.status_code == 413
    assert r.json()["decision"] == "ERROR"
    assert_contract(r.json())


def test_oversize_image_returns_413_with_model_version(model_path: Path) -> None:
    c = client(model_path, max_image_bytes=100)
    for r in (
        c.post("/v1/score", files=one(png(255))),
        c.post("/v1/decide", files=many([png(255)] * 2)),
    ):
        assert r.status_code == 413
        assert_contract(r.json())


def test_oversize_content_length_rejected_early(model_path: Path) -> None:
    c = client(model_path, max_image_bytes=1000, max_batch_frames=1)
    r = c.post("/v1/score", content=b"x" * 200_000, headers={"content-type": "text/plain"})
    assert r.status_code == 413
    assert_contract(r.json())


def test_validation_and_routing_errors_keep_contract(model_path: Path) -> None:
    c = client(model_path)
    missing = c.post("/v1/score")
    assert missing.status_code == 422
    assert_contract(missing.json())
    nf = c.get("/nope")
    assert nf.status_code == 404
    assert_contract(nf.json())


def test_unhandled_error_is_500_with_contract_and_request_id(
    model_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    class Boom:
        def detect(self, rgb: np.ndarray) -> list[FaceBox]:  # type: ignore[type-arg]
            raise RuntimeError("boom")

    c = TestClient(create_app(make_settings(model_path), detector=Boom()))
    r = c.post("/v1/score", files=one(png(255)), headers={"X-Request-ID": "rid-500"})
    assert r.status_code == 500
    assert r.headers["X-Request-ID"] == "rid-500"
    assert_contract(r.json())
    assert "boom" not in r.text


def test_threshold_change_via_settings_changes_decision(model_path: Path) -> None:
    img = png(128)
    lenient = client(model_path, decision_threshold=0.3).post("/v1/score", files=one(img))
    strict = client(model_path, decision_threshold=0.95).post("/v1/score", files=one(img))
    assert lenient.json()["decision"] == "LIVE"
    assert strict.json()["decision"] == "SPOOF"


def test_thresholds_read_from_environment(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LIVENESS_API_MODEL_PATH", str(model_path))
    monkeypatch.setenv("LIVENESS_API_DECISION_THRESHOLD", "0.95")
    monkeypatch.setenv("LIVENESS_API_CLIP_ACCEPT_THRESHOLD", "0.9")
    monkeypatch.setenv("LIVENESS_API_CLIP_REJECT_THRESHOLD", "0.1")
    c = TestClient(create_app(detector=StubDetector()))
    assert c.post("/v1/score", files=one(png(128))).json()["decision"] == "SPOOF"


def test_correlation_id_echoed_generated_and_logged(
    model_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    c = client(model_path)
    with caplog.at_level("INFO", logger="liveness.api"):
        echoed = c.get("/healthz", headers={"X-Request-ID": "abc-123"})
        generated = c.get("/healthz")
        hostile = c.get("/healthz", headers={"X-Request-ID": "bad id\twith spaces"})
    assert echoed.headers["X-Request-ID"] == "abc-123"
    assert re.fullmatch(r"[0-9a-f]{32}", generated.headers["X-Request-ID"])
    assert hostile.headers["X-Request-ID"] != "bad id\twith spaces"
    assert "abc-123" in {getattr(r, "correlation_id", None) for r in caplog.records}


def test_logs_contain_no_image_bytes_or_scores(
    model_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    c = client(model_path)
    with caplog.at_level("DEBUG"):
        r = c.post("/v1/score", files=one(png(255)))
    score = str(r.json()["score"])[:6]
    assert "a.png" not in caplog.text
    assert score not in caplog.text


def test_missing_model_path_fails_fast(model_path: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        create_app(make_settings(tmp_path / "missing.onnx"), detector=StubDetector())


def test_settings_without_model_path_or_with_bad_thresholds_fail_fast(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in ("MODEL_PATH", "DECISION_THRESHOLD"):
        monkeypatch.delenv(f"LIVENESS_API_{k}", raising=False)
    with pytest.raises(ValidationError):
        ApiSettings()  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        make_settings(model_path, clip_accept_threshold=0.3, clip_reject_threshold=0.6)


def test_uploads_are_not_persisted(
    model_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    c = client(model_path)
    c.post("/v1/score", files=one(png(255)))
    assert list(tmp_path.iterdir()) == []
