import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import timm
import torch
from scipy import stats

from liveness import quantize
from liveness.data import Split, make_subject_disjoint_protocol
from liveness.data.synthetic import generate_synthetic_dataset
from liveness.eval import ScoredSplit
from liveness.export_onnx import export_to_onnx
from liveness.infer import OnnxScorer
from liveness.quantize import (
    LabelledChips,
    calibration_inputs,
    chips_from_samples,
    compare_fp32_int8,
    compare_scored,
    quantize_int8,
    score_chips,
    score_shift,
)

SIZE = 64


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> dict[Split, LabelledChips]:
    samples = generate_synthetic_dataset(tmp_path_factory.mktemp("syn"), n_subjects=24, seed=0)
    proto = make_subject_disjoint_protocol("syn-q", samples, seed=0)
    return {s: chips_from_samples(list(proto.splits[s]), SIZE) for s in Split}


@pytest.fixture(scope="module")
def fp32(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = timm.create_model("mobilenetv3_small_050", pretrained=False, num_classes=1).eval()
    with torch.no_grad():  # random init gives near-constant logits; widen them
        model.classifier.weight.mul_(40.0)
    return export_to_onnx(model, tmp_path_factory.mktemp("fp") / "fp32.onnx", SIZE, "q-test")


@pytest.fixture(scope="module")
def int8(fp32: Path, data: dict[Split, LabelledChips]) -> quantize.QuantizationResult:
    calib = calibration_inputs(list(data[Split.TRAIN].chips)[:32])
    return quantize_int8(fp32, fp32.parent / "int8.onnx", calib)


def test_static_int8_is_smaller_loadable_and_keeps_metadata(
    fp32: Path, int8: quantize.QuantizationResult
) -> None:
    assert int8.mode == "static" and int8.fallback_reason is None
    assert int8.int8_bytes < int8.fp32_bytes
    assert 0.0 < int8.size_ratio < 0.6
    assert int8.size_ratio == pytest.approx(int8.int8_bytes / int8.fp32_bytes)
    scorer = OnnxScorer(int8.path)
    assert scorer.model_version == "q-test"
    meta = ort.InferenceSession(str(int8.path)).get_modelmeta().custom_metadata_map
    assert meta["quantization"] == "int8-static" and meta["export_image_size"] == "64x64"
    assert json.dumps(int8.as_dict())


def test_int8_scores_are_rank_close_but_not_identical(
    fp32: Path, int8: quantize.QuantizationResult, data: dict[Split, LabelledChips]
) -> None:
    chips = data[Split.TEST].chips
    a, b = score_chips(fp32, chips), score_chips(int8.path, chips)
    assert not np.array_equal(a, b)
    assert float(np.abs(a - b).max()) > 0.0
    assert stats.spearmanr(a, b).statistic > 0.9
    assert np.all(np.isfinite(b))


def test_dynamic_mode_needs_no_calibration(fp32: Path) -> None:
    res = quantize_int8(fp32, fp32.parent / "dyn.onnx", mode="dynamic")
    assert res.mode == "dynamic" and res.requested_mode == "dynamic"
    assert res.fallback_reason is None and res.int8_bytes < res.fp32_bytes
    assert len(score_chips(res.path, [np.zeros((SIZE, SIZE, 3), np.uint8)] * 3)) == 3


def test_static_failure_falls_back_to_dynamic_with_logged_reason(
    fp32: Path,
    data: dict[Split, LabelledChips],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("calibration exploded")

    monkeypatch.setattr(quantize, "_static", boom)
    calib = calibration_inputs(list(data[Split.TRAIN].chips)[:4])
    with caplog.at_level(logging.WARNING, logger="liveness.quantize"):
        res = quantize_int8(fp32, fp32.parent / "fb.onnx", calib)
    assert res.mode == "dynamic" and res.requested_mode == "static"
    assert res.fallback_reason is not None and "calibration exploded" in res.fallback_reason
    assert "falling back to dynamic" in caplog.text and "calibration exploded" in caplog.text
    meta = ort.InferenceSession(str(res.path)).get_modelmeta().custom_metadata_map
    assert meta["quantization"] == "int8-dynamic"


def test_unusable_static_output_also_falls_back(
    fp32: Path, data: dict[Split, LabelledChips], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = quantize._smoke_run
    calls: list[Path] = []

    def flaky(path: Path) -> None:
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("cannot run static model")
        real(path)

    monkeypatch.setattr(quantize, "_smoke_run", flaky)
    calib = calibration_inputs(list(data[Split.TRAIN].chips)[:4])
    res = quantize_int8(fp32, fp32.parent / "fb2.onnx", calib)
    assert res.mode == "dynamic" and "cannot run static model" in (res.fallback_reason or "")


def test_quantize_argument_errors(fp32: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="calibration"):
        quantize_int8(fp32, tmp_path / "o.onnx")
    with pytest.raises(ValueError, match="mode"):
        quantize_int8(fp32, tmp_path / "o.onnx", mode="int4")
    with pytest.raises(FileNotFoundError):
        quantize_int8(tmp_path / "missing.onnx", tmp_path / "o.onnx", mode="dynamic")


def scored(scores: list[float], labels: list[int]) -> ScoredSplit:
    n = len(scores)
    return ScoredSplit(
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        attack_types=np.asarray(["none" if v else "print" for v in labels]),
        subject_keys=[f"s{i}" for i in range(n)],
        sample_ids=[f"i{i}" for i in range(n)],
    )


def splits(seed: int, shift: float = 0.0) -> tuple[ScoredSplit, ScoredSplit]:
    rng = np.random.default_rng(seed)
    labels = [1] * 40 + [0] * 40

    def one() -> ScoredSplit:
        s = np.where(np.asarray(labels) == 1, 0.65, 0.35) + rng.normal(0, 0.15, 80) + shift
        return scored(np.clip(s, 0.001, 0.999).tolist(), labels)

    return one(), one()


KW = {
    "protocol_name": "p",
    "protocol_manifest_hash": "h",
    "model_name": "m",
    "data_kind": "synthetic_smoke",
    "n_boot": 20,
}


def test_thresholds_use_val_only_and_are_chosen_per_model() -> None:
    fv, ft = splits(0)
    iv, it = splits(1, shift=0.05)
    base = compare_scored(fv, ft, iv, it, **KW)
    other_ft = scored((1.0 - ft.scores).tolist(), ft.labels.tolist())
    other_it = scored((1.0 - it.scores).tolist(), it.labels.tolist())
    changed_test = compare_scored(fv, other_ft, iv, other_it, **KW)
    assert changed_test["thresholds"] == base["thresholds"]
    assert changed_test["metrics"]["fp32"]["eer"] != base["metrics"]["fp32"]["eer"]
    assert base["thresholds"]["fp32"] != base["thresholds"]["int8"]
    assert base["threshold_selection"].startswith("val only")
    moved_val = compare_scored(iv, ft, iv, it, **KW)
    assert moved_val["thresholds"]["fp32"] != base["thresholds"]["fp32"]


def test_identical_models_have_zero_delta() -> None:
    v, t = splits(2)
    res = compare_scored(v, t, v, t, **KW)
    assert res["score_shift"]["test"]["max_abs_diff"] == 0.0
    assert res["score_shift"]["test"]["ks_statistic"] == 0.0
    assert res["decision_disagreement_at_val_eer_thresholds"] == 0.0
    delta = res["metrics"]["delta_int8_minus_fp32"]
    assert delta["eer"] == 0.0 and delta["operating_points"]["eer_val"]["acer"] == 0.0


def test_compare_fp32_int8_end_to_end(
    fp32: Path, int8: quantize.QuantizationResult, data: dict[Split, LabelledChips]
) -> None:
    res = compare_fp32_int8(
        fp32,
        int8.path,
        data[Split.VAL],
        data[Split.TEST],
        **KW,  # type: ignore[arg-type]
    )
    json.dumps(res, allow_nan=False)
    assert res["data_kind"] == "synthetic_smoke"
    shift = res["score_shift"]["test"]
    assert shift["n"] == len(data[Split.TEST].chips)
    assert shift["spearman"] > 0.9 and 0.0 < shift["max_abs_diff"] < 0.5
    ops = res["metrics"]["delta_int8_minus_fp32"]["operating_points"]
    assert set(ops) == {"eer_val", "bpcer_0.01", "bpcer_0.05"}
    assert set(res["thresholds"]) == {"fp32", "int8"}


def test_score_shift_edge_cases() -> None:
    const = np.full(5, 0.5)
    assert score_shift(const, const)["spearman"] is None
    with pytest.raises(ValueError, match="equal length"):
        score_shift(np.zeros(3), np.zeros(4))
    s = score_shift(np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.5, 0.8]))
    assert s["spearman"] == pytest.approx(1.0) and s["mean_diff"] == pytest.approx(0.0)


def test_chips_from_samples_shape_and_labels(data: dict[Split, LabelledChips]) -> None:
    c = data[Split.VAL]
    assert all(ch.shape == (SIZE, SIZE, 3) and ch.dtype == np.uint8 for ch in c.chips)
    assert len(c.labels) == len(c.chips) == len(c.sample_ids)
    assert set(c.labels.tolist()) == {0, 1}
