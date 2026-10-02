"""INT8 post-training quantization of the exported PAD model and an FP32-vs-INT8 comparison.

Static quantization (QDQ format, per-channel int8 weights, int8 activations calibrated on a
list of preprocessed images) is the default because the model is conv-heavy: dynamic
quantization only quantizes weights and computes activation scales at run time, which saves
size but gives little CPU speed-up for convolutions. If static quantization raises, or the
result cannot be loaded and run by onnxruntime, ``quantize_int8`` falls back to dynamic
quantization, logs a warning and records ``fallback_reason`` in the result; it never falls
back silently.

An INT8 model is a different model: ``compare_fp32_int8`` evaluates both on the same inputs
with thresholds chosen on validation scores separately for each (never the FP32 threshold
reused for INT8). On an untrained or synthetic-trained model the reported deltas demonstrate
the measurement and are not a result.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
from onnxruntime.quantization import (  # type: ignore[import-untyped]
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from scipy import stats  # type: ignore[import-untyped]

from liveness.data import Sample
from liveness.eval import ScoredSplit, evaluate_protocol
from liveness.infer import OnnxScorer, UInt8Image, decode_image, normalize

logger = logging.getLogger(__name__)

_F32 = npt.NDArray[np.float32]
_F64 = npt.NDArray[np.float64]
MODES = ("static", "dynamic")
QUANT_KEY = "quantization"


@dataclass(frozen=True)
class QuantizationResult:
    path: Path
    mode: str
    requested_mode: str
    fallback_reason: str | None
    fp32_bytes: int
    int8_bytes: int
    size_ratio: float

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


@dataclass(frozen=True)
class LabelledChips:
    """RGB uint8 chips of the model's input size with the metadata a ScoredSplit needs."""

    chips: Sequence[UInt8Image]
    labels: npt.NDArray[np.int64]
    attack_types: npt.NDArray[np.str_]
    subject_keys: Sequence[object]
    sample_ids: Sequence[str]

    def scored(self, scores: _F64) -> ScoredSplit:
        return ScoredSplit(
            scores=scores,
            labels=self.labels,
            attack_types=self.attack_types,
            subject_keys=self.subject_keys,
            sample_ids=self.sample_ids,
        )


def chips_from_samples(samples: Sequence[Sample], size: int | tuple[int, int]) -> LabelledChips:
    """Decode and area-resize whole images to ``size`` (H, W); no face detection."""
    h, w = (size, size) if isinstance(size, int) else size
    chips: list[UInt8Image] = []
    for s in samples:
        resized = cv2.resize(
            decode_image(s.path.read_bytes()), (w, h), interpolation=cv2.INTER_AREA
        )
        chips.append(np.ascontiguousarray(resized, dtype=np.uint8))
    return LabelledChips(
        chips=chips,
        labels=np.array([int(s.label) for s in samples], dtype=np.int64),
        attack_types=np.array([s.attack_type.value for s in samples]),
        subject_keys=[f"{s.dataset}:{s.subject_id}" for s in samples],
        sample_ids=[s.path.name for s in samples],
    )


class _ListReader(CalibrationDataReader):  # type: ignore[misc]
    def __init__(self, arrays: Sequence[_F32], input_name: str) -> None:
        self._it = iter([{input_name: a[None].astype(np.float32)} for a in arrays])

    def get_next(self) -> dict[str, _F32] | None:
        return next(self._it, None)


def calibration_inputs(chips: Sequence[UInt8Image]) -> list[_F32]:
    """Chips -> normalised 3xHxW float32 arrays, exactly as the scorer feeds the model."""
    return [normalize(c) for c in chips]


def _smoke_run(path: Path) -> None:
    """Raise unless onnxruntime can load the model and return finite [N,1] logits."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    dims = sess.get_inputs()[0].shape
    h = dims[2] if isinstance(dims[2], int) else 128
    w = dims[3] if isinstance(dims[3], int) else 128
    out = np.asarray(sess.run(None, {"input": np.zeros((2, 3, h, w), np.float32)})[0])
    if out.shape != (2, 1) or not np.all(np.isfinite(out)):
        raise RuntimeError(f"quantized model produced bad output (shape {out.shape})")


def _stamp(src: Path, dst: Path, mode: str) -> None:
    """Carry the FP32 metadata (model_version etc.) over and mark the quantization mode."""
    meta = {p.key: p.value for p in onnx.load(str(src), load_external_data=False).metadata_props}
    proto = onnx.load(str(dst))
    del proto.metadata_props[:]
    for key, value in {**meta, QUANT_KEY: f"int8-{mode}"}.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(dst))


def _static(fp32: Path, out: Path, calibration: Sequence[_F32], per_channel: bool) -> None:
    quantize_static(
        str(fp32),
        str(out),
        _ListReader(calibration, "input"),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=per_channel,
    )


def _dynamic(fp32: Path, out: Path) -> None:
    quantize_dynamic(str(fp32), str(out), weight_type=QuantType.QInt8)


def quantize_int8(
    fp32_path: str | Path,
    out_path: str | Path,
    calibration: Sequence[_F32] | None = None,
    *,
    mode: str = "static",
    per_channel: bool = True,
) -> QuantizationResult:
    """Quantize to INT8. ``mode="static"`` needs ``calibration`` (see calibration_inputs)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    src, dst = Path(fp32_path), Path(out_path)
    if not src.is_file():
        raise FileNotFoundError(f"model file not found: {src}")
    if mode == "static" and not calibration:
        raise ValueError("static quantization needs a non-empty calibration set")
    dst.parent.mkdir(parents=True, exist_ok=True)
    used, reason = mode, None
    if mode == "static":
        try:
            _static(src, dst, calibration or (), per_channel)
            _smoke_run(dst)
        except Exception as exc:  # onnxruntime raises many unrelated types here
            reason = f"static quantization failed ({type(exc).__name__}: {exc})"
            logger.warning("%s; falling back to dynamic quantization", reason)
            used = "dynamic"
    if used == "dynamic":
        _dynamic(src, dst)
        _smoke_run(dst)
    _stamp(src, dst, used)
    fp32_bytes, int8_bytes = src.stat().st_size, dst.stat().st_size
    return QuantizationResult(
        path=dst,
        mode=used,
        requested_mode=mode,
        fallback_reason=reason,
        fp32_bytes=fp32_bytes,
        int8_bytes=int8_bytes,
        size_ratio=int8_bytes / fp32_bytes,
    )


def score_chips(model_path: str | Path, chips: Sequence[UInt8Image], batch_size: int = 32) -> _F64:
    """Sigmoid scores of ``chips`` from an ONNX model, via the production OnnxScorer."""
    scorer = OnnxScorer(model_path)
    out: list[float] = []
    for i in range(0, len(chips), batch_size):
        out += scorer.score_chips(chips[i : i + batch_size])
    return np.asarray(out, dtype=np.float64)


def _finite(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def score_shift(fp32: _F64, int8: _F64) -> dict[str, Any]:
    """Paired score differences (int8 - fp32), KS statistic between the two score
    distributions and Spearman rank correlation (None when undefined, e.g. constant scores)."""
    if fp32.shape != int8.shape or fp32.size < 2:
        raise ValueError("need two score arrays of equal length >= 2")
    diff = int8 - fp32
    constant = bool(np.ptp(fp32) == 0 or np.ptp(int8) == 0)
    rho = float("nan") if constant else stats.spearmanr(fp32, int8).statistic
    return {
        "n": int(fp32.size),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "mean_abs_diff": float(np.abs(diff).mean()),
        "max_abs_diff": float(np.abs(diff).max()),
        "ks_statistic": float(stats.ks_2samp(fp32, int8).statistic),
        "spearman": _finite(rho),
    }


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    test = result["test"]
    ops = {}
    for name, op in test["operating_points"].items():
        apcer = op["apcer"]["max"]
        ops[name] = {
            "threshold": op["threshold"],
            "apcer_max": apcer,
            "bpcer": op["bpcer"],
            "acer": op["acer"],
        }
    return {
        "eer": test["threshold_free"]["eer"]["estimate"],
        "auc": test["threshold_free"]["auc"]["estimate"],
        "operating_points": ops,
    }


def _delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, va in a.items():
        out[key] = _delta(va, b[key]) if isinstance(va, dict) else b[key] - va
    return out


def compare_scored(
    fp32_val: ScoredSplit,
    fp32_test: ScoredSplit,
    int8_val: ScoredSplit,
    int8_test: ScoredSplit,
    *,
    protocol_name: str,
    protocol_manifest_hash: str,
    model_name: str,
    data_kind: str,
    target_bpcers: Sequence[float] = (0.01, 0.05),
    n_boot: int = 200,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Compare already-scored splits. Each model's thresholds come from its own val scores."""
    common: dict[str, Any] = {
        "protocol_name": protocol_name,
        "protocol_manifest_hash": protocol_manifest_hash,
        "data_kind": data_kind,
        "target_bpcers": target_bpcers,
        "n_boot": n_boot,
        "bootstrap_seed": bootstrap_seed,
    }
    full = {
        "fp32": evaluate_protocol(fp32_val, fp32_test, model_name=f"{model_name}-fp32", **common),
        "int8": evaluate_protocol(int8_val, int8_test, model_name=f"{model_name}-int8", **common),
    }
    summary = {k: _summary(v) for k, v in full.items()}
    thr = {
        k: {n: v for n, v in r["thresholds"].items() if n != "selected_on"} for k, r in full.items()
    }
    t_fp, t_i8 = thr["fp32"]["eer_val"], thr["int8"]["eer_val"]
    flips = (fp32_test.scores >= t_fp) != (int8_test.scores >= t_i8)
    return {
        "data_kind": data_kind,
        "model": model_name,
        "threshold_selection": "val only, chosen separately for fp32 and int8",
        "thresholds": thr,
        "score_shift": {
            "val": score_shift(fp32_val.scores, int8_val.scores),
            "test": score_shift(fp32_test.scores, int8_test.scores),
        },
        "metrics": {**summary, "delta_int8_minus_fp32": _delta(summary["fp32"], summary["int8"])},
        "decision_disagreement_at_val_eer_thresholds": float(flips.mean()),
        "val_n": full["fp32"]["val_n"],
        "test_n": full["fp32"]["test"]["n"],
        "small_sample_warning": bool(
            full["fp32"]["val_small_sample_warning"] or full["fp32"]["test"]["small_sample_warning"]
        ),
    }


def compare_fp32_int8(
    fp32_path: str | Path,
    int8_path: str | Path,
    val: LabelledChips,
    test: LabelledChips,
    *,
    protocol_name: str,
    protocol_manifest_hash: str,
    model_name: str,
    data_kind: str,
    target_bpcers: Sequence[float] = (0.01, 0.05),
    n_boot: int = 200,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Score the same val and test chips with both models and compare (see compare_scored)."""
    return compare_scored(
        val.scored(score_chips(fp32_path, val.chips)),
        test.scored(score_chips(fp32_path, test.chips)),
        val.scored(score_chips(int8_path, val.chips)),
        test.scored(score_chips(int8_path, test.chips)),
        protocol_name=protocol_name,
        protocol_manifest_hash=protocol_manifest_hash,
        model_name=model_name,
        data_kind=data_kind,
        target_bpcers=target_bpcers,
        n_boot=n_boot,
        bootstrap_seed=bootstrap_seed,
    )
