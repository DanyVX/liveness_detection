"""PyTorch -> ONNX export and a PyTorch-vs-onnxruntime parity check.

Model contract (see liveness.infer): input "input" float32 [N,3,H,W] (RGB, 0..1 then ImageNet
normalised by the caller, not in the graph), output "logit" float32 [N,1], batch axis dynamic,
``model_version`` stored in the ONNX ``metadata_props``.

The legacy TorchScript exporter (``dynamo=False``) is used on purpose: with torch 2.14 the
dynamo exporter is the default but needs the extra ``onnxscript`` package, and the legacy path
exports timm MobileNetV3 (hardswish/hardsigmoid) with a fixed opset reliably. Exporting twice
yields byte-identical files (no timestamps are written).

    uv run python -m liveness.export_onnx --checkpoint runs/m.pt --out models/pad.onnx \
        --image-size 128 --model-version pad-v1
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import torch
from torch import nn

from liveness.infer import (
    DEFAULT_INPUT_SIZE,
    MODEL_VERSION_KEY,
    UInt8Image,
    decode_image,
    normalize,
)

logger = logging.getLogger(__name__)

DEFAULT_ARCH = "mobilenetv3_small_050"
DEFAULT_OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "logit"
EXPORTER = "torchscript (dynamo=False)"
_F32 = npt.NDArray[np.float32]


class ExportError(ValueError):
    """The model or checkpoint cannot be exported under the model contract."""


@dataclass(frozen=True)
class ParityReport:
    """PyTorch vs onnxruntime. ``max_*`` are over logits; the ``score`` fields over sigmoid."""

    max_abs_diff: float
    max_rel_diff: float
    max_abs_diff_score: float
    n_inputs: int
    atol: float
    rtol: float
    passed: bool


def _hw(image_size: int | tuple[int, int]) -> tuple[int, int]:
    h, w = (image_size, image_size) if isinstance(image_size, int) else image_size
    if h < 1 or w < 1:
        raise ExportError(f"image_size must be positive, got {image_size}")
    return int(h), int(w)


def export_to_onnx(
    model: nn.Module,
    path: str | Path,
    image_size: int | tuple[int, int],
    model_version: str,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Export ``model`` (put in eval mode for the export, original mode restored afterwards)."""
    if not model_version:
        raise ExportError("model_version must be non-empty")
    h, w = _hw(image_size)
    was_training = model.training
    model.eval()
    try:
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        dummy = torch.zeros(2, 3, h, w, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(dummy)
        if tuple(out.shape) != (2, 1):
            raise ExportError(f"model must output logits of shape [N,1], got {tuple(out.shape)}")
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # legacy-exporter deprecation noise
            raw = Path(tmp) / "raw.onnx"
            torch.onnx.export(
                model,
                (dummy,),
                raw,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
                opset_version=opset,
                dynamo=False,
            )
            proto = onnx.load(str(raw))
    finally:
        model.train(was_training)

    meta = {
        MODEL_VERSION_KEY: model_version,
        "export_torch_version": str(torch.__version__),
        "export_onnx_version": onnx.__version__,
        "export_image_size": f"{h}x{w}",
        "export_opset": str(opset),
        "export_exporter": EXPORTER,
    }
    del proto.metadata_props[:]
    for key, value in meta.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(proto, full_check=True)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(out_path))
    logger.info("exported %s (opset %d, %dx%d, version %s)", out_path, opset, h, w, model_version)
    return out_path


def _fixture_chips(h: int, w: int, seed: int, n_subjects: int = 2) -> list[UInt8Image]:
    import cv2

    from liveness.data.synthetic import generate_synthetic_dataset

    with tempfile.TemporaryDirectory() as tmp:
        samples = generate_synthetic_dataset(Path(tmp) / "fx", n_subjects, seed)
        chips: list[UInt8Image] = []
        for s in samples:
            rgb = decode_image(s.path.read_bytes())
            chips.append(
                np.ascontiguousarray(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA))
            )
    return chips


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))), dtype=np.float64)


def verify_parity(
    model: nn.Module,
    onnx_path: str | Path,
    image_size: int | tuple[int, int],
    n: int = 8,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    seed: int = 0,
    *,
    use_fixture: bool = True,
) -> ParityReport:
    """Compare PyTorch and onnxruntime logits and scores on ``n`` seeded random images plus
    (``use_fixture``) the synthetic-fixture images, all normalised like infer.normalize."""
    h, w = _hw(image_size)
    rng = np.random.default_rng(seed)
    chips: list[UInt8Image] = [
        rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(n)
    ]
    if use_fixture:
        chips += _fixture_chips(h, w, seed)
    if not chips:
        raise ExportError("no inputs to compare (n=0 and use_fixture=False)")
    batch = np.stack([normalize(c) for c in chips]).astype(np.float32)

    was_training = model.training
    model.eval()
    try:
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        with torch.no_grad():
            ref = model(torch.from_numpy(batch).to(device)).detach().cpu().numpy()
    finally:
        model.train(was_training)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    got = np.asarray(session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0])
    if got.shape != ref.shape:
        raise ExportError(f"shape mismatch: torch {ref.shape} vs onnx {got.shape}")
    ref64, got64 = ref.astype(np.float64), got.astype(np.float64)
    abs_diff = np.abs(ref64 - got64)
    rel_diff = abs_diff / np.maximum(np.abs(ref64), 1e-12)
    score_diff = np.abs(_sigmoid(ref64) - _sigmoid(got64))
    passed = bool(
        np.all(np.isfinite(got64))
        and np.allclose(got64, ref64, atol=atol, rtol=rtol)
        and np.all(score_diff <= atol + rtol * np.abs(_sigmoid(ref64)))
    )
    return ParityReport(
        max_abs_diff=float(abs_diff.max()),
        max_rel_diff=float(rel_diff.max()),
        max_abs_diff_score=float(score_diff.max()),
        n_inputs=len(chips),
        atol=atol,
        rtol=rtol,
        passed=passed,
    )


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], str | None]:
    """Read ``{"model": state_dict, ...}`` written by torch.save; returns (state_dict, arch|None).

    The architecture name is taken from an optional ``"arch"`` key.
    """
    p = Path(path)
    if not p.is_file():
        raise ExportError(f"checkpoint not found: {p}")
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ExportError(f"could not read checkpoint {p}: {exc}") from exc
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model"), dict):
        raise ExportError(
            f"checkpoint {p} must be a dict with a 'model' state_dict "
            f"(found {type(ckpt).__name__} with keys "
            f"{sorted(ckpt) if isinstance(ckpt, dict) else 'n/a'})"
        )
    arch = ckpt.get("arch")
    return ckpt["model"], arch if isinstance(arch, str) else None


def load_model_from_checkpoint(ckpt_path: str | Path, arch: str | None = None) -> nn.Module:
    """Build a timm single-logit model and load the checkpoint's state_dict (eval mode)."""
    import timm

    state, ckpt_arch = load_checkpoint(ckpt_path)
    name = arch or ckpt_arch or DEFAULT_ARCH
    try:
        model: nn.Module = timm.create_model(name, pretrained=False, num_classes=1)
    except Exception as exc:
        raise ExportError(f"cannot build timm architecture {name!r}: {exc}") from exc
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ExportError(f"checkpoint does not match architecture {name!r}: {exc}") from exc
    return model.eval()


def export_from_checkpoint(
    ckpt_path: str | Path,
    out_path: str | Path,
    image_size: int | tuple[int, int] = DEFAULT_INPUT_SIZE,
    model_version: str = "unversioned",
    *,
    arch: str | None = None,
    opset: int = DEFAULT_OPSET,
) -> Path:
    model = load_model_from_checkpoint(ckpt_path, arch)
    return export_to_onnx(model, out_path, image_size, model_version, opset)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a PAD checkpoint to ONNX and check parity.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-size", type=int, default=DEFAULT_INPUT_SIZE)
    ap.add_argument("--model-version", required=True)
    ap.add_argument(
        "--arch", default=None, help=f"timm name (default: ckpt 'arch' or {DEFAULT_ARCH})"
    )
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        model = load_model_from_checkpoint(args.checkpoint, args.arch)
        out = export_to_onnx(model, args.out, args.image_size, args.model_version, args.opset)
        if args.no_verify:
            print(f"wrote {out}")
            return 0
        report = verify_parity(model, out, args.image_size, atol=args.atol)
    except ExportError as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote {out}\n{report}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
