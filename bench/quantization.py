"""FP32 vs INT8 comparison on the synthetic fixture (NOT a real result).

Trains a tiny timm MobileNetV3 for a few epochs on synthetic domain A (or leaves it at
random init with --epochs 0), exports it to ONNX, checks PyTorch/ONNX parity, quantizes to INT8
(calibration on train images only), and compares sizes, CPU latency and PAD metrics with
thresholds chosen on validation separately per model. Writes results/quantization_<label>.json
labelled data_kind=synthetic_smoke: it demonstrates the measurement, not a performance claim.

    uv run python bench/quantization.py --label smoke
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latency import measure, summarise

from liveness.data import Split, make_subject_disjoint_protocol
from liveness.data.synthetic import generate_synthetic_dataset
from liveness.eval import write_results_json
from liveness.export_onnx import export_to_onnx, verify_parity
from liveness.infer import OnnxScorer, normalize
from liveness.quantize import (
    LabelledChips,
    calibration_inputs,
    chips_from_samples,
    compare_fp32_int8,
    quantize_int8,
)

DATA_KIND = "synthetic_smoke"
ARCH = "mobilenetv3_small_050"
NOTE = (
    "synthetic_smoke: tiny model on the synthetic fixture. Demonstrates how FP32 vs INT8 is "
    "measured; the numbers say nothing about a real model or real attacks."
)


def train_briefly(model: nn.Module, chips: LabelledChips, epochs: int, seed: int) -> list[float]:
    """Plain BCE training on normalised chips; returns the mean loss of each epoch."""
    if epochs <= 0:
        return []
    x = torch.from_numpy(np.stack([normalize(c) for c in chips.chips]))
    y = torch.from_numpy(chips.labels.astype(np.float32)).unsqueeze(1)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        total = 0.0
        for i in range(0, len(x), 16):
            idx = order[i : i + 16]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        losses.append(total / len(x))
    model.eval()
    return losses


def _latency(path: Path, runs: int, warmup: int, seed: int) -> dict[str, Any]:
    stats = summarise(measure(OnnxScorer(path), runs, warmup, seed))
    return {"p50_ms": stats["p50_ms"], "p95_ms": stats["p95_ms"], "mean_ms": stats["mean_ms"]}


def run(
    out_dir: Path,
    label: str,
    n_subjects: int = 60,
    seed: int = 0,
    epochs: int = 6,
    image_size: int = 64,
    n_calibration: int = 64,
    runs: int = 100,
    warmup: int = 10,
    n_boot: int = 100,
    model_dir: Path | None = None,
) -> Path:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        data = generate_synthetic_dataset(Path(tmp) / "a", n_subjects, seed, domain="A")
        proto = make_subject_disjoint_protocol("synthetic-a-v0", data, seed=seed)
        chips = {s: chips_from_samples(list(proto.splits[s]), image_size) for s in Split}
        model = timm.create_model(ARCH, pretrained=False, num_classes=1)
        losses = train_briefly(model, chips[Split.TRAIN], epochs, seed)

        mdir = model_dir or Path(tmp) / "models"
        fp32 = export_to_onnx(model, mdir / "pad_fp32.onnx", image_size, f"synthetic-smoke-{label}")
        parity = verify_parity(model, fp32, image_size, seed=seed)
        calib = calibration_inputs(list(chips[Split.TRAIN].chips)[:n_calibration])
        quant = quantize_int8(fp32, mdir / "pad_int8.onnx", calib)
        comparison = compare_fp32_int8(
            fp32,
            quant.path,
            chips[Split.VAL],
            chips[Split.TEST],
            protocol_name=proto.name,
            protocol_manifest_hash=proto.manifest_hash(),
            model_name=f"timm-{ARCH}",
            data_kind=DATA_KIND,
            n_boot=n_boot,
            bootstrap_seed=seed,
        )
        latency = {
            "fp32": _latency(fp32, runs, warmup, seed),
            "int8": _latency(quant.path, runs, warmup, seed),
            "batch_size": 1,
            "runs": runs,
            "warmup_runs_excluded": warmup,
            "measured": "normalisation + onnxruntime CPU inference on one chip",
            "threads_torch": 1,
        }
        latency["int8_over_fp32_p50"] = latency["int8"]["p50_ms"] / latency["fp32"]["p50_ms"]

    report: dict[str, Any] = {
        "kind": "quantization_comparison",
        "label": label,
        "data_kind": DATA_KIND,
        "note": NOTE,
        "model": {
            "arch": ARCH,
            "image_size": image_size,
            "init": "random" if epochs <= 0 else f"trained {epochs} epoch(s) on synthetic train",
            "train_loss_per_epoch": losses,
        },
        "seed": seed,
        "onnx_parity": asdict(parity),
        "quantization": quant.as_dict(),
        "calibration": {"n_images": len(calib), "source": "synthetic train split only"},
        "file_sizes_bytes": {"fp32": quant.fp32_bytes, "int8": quant.int8_bytes},
        "latency_cpu": latency,
        "comparison": comparison,
    }
    out = out_dir / f"quantization_{label}.json"
    write_results_json(out, report)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--label", default="smoke")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-subjects", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6, help="0 = random init")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--calibration", type=int, default=64)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--model-dir", type=Path, default=None, help="keep the ONNX files here")
    args = ap.parse_args(argv)
    if args.runs < 1 or args.warmup < 0 or args.calibration < 1:
        ap.error("--runs and --calibration must be >= 1 and --warmup >= 0")
    out = run(
        args.out_dir,
        args.label,
        args.n_subjects,
        args.seed,
        args.epochs,
        args.image_size,
        args.calibration,
        args.runs,
        args.warmup,
        args.n_boot,
        args.model_dir,
    )
    print(f"wrote {out} (data_kind={DATA_KIND})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
