"""Latency benchmark for the ONNX scorer (normalisation + inference on a face chip).

Writes results/latency_<label>.json. Without --model a tiny hand-built model is used and the
output is labelled data_kind=synthetic_smoke: it says nothing about a real model's speed.

    uv run python bench/latency.py --label smoke
    uv run python bench/latency.py --model models/pad.onnx --label pad_v1 --runs 500
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from liveness.infer import OnnxScorer, build_tiny_onnx_model


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "git_commit": _git_commit(),
    }


def measure(scorer: OnnxScorer, runs: int, warmup: int, seed: int) -> list[float]:
    h, w = scorer.input_size
    chip = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    for _ in range(warmup):
        scorer.score_chips([chip])
    times_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        scorer.score_chips([chip])
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def summarise(times_ms: list[float]) -> dict[str, float]:
    a = np.asarray(times_ms)
    p50, p95, p99 = (float(np.percentile(a, q)) for q in (50, 95, 99))
    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": float(a.mean()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def run_provider(
    model_path: Path, providers: list[str], runs: int, warmup: int, seed: int
) -> dict[str, Any]:
    scorer = OnnxScorer(model_path, providers=providers)
    stats = summarise(measure(scorer, runs, warmup, seed))
    return {
        "providers": scorer.providers,
        "model_version": scorer.model_version,
        "input_size": list(scorer.input_size),
        **stats,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--model", type=Path, default=None, help="ONNX model; default: tiny test model")
    ap.add_argument("--label", default="smoke")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        ap.error("--runs must be >= 1 and --warmup >= 0")

    available = ort.get_available_providers()
    with tempfile.TemporaryDirectory() as tmp:
        if args.model is None:
            model_path = build_tiny_onnx_model(Path(tmp) / "tiny.onnx", model_version="tiny-smoke")
            data_kind = "synthetic_smoke"
        else:
            model_path = args.model
            data_kind = "measured_model"
        cpu = run_provider(model_path, ["CPUExecutionProvider"], args.runs, args.warmup, args.seed)
        gpu: dict[str, Any] | str = "unavailable"
        if "CUDAExecutionProvider" in available:
            gpu = run_provider(
                model_path,
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                args.runs,
                args.warmup,
                args.seed,
            )

    report = {
        "label": args.label,
        "data_kind": data_kind,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": args.runs,
        "warmup_runs_excluded": args.warmup,
        "batch_size": 1,
        "measured": "normalisation + onnxruntime inference on one face chip (no decode/detect)",
        "env": env_info(),
        "cpu": cpu,
        "gpu": gpu,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"latency_{args.label}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out} (data_kind={data_kind})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
