"""Plots, markdown results tables and failure grids built from saved results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from liveness.data.base import Label, Sample

TBD = "TBD (not yet measured)"
SMOKE_MARK = "SYNTHETIC SMOKE (not a real result)"
_EPS = 1e-4


def plot_roc_det(
    curves: Mapping[str, Mapping[str, Sequence[float]]], roc_path: Path, det_path: Path
) -> None:
    """Write ROC and DET PNGs. `curves[name]` holds `fpr` and `tpr` (as in result JSONs)."""
    if not curves:
        raise ValueError("no curves to plot")
    for path in (roc_path, det_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, c in curves.items():
        ax.plot(c["fpr"], c["tpr"], label=name)
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)
    ax.set(xlabel="APCER (attacks accepted)", ylabel="1 - BPCER (bona fide accepted)")
    ax.set(title="ROC", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, c in curves.items():
        far = np.clip(np.asarray(c["fpr"], dtype=np.float64), _EPS, 1)
        frr = np.clip(1 - np.asarray(c["tpr"], dtype=np.float64), _EPS, 1)
        ax.plot(far, frr, label=name)
    ax.set(xscale="log", yscale="log", xlabel="APCER", ylabel="BPCER", title="DET")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(det_path, dpi=120)
    plt.close(fig)


def _num(v: Any) -> str:
    if isinstance(v, int | float) and not isinstance(v, bool):
        return f"{v:.3f}"
    if isinstance(v, dict) and "mean" in v:
        return f"{v['mean']:.3f} ± {v['std']:.3f}"
    return "n/a"


def _with_ci(ci: Any) -> str:
    """Cell for a bootstrap-CI dict (single run) or its seed-aggregated form."""
    if not isinstance(ci, dict) or "estimate" not in ci:
        return "n/a"
    lo, hi = ci.get("lower"), ci.get("upper")
    base = _num(ci["estimate"])
    if isinstance(lo, int | float) and isinstance(hi, int | float):
        return f"{base} [{lo:.3f}, {hi:.3f}]"
    return base


def _by_type(apcer: Mapping[str, Any]) -> str:
    parts = [f"{k} {_num(v)}" for k, v in apcer.items() if k not in ("max", "mean")]
    return ", ".join(parts) or "n/a"


def _blocks(result: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    if result.get("kind") == "single" or result.get("base_kind") == "single":
        return [("test (val thresholds)", result["test"])]
    return [
        ("source test (source-val thresholds)", result["source"]["test"]),
        ("cross-domain target (source-val thresholds)", result["target"]),
    ]


def _rows(result: Mapping[str, Any]) -> list[str]:
    smoke = result.get("data_kind") == "synthetic_smoke"
    rows = []
    for label, block in _blocks(result):
        auc = _with_ci(block["threshold_free"]["auc"])
        eer = _with_ci(block["threshold_free"]["eer"])
        small = block.get("small_sample_warning")
        n = block["n"]
        notes = [SMOKE_MARK] if smoke else []
        if small is True:
            notes.append("small sample (<30 per class): CIs unreliable")
        if result.get("kind") == "multi_seed":
            notes.append(f"{result['n_seeds']} seeds, mean ± std")
        for op, m in block["operating_points"].items():
            rows.append(
                f"| {result['model']} | {result['protocol']['name']} | {label} | {op} "
                f"| {_num(m['apcer']['max'])} ({_by_type(m['apcer'])}) | {_num(m['bpcer'])} "
                f"| {_with_ci(m['acer_ci'])} | {eer} | {auc} "
                f"| {_num(n['bona_fide'])}/{_num(n['attack'])} | {'; '.join(notes)} |"
            )
    return rows


_HEADER = (
    "| Model | Protocol | Evaluation | Operating point | APCER worst (by type) | BPCER "
    "| ACER (worst-type APCER) | EER | AUC | n bona fide/attack | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def render_results_table(results_dir: Path, expected: Sequence[tuple[str, str]] = ()) -> str:
    """Markdown table from `results_dir/*.json`; `expected` (model, protocol) pairs with no
    result file are listed as TBD."""
    rows: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(results_dir.glob("*.json")) if results_dir.is_dir() else []:
        try:
            result = json.loads(path.read_text())
            rows.extend(_rows(result))
            seen.add((result["model"], result["protocol"]["name"]))
        except (ValueError, KeyError, TypeError):
            rows.append(f"| {path.name} | unreadable result file | | | | | | | | | {TBD} |")
    for model, protocol in expected:
        if (model, protocol) not in seen:
            rows.append(f"| {model} | {protocol} | | | {TBD} | {TBD} | {TBD} | {TBD} | {TBD} | | |")
    if not rows:
        rows.append(f"| {TBD} | | | | | | | | | | |")
    return _HEADER + "\n" + "\n".join(rows) + "\n"


def failure_grid(
    samples: Sequence[Sample],
    scores: Sequence[float],
    threshold: float,
    path: Path,
    kind: str = "false_accept",
    max_items: int = 16,
    thumb: int = 128,
    columns: int = 4,
) -> int:
    """Write a grid of the worst failures (highest-scoring attacks for ``false_accept``,
    lowest-scoring bona fide for ``false_reject``) and return how many were drawn."""
    if kind not in ("false_accept", "false_reject"):
        raise ValueError(f"kind must be 'false_accept' or 'false_reject', got {kind!r}")
    if len(samples) != len(scores):
        raise ValueError("samples and scores differ in length")
    sc = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(sc)):
        raise ValueError("scores contain NaN or infinite values")
    if kind == "false_accept":
        cand = [i for i, s in enumerate(samples) if s.label is Label.ATTACK and sc[i] >= threshold]
        cand.sort(key=lambda i: -sc[i])
    else:
        cand = [
            i for i, s in enumerate(samples) if s.label is Label.BONA_FIDE and sc[i] < threshold
        ]
        cand.sort(key=lambda i: sc[i])
    cand = cand[:max_items]
    cols = max(1, min(columns, len(cand) or 1))
    rows = max(1, -(-len(cand) // cols))
    canvas = np.full((rows * thumb, cols * thumb, 3), 255, dtype=np.uint8)
    if not cand:
        cv2.putText(canvas, "no failures", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for k, i in enumerate(cand):
        img = Image.open(samples[i].path).convert("RGB").resize((thumb, thumb))
        tile = np.array(img, dtype=np.uint8)
        text = f"{sc[i]:.2f} {samples[i].attack_type.value}"
        cv2.putText(tile, text, (3, thumb - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        r, c = divmod(k, cols)
        canvas[r * thumb : (r + 1) * thumb, c * thumb : (c + 1) * thumb] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)
    return len(cand)
