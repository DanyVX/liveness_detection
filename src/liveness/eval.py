"""Protocol evaluation: thresholds are chosen on validation only, then frozen for test.

Result dicts are plain JSON-able structures (see `write_results_json`). A result whose
``data_kind`` is ``"synthetic_smoke"`` exercises the pipeline only and is not a performance
claim.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from liveness import metrics
from liveness.env_info import collect_environment

SCHEMA_VERSION = 1
DATA_KINDS = ("synthetic_smoke", "real")
MIN_CLASS_SAMPLES = 30
MAX_CURVE_POINTS = 400
_F = NDArray[np.float64]


@dataclass(frozen=True)
class ScoredSplit:
    """Model scores for one split. `labels`: 1 bona fide, 0 attack; bona fide type is "none"."""

    scores: _F
    labels: NDArray[np.int64]
    attack_types: NDArray[np.str_]
    subject_keys: Sequence[object]
    sample_ids: Sequence[str]

    def __post_init__(self) -> None:
        n = len(self.scores)
        for name in ("labels", "attack_types", "subject_keys", "sample_ids"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, expected {n}")
        # Fail early on NaN / empty / single-class splits.
        metrics.auc(self.scores, self.labels)

    @property
    def n_bona_fide(self) -> int:
        return int(np.sum(np.asarray(self.labels) == 1))

    @property
    def n_attack(self) -> int:
        return int(np.sum(np.asarray(self.labels) == 0))


def _split_counts(split: ScoredSplit) -> dict[str, int]:
    return {"bona_fide": split.n_bona_fide, "attack": split.n_attack}


def _op_name(target: float) -> str:
    return f"bpcer_{target:g}"


def _fit_thresholds(val: ScoredSplit, target_bpcers: Sequence[float]) -> dict[str, float]:
    """The only place thresholds are derived; it sees validation data and nothing else."""
    _, eer_thr = metrics.eer(val.scores, val.labels)
    out = {"eer_val": eer_thr}
    bona = np.asarray(val.scores)[np.asarray(val.labels) == 1]
    for target in target_bpcers:
        out[_op_name(target)] = metrics.threshold_at_bpcer(bona, target)
    return out


def _at_threshold(split: ScoredSplit, threshold: float, n_boot: int, seed: int) -> dict[str, Any]:
    scores, labels, types = split.scores, split.labels, split.attack_types
    res = metrics.acer(scores, labels, types, threshold)
    attack = labels == 0
    apcer = metrics.apcer(scores[attack], types[attack], threshold)
    ci = metrics.bootstrap_ci(
        lambda s, lab, t: metrics.acer(s, lab, t, threshold)["acer"],
        scores,
        labels,
        types,
        split.subject_keys,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "threshold": float(threshold),
        "apcer": apcer,
        "bpcer": res["bpcer"],
        "acer": res["acer"],
        "acer_mean": res["acer_mean"],
        "hter": metrics.hter(scores, labels, threshold),
        "acer_ci": ci,
        "confusion": metrics.confusion_matrix(scores, labels, threshold),
    }


def _evaluate_split(
    split: ScoredSplit, thresholds: dict[str, float], n_boot: int, seed: int
) -> dict[str, Any]:
    """Apply frozen thresholds to a split. Never derives a threshold itself."""
    e, e_thr = metrics.eer(split.scores, split.labels)

    def stat_eer(s: _F, lab: NDArray[np.int64], _t: NDArray[np.str_]) -> float:
        return metrics.eer(s, lab)[0]

    def stat_auc(s: _F, lab: NDArray[np.int64], _t: NDArray[np.str_]) -> float:
        return metrics.auc(s, lab)

    args = (split.scores, split.labels, split.attack_types, split.subject_keys)
    curve = metrics.roc_curve(split.scores, split.labels)
    keep = np.unique(np.linspace(0, len(curve["fpr"]) - 1, MAX_CURVE_POINTS).astype(int))
    small = split.n_bona_fide < MIN_CLASS_SAMPLES or split.n_attack < MIN_CLASS_SAMPLES
    return {
        "n": _split_counts(split),
        "small_sample_warning": small,
        "threshold_free": {
            "auc": metrics.bootstrap_ci(stat_auc, *args, n_boot=n_boot, seed=seed),
            "eer": metrics.bootstrap_ci(stat_eer, *args, n_boot=n_boot, seed=seed),
            "eer_own_threshold_diagnostic": e_thr,
            "eer_point": e,
        },
        "roc": {
            "fpr": [curve["fpr"][i] for i in keep],
            "tpr": [curve["tpr"][i] for i in keep],
        },
        "operating_points": {
            name: _at_threshold(split, thr, n_boot, seed) for name, thr in thresholds.items()
        },
    }


def _header(
    kind: str,
    protocol_name: str,
    protocol_manifest_hash: str,
    model_name: str,
    data_kind: str,
    seeds_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if data_kind not in DATA_KINDS:
        raise ValueError(f"data_kind must be one of {DATA_KINDS}, got {data_kind!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "data_kind": data_kind,
        "model": model_name,
        "protocol": {"name": protocol_name, "manifest_hash": protocol_manifest_hash},
        "seeds_info": seeds_info,
        "score_convention": "higher = more bona fide; accept when score >= threshold",
    }


def evaluate_protocol(
    val: ScoredSplit,
    test: ScoredSplit,
    target_bpcers: Sequence[float] = (0.01, 0.05),
    *,
    protocol_name: str,
    protocol_manifest_hash: str,
    model_name: str,
    data_kind: str,
    seeds_info: dict[str, Any] | None = None,
    n_boot: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Pick thresholds on `val` only, apply them unchanged to `test`."""
    result = _header(
        "single", protocol_name, protocol_manifest_hash, model_name, data_kind, seeds_info
    )
    thresholds = _fit_thresholds(val, target_bpcers)
    result["thresholds"] = {"selected_on": "val_only", **thresholds}
    result["val_n"] = _split_counts(val)
    result["val_small_sample_warning"] = (
        val.n_bona_fide < MIN_CLASS_SAMPLES or val.n_attack < MIN_CLASS_SAMPLES
    )
    result["test"] = _evaluate_split(test, thresholds, n_boot, bootstrap_seed)
    return result


def cross_dataset_report(
    source_val: ScoredSplit,
    source_test: ScoredSplit,
    target_test: ScoredSplit,
    target_bpcers: Sequence[float] = (0.01, 0.05),
    *,
    protocol_name: str,
    protocol_manifest_hash: str,
    model_name: str,
    data_kind: str,
    target_name: str = "target",
    seeds_info: dict[str, Any] | None = None,
    n_boot: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Source-domain thresholds applied unchanged to a target domain.

    The reported target APCER/BPCER always use the source-val thresholds. The threshold the
    target's own EER would pick is included under ``threshold_shift_diagnostic`` only to
    quantify calibration shift; it is never used for any reported error rate.
    """
    source = evaluate_protocol(
        source_val,
        source_test,
        target_bpcers,
        protocol_name=protocol_name,
        protocol_manifest_hash=protocol_manifest_hash,
        model_name=model_name,
        data_kind=data_kind,
        seeds_info=seeds_info,
        n_boot=n_boot,
        bootstrap_seed=bootstrap_seed,
    )
    frozen = {k: v for k, v in source["thresholds"].items() if k != "selected_on"}
    target = _evaluate_split(target_test, frozen, n_boot, bootstrap_seed)
    target["name"] = target_name
    target["thresholds_from"] = "source_val"
    _, own = metrics.eer(target_test.scores, target_test.labels)
    result = _header(
        "cross_dataset", protocol_name, protocol_manifest_hash, model_name, data_kind, seeds_info
    )
    result["source"] = source
    result["target"] = target
    result["threshold_shift_diagnostic"] = {
        "diagnostic_only": True,
        "note": "target-test EER threshold uses target labels; never used for reported errors",
        "source_val_eer_threshold": frozen["eer_val"],
        "target_test_eer_threshold": own,
        "shift": own - frozen["eer_val"],
    }
    return result


def _agg(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {k: _agg([v[k] for v in values]) for k in first}
    if isinstance(first, bool) or not isinstance(first, int | float):
        return first
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def aggregate_seeds(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean and sample std (ddof=1; 0 for a single run) of every numeric leaf across seeds."""
    if not results:
        raise ValueError("no results to aggregate")
    sig = {(r["kind"], r["data_kind"], r["model"], r["protocol"]["name"]) for r in results}
    if len(sig) != 1:
        raise ValueError(f"results differ in kind/data_kind/model/protocol: {sorted(sig)}")
    body_keys = [
        k for k in ("test", "source", "target", "threshold_shift_diagnostic") if k in results[0]
    ]
    first = results[0]
    out = _header(
        "multi_seed",
        first["protocol"]["name"],
        first["protocol"]["manifest_hash"],
        first["model"],
        first["data_kind"],
        None,
    )
    out["base_kind"] = first["kind"]
    out["n_seeds"] = len(results)
    out["seeds_info"] = [r.get("seeds_info") for r in results]
    for key in body_keys:
        out[key] = _agg([copy.deepcopy(r[key]) for r in results])
    return out


def write_results_json(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Write result plus environment info; validates a strict JSON round trip (no NaN)."""
    payload = dict(result)
    payload["environment"] = collect_environment()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if json.loads(text) != json.loads(json.dumps(payload, sort_keys=True, allow_nan=False)):
        raise ValueError("results did not survive a JSON round trip")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return payload
