import json
from pathlib import Path

import numpy as np
import pytest

from liveness import metrics
from liveness.eval import (
    ScoredSplit,
    aggregate_seeds,
    cross_dataset_report,
    evaluate_protocol,
    write_results_json,
)

KW = {
    "protocol_name": "p",
    "protocol_manifest_hash": "abc",
    "model_name": "m",
    "data_kind": "synthetic_smoke",
    "n_boot": 30,
}


def make_split(
    seed: int, n_subjects: int = 20, shift: float = 0.0, prefix: str = "s"
) -> ScoredSplit:
    rng = np.random.default_rng(seed)
    scores, labels, types, subj, ids = [], [], [], [], []
    for i in range(n_subjects):
        for tag, label, kind, mu in (
            ("bf0", 1, "none", 1.0),
            ("bf1", 1, "none", 1.0),
            ("p", 0, "print", 0.0),
            ("r", 0, "replay", 0.4),
        ):
            scores.append(rng.normal(mu + shift, 0.5))
            labels.append(label)
            types.append(kind)
            subj.append(("d", f"{prefix}{i}"))
            ids.append(f"{prefix}{i}_{tag}")
    return ScoredSplit(np.array(scores), np.array(labels), np.array(types), subj, ids)


def test_scored_split_validates() -> None:
    with pytest.raises(ValueError, match="length"):
        ScoredSplit(np.array([0.1, 0.2]), np.array([1, 0]), np.array(["none"]), [1, 2], ["a", "b"])
    with pytest.raises(ValueError, match="NaN"):
        ScoredSplit(
            np.array([np.nan, 0.2]), np.array([1, 0]), np.array(["none", "p"]), [1, 2], ["a", "b"]
        )
    with pytest.raises(ValueError, match="single class"):
        ScoredSplit(np.array([0.1, 0.2]), np.array([1, 1]), np.array(["none"] * 2), [1, 2], "ab")


def test_thresholds_chosen_from_val_only() -> None:
    val, test_a, test_b = make_split(1), make_split(2), make_split(3, shift=-2.0)
    ra = evaluate_protocol(val, test_a, **KW)
    rb = evaluate_protocol(val, test_b, **KW)
    assert ra["thresholds"] == rb["thresholds"]
    assert ra["thresholds"]["selected_on"] == "val_only"
    assert ra["test"]["operating_points"]["eer_val"]["threshold"] == ra["thresholds"]["eer_val"]
    rc = evaluate_protocol(make_split(9), test_a, **KW)
    assert rc["thresholds"]["eer_val"] != ra["thresholds"]["eer_val"]


def test_reported_metrics_match_direct_computation() -> None:
    val, test = make_split(1), make_split(2)
    r = evaluate_protocol(val, test, target_bpcers=(0.05,), **KW)
    thr = metrics.threshold_at_bpcer(val.scores[val.labels == 1], 0.05)
    op = r["test"]["operating_points"]["bpcer_0.05"]
    direct = metrics.acer(test.scores, test.labels, test.attack_types, thr)
    assert op["acer"] == direct["acer"] and op["bpcer"] == direct["bpcer"]
    assert set(op["apcer"]) == {"print", "replay", "max", "mean"}
    assert r["test"]["threshold_free"]["auc"]["estimate"] == metrics.auc(test.scores, test.labels)
    assert r["test"]["n"] == {"bona_fide": 40, "attack": 40}
    assert r["data_kind"] == "synthetic_smoke"


def test_small_sample_warning() -> None:
    big, small = make_split(1, 20), make_split(2, 5)
    assert evaluate_protocol(big, small, **KW)["test"]["small_sample_warning"] is True
    assert evaluate_protocol(big, big, **KW)["test"]["small_sample_warning"] is False
    assert evaluate_protocol(small, big, **KW)["val_small_sample_warning"] is True


def test_bad_data_kind_rejected() -> None:
    with pytest.raises(ValueError, match="data_kind"):
        evaluate_protocol(make_split(1), make_split(2), **{**KW, "data_kind": "synthetic"})


def test_cross_dataset_uses_source_threshold_and_labels_shift_diagnostic() -> None:
    sv, st_, tt = make_split(1), make_split(2), make_split(3, shift=-0.8)
    r = cross_dataset_report(sv, st_, tt, target_name="B", **KW)
    thr = r["source"]["thresholds"]["eer_val"]
    tgt_op = r["target"]["operating_points"]["eer_val"]
    assert tgt_op["threshold"] == thr
    assert tgt_op["bpcer"] == metrics.bpcer(tt.scores[tt.labels == 1], thr)
    d = r["threshold_shift_diagnostic"]
    assert d["diagnostic_only"] is True
    assert d["shift"] == pytest.approx(d["target_test_eer_threshold"] - thr)
    assert d["shift"] < 0  # target scores are lower, so its own threshold is lower
    assert r["target"]["thresholds_from"] == "source_val"


def test_target_scores_do_not_change_source_thresholds() -> None:
    sv, st_ = make_split(1), make_split(2)
    a = cross_dataset_report(sv, st_, make_split(3), **KW)
    b = cross_dataset_report(sv, st_, make_split(4, shift=-3), **KW)
    assert a["source"]["thresholds"] == b["source"]["thresholds"]


def test_results_json_contains_env_and_commit(tmp_path: Path) -> None:
    r = evaluate_protocol(make_split(1), make_split(2), **KW)
    path = tmp_path / "out" / "r.json"
    write_results_json(path, r)
    loaded = json.loads(path.read_text())
    env = loaded["environment"]
    assert "git_commit" in env and "git_dirty" in env and "timestamp_utc" in env
    assert {"python", "platform", "cpu_count", "ram_bytes", "gpu", "libraries"} <= set(env)
    assert loaded["data_kind"] == "synthetic_smoke"


def test_write_results_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_results_json(tmp_path / "x.json", {"bad": float("nan")})


def test_aggregate_seeds_mean_std() -> None:
    results = [
        evaluate_protocol(make_split(s), make_split(s + 10), seeds_info={"seed": s}, **KW)
        for s in (1, 2, 3)
    ]
    agg = aggregate_seeds(results)
    vals = [r["test"]["operating_points"]["eer_val"]["bpcer"] for r in results]
    cell = agg["test"]["operating_points"]["eer_val"]["bpcer"]
    assert cell["mean"] == pytest.approx(np.mean(vals))
    assert cell["std"] == pytest.approx(np.std(vals, ddof=1))
    assert agg["n_seeds"] == 3 and agg["kind"] == "multi_seed"
    assert [s["seed"] for s in agg["seeds_info"]] == [1, 2, 3]


def test_aggregate_single_run_has_zero_std_and_cross_kind() -> None:
    r = cross_dataset_report(make_split(1), make_split(2), make_split(3), **KW)
    agg = aggregate_seeds([r])
    assert agg["base_kind"] == "cross_dataset"
    assert agg["target"]["operating_points"]["eer_val"]["bpcer"]["std"] == 0.0


def test_aggregate_errors() -> None:
    with pytest.raises(ValueError, match="no results"):
        aggregate_seeds([])
    a = evaluate_protocol(make_split(1), make_split(2), **KW)
    b = evaluate_protocol(make_split(1), make_split(2), **{**KW, "model_name": "other"})
    with pytest.raises(ValueError, match="differ"):
        aggregate_seeds([a, b])
