import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from liveness.data import Sample
from liveness.env_info import collect_environment
from liveness.eval import aggregate_seeds, cross_dataset_report, evaluate_protocol
from liveness.reporting import (
    SMOKE_MARK,
    TBD,
    failure_grid,
    plot_roc_det,
    render_results_table,
)
from test_eval import KW, make_split


def _write(path: Path, result: dict) -> None:  # type: ignore[type-arg]
    path.write_text(json.dumps(result))


def test_empty_results_dir_is_tbd(tmp_path: Path) -> None:
    assert TBD in render_results_table(tmp_path)
    assert TBD in render_results_table(tmp_path / "missing")


def test_expected_but_absent_is_tbd(tmp_path: Path) -> None:
    table = render_results_table(tmp_path, expected=[("cnn", "replay_attack")])
    assert "cnn" in table and TBD in table


def test_smoke_rows_are_marked_and_small_sample_flagged(tmp_path: Path) -> None:
    r = evaluate_protocol(make_split(1), make_split(2, 5), **KW)
    _write(tmp_path / "a.json", r)
    table = render_results_table(tmp_path, expected=[("m", "p")])
    assert SMOKE_MARK in table and "small sample" in table
    assert "bpcer_0.01" in table and "TBD (not" not in table.split("\n", 2)[2]


def test_cross_and_multiseed_rows(tmp_path: Path) -> None:
    r = cross_dataset_report(make_split(1), make_split(2), make_split(3), **KW)
    _write(tmp_path / "c.json", r)
    _write(tmp_path / "m.json", aggregate_seeds([r, r]))
    _write(tmp_path / "bad.json", {"oops": 1})
    (tmp_path / "junk.json").write_text("{not json")
    table = render_results_table(tmp_path)
    assert "cross-domain target" in table and "mean ±" in table
    assert table.count("unreadable") == 2
    assert table.count(SMOKE_MARK) >= 4


def test_real_result_has_no_smoke_marker(tmp_path: Path) -> None:
    r = evaluate_protocol(make_split(1), make_split(2), **{**KW, "data_kind": "real"})
    _write(tmp_path / "r.json", r)
    assert SMOKE_MARK not in render_results_table(tmp_path)


def test_plot_roc_det_writes_pngs(tmp_path: Path) -> None:
    r = evaluate_protocol(make_split(1), make_split(2), **KW)
    plot_roc_det({"m": r["test"]["roc"]}, tmp_path / "p" / "roc.png", tmp_path / "p" / "det.png")
    for name in ("roc.png", "det.png"):
        assert Image.open(tmp_path / "p" / name).size[0] > 100
    with pytest.raises(ValueError, match="no curves"):
        plot_roc_det({}, tmp_path / "a.png", tmp_path / "b.png")


def test_failure_grid_both_kinds(synthetic_samples: list[Sample], tmp_path: Path) -> None:
    from liveness.data import Label

    labels = np.array([int(s.label) for s in synthetic_samples])
    scores = np.where(labels == 1, 0.2, 0.9)  # everything is wrong at threshold 0.5
    fa = failure_grid(synthetic_samples, scores, 0.5, tmp_path / "fa.png", "false_accept")
    fr = failure_grid(synthetic_samples, scores, 0.5, tmp_path / "fr.png", "false_reject")
    assert fa == fr == 16
    assert Image.open(tmp_path / "fa.png").size == (4 * 128, 4 * 128)
    assert sum(s.label is Label.ATTACK for s in synthetic_samples) > 16


def test_failure_grid_orders_worst_first_and_handles_none(
    synthetic_samples: list[Sample], tmp_path: Path
) -> None:
    perfect = np.array([float(s.label) for s in synthetic_samples])
    assert failure_grid(synthetic_samples, perfect, 0.5, tmp_path / "n.png") == 0
    assert (tmp_path / "n.png").exists()
    few = failure_grid(synthetic_samples, perfect, 0.5, tmp_path / "x.png", max_items=3)
    assert few == 0


def test_failure_grid_errors(synthetic_samples: list[Sample], tmp_path: Path) -> None:
    n = len(synthetic_samples)
    with pytest.raises(ValueError, match="kind"):
        failure_grid(synthetic_samples, np.zeros(n), 0.5, tmp_path / "a.png", "nope")
    with pytest.raises(ValueError, match="length"):
        failure_grid(synthetic_samples, np.zeros(n - 1), 0.5, tmp_path / "a.png")
    with pytest.raises(ValueError, match="NaN"):
        failure_grid(synthetic_samples, np.full(n, np.nan), 0.5, tmp_path / "a.png")


def test_collect_environment_without_git(tmp_path: Path) -> None:
    env = collect_environment(tmp_path)
    assert env["git_commit"] is None and env["git_dirty"] is None
    assert env["gpu"] == "none" or isinstance(env["gpu"], str)
    assert isinstance(env["libraries"], dict) and "numpy" in env["libraries"]


def test_render_results_script(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "render_results", Path(__file__).parents[1] / "scripts" / "render_results.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = tmp_path / "results"
    res.mkdir()
    r = cross_dataset_report(make_split(1), make_split(2), make_split(3), **KW)
    _write(res / "c.json", r)
    _write(res / "s.json", evaluate_protocol(make_split(1), make_split(2), **KW))
    _write(res / "bad.json", {"x": 1})
    out = tmp_path / "docs" / "RESULTS.generated.md"
    code = mod.main(
        ["--results-dir", str(res), "--out", str(out), "--plots-dir", str(res / "plots")]
    )
    assert code == 0
    assert SMOKE_MARK in out.read_text()
    assert (res / "plots" / "roc.png").exists() and (res / "plots" / "det.png").exists()
    with pytest.raises(SystemExit):
        mod.main(["--results-dir", str(res), "--out", str(tmp_path / "RESULTS.md")])
