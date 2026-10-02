import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from quantization import run


def test_tiny_run_writes_labelled_json(tmp_path: Path) -> None:
    out = run(
        tmp_path,
        "unit",
        n_subjects=16,
        epochs=1,
        n_calibration=8,
        runs=3,
        warmup=1,
        n_boot=10,
        model_dir=tmp_path / "models",
    )
    assert out == tmp_path / "quantization_unit.json"
    data = json.loads(out.read_text())
    assert data["data_kind"] == "synthetic_smoke"
    assert data["comparison"]["data_kind"] == "synthetic_smoke"
    assert "not a performance claim" in data["note"] or "say nothing" in data["note"]
    assert "environment" in data and "python" in data["environment"]
    assert data["onnx_parity"]["passed"] is True
    assert data["quantization"]["mode"] in {"static", "dynamic"}
    sizes = data["file_sizes_bytes"]
    assert 0 < sizes["int8"] < sizes["fp32"]
    for kind in ("fp32", "int8"):
        assert data["latency_cpu"][kind]["p50_ms"] > 0
        assert data["latency_cpu"][kind]["p95_ms"] >= data["latency_cpu"][kind]["p50_ms"]
    assert set(data["comparison"]["thresholds"]) == {"fp32", "int8"}
    assert (tmp_path / "models" / "pad_int8.onnx").is_file()


def test_random_init_is_labelled(tmp_path: Path) -> None:
    out = run(
        tmp_path, "rand", n_subjects=16, epochs=0, n_calibration=4, runs=2, warmup=0, n_boot=5
    )
    data = json.loads(out.read_text())
    assert data["model"]["init"] == "random"
    assert data["model"]["train_loss_per_epoch"] == []
