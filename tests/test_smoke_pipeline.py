import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from smoke_pipeline import run


def test_smoke_pipeline_writes_labelled_results(tmp_path: Path) -> None:
    paths = run(tmp_path, n_subjects=12, seed=0, mode="fft", n_boot=20)
    assert len(paths) == 2
    for p in paths:
        data = json.loads(p.read_text())
        assert data["data_kind"] == "synthetic_smoke"
        assert "environment" in data
