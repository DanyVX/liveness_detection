import json
import logging
from pathlib import Path

import numpy as np
import pytest

from liveness.data import Label, Sample, Split, make_subject_disjoint_protocol
from liveness.models import ClassicalPAD


@pytest.mark.parametrize("mode", ["lbp", "fft", "lbp+fft"])
def test_fit_score_save_load_roundtrip(
    synthetic_samples: list[Sample], tmp_path: Path, mode: str
) -> None:
    proto = make_subject_disjoint_protocol("t", synthetic_samples)
    train, test = proto.splits[Split.TRAIN], proto.splits[Split.TEST]
    model = ClassicalPAD(mode).fit(train)  # type: ignore[arg-type]
    scores = model.decision_scores(test)
    assert scores.shape == (len(test),) and np.isfinite(scores).all()
    model.save(tmp_path / "m.joblib")
    reloaded = ClassicalPAD.load(tmp_path / "m.joblib")
    assert np.array_equal(reloaded.decision_scores(test), scores)


def test_deterministic_refit(synthetic_samples: list[Sample]) -> None:
    a = ClassicalPAD("fft").fit(synthetic_samples).decision_scores(synthetic_samples)
    b = ClassicalPAD("fft").fit(synthetic_samples).decision_scores(synthetic_samples)
    assert np.array_equal(a, b)


def test_higher_score_means_bona_fide_on_training_data(synthetic_samples: list[Sample]) -> None:
    model = ClassicalPAD("lbp+fft").fit(synthetic_samples)
    scores = model.decision_scores(synthetic_samples)
    labels = np.array([int(s.label) for s in synthetic_samples])
    assert scores[labels == Label.BONA_FIDE].mean() > scores[labels == Label.ATTACK].mean()


def test_face_fallback_is_logged_and_counted(
    synthetic_samples: list[Sample],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("liveness.features.preprocess.detect_face", lambda img: None)
    model = ClassicalPAD("fft")
    with caplog.at_level(logging.WARNING, logger="liveness.features.preprocess"):
        model.extract(synthetic_samples[0])
    assert model.n_fallback == 1
    assert any("falling back to centre crop" in r.message for r in caplog.records)


def test_errors(synthetic_samples: list[Sample], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="feature_mode"):
        ClassicalPAD("nope")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="not fitted"):
        ClassicalPAD().decision_scores(synthetic_samples)
    bona = [s for s in synthetic_samples if s.label is Label.BONA_FIDE]
    with pytest.raises(ValueError, match="both"):
        ClassicalPAD().fit(bona)
    import joblib

    joblib.dump({"not": "a model"}, tmp_path / "x.joblib")
    with pytest.raises(TypeError):
        ClassicalPAD.load(tmp_path / "x.joblib")


def test_bench_script_writes_scores_labelled_synthetic(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_classical", Path(__file__).parents[1] / "bench" / "run_classical.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(["--mode", "lbp", "--out-dir", str(tmp_path), "--n-subjects", "8"])
    out = json.loads((tmp_path / "classical_lbp_scores.json").read_text())
    assert out["data_kind"] == "synthetic_smoke"
    assert set(out["splits"]) == {"val", "test"}
    test = out["splits"]["test"]
    assert len(test["scores"]) == len(test["labels"]) == len(test["subject_keys"])
    assert "metrics" not in out
