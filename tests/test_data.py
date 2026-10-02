from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from liveness.config import Settings, load_settings
from liveness.data import AttackType, Label, Sample
from liveness.data.registry import DatasetNotFoundError, load_dataset
from liveness.data.synthetic import generate_synthetic_dataset, load_synthetic


def test_sample_rejects_inconsistent_label() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        Sample("d", "s", Path("x.png"), Label.BONA_FIDE, AttackType.PRINT)
    with pytest.raises(ValueError, match="inconsistent"):
        Sample("d", "s", Path("x.png"), Label.ATTACK, AttackType.NONE)


def test_sample_rejects_empty_ids() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Sample("", "s", Path("x.png"), Label.ATTACK, AttackType.PRINT)


def test_synthetic_is_deterministic(tmp_path: Path) -> None:
    a = generate_synthetic_dataset(tmp_path / "a", n_subjects=3, seed=5)
    b = generate_synthetic_dataset(tmp_path / "b", n_subjects=3, seed=5)
    for sa, sb in zip(a, b, strict=True):
        assert np.array_equal(np.asarray(Image.open(sa.path)), np.asarray(Image.open(sb.path)))


def test_synthetic_roundtrip_and_attack_types(tmp_path: Path) -> None:
    samples = generate_synthetic_dataset(tmp_path, n_subjects=4, seed=0)
    assert load_synthetic(tmp_path) == samples
    types = {s.attack_type for s in samples}
    assert types == {AttackType.NONE, AttackType.PRINT, AttackType.REPLAY}
    assert all(s.path.exists() for s in samples)


def test_registry_synthetic(tmp_path: Path) -> None:
    generate_synthetic_dataset(tmp_path, n_subjects=3)
    assert len(load_dataset("synthetic", tmp_path)) == 12


def test_registry_real_dataset_missing_points_to_data_md(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match=r"DATA\.md"):
        load_dataset("replay_attack", tmp_path / "nope")


def test_registry_real_loader_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        load_dataset("casia_fasd", tmp_path)


def test_registry_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        load_dataset("nope", tmp_path)


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.seed == 1337 and s.log_level == "INFO"


def test_settings_bad_value_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVENESS_LOG_LEVEL", "LOUD")
    with pytest.raises(SystemExit, match="LIVENESS_LOG_LEVEL"):
        load_settings()
