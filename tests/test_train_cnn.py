from pathlib import Path

import pytest
import torch
from torch import nn

import liveness.train as training
from liveness.data import make_subject_disjoint_protocol
from liveness.data.synthetic import generate_synthetic_dataset
from liveness.models.cnn import CNNConfig
from liveness.train import TrainConfig, fit


class TinyPAD(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@pytest.fixture
def protocol(tmp_path: Path):  # type: ignore[no-untyped-def]
    samples = generate_synthetic_dataset(tmp_path / "data", n_subjects=6, seed=3, size=40)
    return make_subject_disjoint_protocol("tiny", samples, seed=2)


def test_fit_writes_best_and_last_checkpoint(
    protocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(training, "build_model", lambda _config: TinyPAD())
    run = fit(
        protocol,
        CNNConfig(image_size=32),
        TrainConfig(epochs=2, patience=2, batch_size=8, seed=4),
        tmp_path / "checkpoints",
    )
    assert len(run.history) == 2
    assert 0 <= run.best_val_acer <= 1
    assert (tmp_path / "checkpoints" / "best.pt").is_file()
    last = torch.load(tmp_path / "checkpoints" / "last.pt", weights_only=False)
    assert last["arch"] == "mobilenetv3_small_050"
    assert last["history"][-1]["epoch"] == 1


def test_resume_matches_uninterrupted_curve(
    protocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    monkeypatch.setattr(training, "build_model", lambda _config: TinyPAD())
    model_config = CNNConfig(image_size=32)
    full = fit(
        protocol,
        model_config,
        TrainConfig(epochs=2, patience=3, batch_size=8, seed=9),
        tmp_path / "full",
    )
    fit(
        protocol,
        model_config,
        TrainConfig(epochs=1, patience=3, batch_size=8, seed=9),
        tmp_path / "resume",
    )
    resumed = fit(
        protocol,
        model_config,
        TrainConfig(epochs=2, patience=3, batch_size=8, seed=9),
        tmp_path / "resume",
        resume_from=tmp_path / "resume" / "last.pt",
    )
    assert resumed.history == full.history
    for expected, actual in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_train_config_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=0)
