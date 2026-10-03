from pathlib import Path

import numpy as np

from liveness.data.synthetic import generate_synthetic_dataset
from liveness.models.cnn import (
    CNNConfig,
    FaceDataset,
    balanced_sampler,
    build_model,
    logits_to_scores,
)


def test_face_dataset_and_model_contract(tmp_path: Path) -> None:
    samples = generate_synthetic_dataset(tmp_path, n_subjects=3, seed=0)
    config = CNNConfig(image_size=64)
    image, label = FaceDataset(samples, config, training=False)[0]
    assert image.shape == (3, 64, 64) and label.item() in (0.0, 1.0)
    logits = build_model(config)(image.unsqueeze(0))
    assert logits.shape == (1, 1)
    assert 0 <= logits_to_scores(logits).item() <= 1


def test_balanced_sampler_exposes_both_classes(tmp_path: Path) -> None:
    samples = generate_synthetic_dataset(tmp_path, n_subjects=3, seed=0)
    sampler = balanced_sampler(samples, seed=1)
    drawn = list(sampler)
    labels = np.asarray([int(samples[i].label) for i in drawn])
    assert set(labels) == {0, 1}
    assert len(drawn) == len(samples)


def test_cnn_config_rejects_unsafe_geometry() -> None:
    for kwargs in ({"image_size": 31}, {"face_margin": -0.1}, {"face_margin": 1.1}):
        try:
            CNNConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid config was accepted")
