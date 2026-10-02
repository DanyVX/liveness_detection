from pathlib import Path

import pytest

from liveness.data import Sample
from liveness.data.synthetic import generate_synthetic_dataset


@pytest.fixture
def synthetic_samples(tmp_path: Path) -> list[Sample]:
    return generate_synthetic_dataset(tmp_path / "synthetic", n_subjects=12, seed=0)
