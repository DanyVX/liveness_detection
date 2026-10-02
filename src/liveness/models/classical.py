"""Classical PAD baseline: LBP and/or FFT features into an RBF-SVM.

Scores follow the repo convention: higher means more bona fide.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from liveness.data.base import Label, Sample
from liveness.features.fft import fft_features
from liveness.features.lbp import lbp_features
from liveness.features.preprocess import crop_face, load_image, normalize_resolution

logger = logging.getLogger(__name__)

FeatureMode = Literal["lbp", "fft", "lbp+fft"]


class ClassicalPAD:
    """Face crop -> fixed-size image -> hand-crafted features -> StandardScaler + SVC.

    Warning: `save`/`load` use joblib (pickle). Only load model files from sources you
    trust; loading a malicious file executes arbitrary code.
    """

    def __init__(
        self,
        feature_mode: FeatureMode = "lbp+fft",
        image_size: int = 128,
        face_margin: float = 0.2,
        lbp_radii: Sequence[int] = (1,),
        C: float = 1.0,
        random_state: int = 0,
    ) -> None:
        if feature_mode not in get_args(FeatureMode):
            raise ValueError(f"feature_mode must be one of {get_args(FeatureMode)}")
        self.feature_mode: FeatureMode = feature_mode
        self.image_size = image_size
        self.face_margin = face_margin
        self.lbp_radii = tuple(lbp_radii)
        self.C = C
        self.random_state = random_state
        self.n_fallback = 0
        self.pipeline: Pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=C,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self._fitted = False

    def extract(self, sample: Sample) -> NDArray[np.float64]:
        crop = crop_face(load_image(sample.path), self.face_margin, source=sample.path.name)
        if crop.used_fallback:
            self.n_fallback += 1
        img = normalize_resolution(crop.image, self.image_size)
        parts: list[NDArray[np.float64]] = []
        if "lbp" in self.feature_mode:
            parts.append(lbp_features(img, self.lbp_radii))
        if "fft" in self.feature_mode:
            parts.append(fft_features(img))
        return np.concatenate(parts)

    def _matrix(self, samples: Sequence[Sample]) -> NDArray[np.float64]:
        return np.stack([self.extract(s) for s in samples])

    def fit(self, samples: Sequence[Sample]) -> ClassicalPAD:
        labels = np.array([int(s.label) for s in samples])
        if set(labels.tolist()) != {int(Label.ATTACK), int(Label.BONA_FIDE)}:
            raise ValueError("training samples must contain both bona fide and attack")
        self.pipeline.fit(self._matrix(samples), labels)
        self._fitted = True
        return self

    def decision_scores(self, samples: Sequence[Sample]) -> NDArray[np.float64]:
        """SVM margin per sample; positive class is BONA_FIDE, so higher = more bona fide."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        return np.asarray(self.pipeline.decision_function(self._matrix(samples)), dtype=np.float64)

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path | str) -> ClassicalPAD:
        """Load a saved model. Only load files from trusted sources (pickle)."""
        obj = joblib.load(path)
        if not isinstance(obj, ClassicalPAD):
            raise TypeError(f"{path} does not contain a ClassicalPAD")
        return obj
