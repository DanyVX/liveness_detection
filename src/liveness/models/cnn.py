"""Small CNN baseline and deterministic training utilities for face PAD.

The module deliberately keeps the model and data pipeline separate from dataset-specific
loaders.  It consumes the unified :class:`~liveness.data.base.Sample` objects, so the same
subject-disjoint protocol is used for synthetic fixtures and (after approval) real datasets.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms  # type: ignore[import-untyped]

from liveness.data.base import Sample
from liveness.features.preprocess import crop_face, load_image, normalize_resolution

DEFAULT_ARCH = "mobilenetv3_small_050"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class CNNConfig:
    """Configuration stored with every checkpoint; scores are bona-fide probabilities."""

    arch: str = DEFAULT_ARCH
    image_size: int = 128
    face_margin: float = 0.20
    pretrained: bool = False

    def __post_init__(self) -> None:
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")
        if not 0 <= self.face_margin <= 1:
            raise ValueError("face_margin must be in [0, 1]")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch and request deterministic CUDA kernels where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():  # type: ignore[no-untyped-call]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_model(config: CNNConfig) -> nn.Module:
    """Build a one-logit timm classifier.  Logit > 0 means bona fide."""
    import timm

    try:
        return timm.create_model(config.arch, pretrained=config.pretrained, num_classes=1)
    except Exception as exc:
        hint = " (pretrained weights may need to be cached/downloaded)" if config.pretrained else ""
        raise ValueError(f"could not create model {config.arch!r}{hint}: {exc}") from exc


def training_transform(size: int) -> transforms.Compose:
    """Augmentations target colour, blur and framing shortcuts rather than attack cues."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size, scale=(0.80, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.35),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def evaluation_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FaceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Decode, crop and normalise unified samples without leaking labels into transforms."""

    def __init__(self, samples: Sequence[Sample], config: CNNConfig, *, training: bool) -> None:
        if not samples:
            raise ValueError("samples must not be empty")
        self.samples = tuple(samples)
        self.config = config
        self.transform = (
            training_transform(config.image_size)
            if training
            else evaluation_transform(config.image_size)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        crop = crop_face(load_image(sample.path), self.config.face_margin, source=sample.path.name)
        image: NDArray[np.uint8] = normalize_resolution(crop.image, self.config.image_size)
        # torchvision expects RGB HWC here; force contiguous after OpenCV operations.
        image = np.ascontiguousarray(image)
        return self.transform(image), torch.tensor(float(sample.label), dtype=torch.float32)


def balanced_sampler(samples: Sequence[Sample], seed: int) -> WeightedRandomSampler:
    """Replacement sampler with equal expected bona-fide/attack exposure each epoch."""
    labels = np.asarray([int(s.label) for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("balanced sampler requires both attack and bona-fide samples")
    weights = (1.0 / counts[labels]).tolist()
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, len(samples), replacement=True, generator=generator)


def logits_to_scores(logits: Tensor) -> Tensor:
    return torch.sigmoid(logits.reshape(-1))


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: CNNConfig,
    epoch: int,
    *,
    best_val_acer: float,
) -> dict[str, Any]:
    """Serializable checkpoint with RNG state needed for reproducible resume."""
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "arch": config.arch,
        "cnn_config": asdict(config),
        "epoch": epoch,
        "best_val_acer": best_val_acer,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

