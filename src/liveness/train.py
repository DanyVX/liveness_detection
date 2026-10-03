"""Train the small CNN PAD baseline on a subject-disjoint protocol.

This CLI intentionally supports the synthetic fixture only until a real dataset's EULA has
been accepted and its loader has been implemented.  Its checkpoints are accepted by
``liveness.export_onnx``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from liveness.data.base import Sample
from liveness.data.protocols import Protocol, Split, make_subject_disjoint_protocol
from liveness.data.synthetic import generate_synthetic_dataset
from liveness.models.cnn import (
    CNNConfig,
    FaceDataset,
    balanced_sampler,
    build_model,
    seed_everything,
)


def _loader(
    samples: Sequence[Sample], config: CNNConfig, training: bool, seed: int
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = FaceDataset(samples, config, training=training)
    return DataLoader(
        dataset,
        batch_size=16,
        sampler=balanced_sampler(dataset.samples, seed) if training else None,
        shuffle=False,
        num_workers=0,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images).reshape(-1)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite loss; stopping before corrupting the checkpoint"
                )
            if training:
                if optimizer is None:  # Defensive narrowing for type checkers and future edits.
                    raise RuntimeError("optimizer is required while training")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train(
    protocol: Protocol,
    config: CNNConfig,
    epochs: int,
    seed: int,
    device: str = "cpu",
) -> tuple[nn.Module, list[float], float]:
    """Train and return a model plus val loss.  Test split is deliberately never read."""
    seed_everything(seed)
    target = torch.device(device)
    model = build_model(config).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train_loader = _loader(protocol.splits[Split.TRAIN], config, True, seed)
    val_loader = _loader(protocol.splits[Split.VAL], config, False, seed)
    train_losses: list[float] = []
    for _ in range(epochs):
        train_losses.append(_run_epoch(model, train_loader, optimizer, target))
    val_loss = _run_epoch(model, val_loader, None, target)
    return model.cpu(), train_losses, val_loss


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the synthetic CNN PAD smoke baseline.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    samples = generate_synthetic_dataset(
        args.out.parent / "synthetic", n_subjects=12, seed=args.seed
    )
    protocol = make_subject_disjoint_protocol("synthetic-cnn", samples, seed=args.seed)
    config = CNNConfig(image_size=args.image_size)
    model, _train_losses, val_loss = train(protocol, config, args.epochs, args.seed, args.device)
    torch.save(
        {"model": model.state_dict(), "arch": config.arch, "cnn_config": config.__dict__},
        args.out,
    )
    print(
        f"wrote {args.out}; val_loss={val_loss:.6f}; "
        "synthetic smoke only, not a PAD result"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
