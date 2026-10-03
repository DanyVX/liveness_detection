"""Train and evaluate the small CNN PAD baseline without using test labels for tuning.

The CLI runs at least three seeds, selects epochs on validation ACER, writes resumable
checkpoints, evaluates frozen models, and labels synthetic runs as smoke tests. Licensed
datasets enter through the audited ``manifest.csv`` adapter documented in DATA.md.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from liveness import metrics
from liveness.data import Sample
from liveness.data.manifest import load_manifest_dataset
from liveness.data.protocols import Protocol, Split, make_subject_disjoint_protocol
from liveness.data.registry import REAL_DATASETS
from liveness.data.synthetic import generate_synthetic_dataset
from liveness.eval import (
    ScoredSplit,
    aggregate_seeds,
    cross_dataset_report,
    evaluate_protocol,
    write_results_json,
)
from liveness.models.cnn import (
    CNNConfig,
    FaceDataset,
    balanced_sampler,
    build_model,
    logits_to_scores,
    seed_everything,
)
from liveness.reporting import failure_grid


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    patience: int = 5
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"
    amp: bool = True
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_acer: float
    val_threshold: float


@dataclass
class TrainingRun:
    model: nn.Module
    history: list[EpochRecord]
    best_epoch: int
    best_val_acer: float
    stopped_early: bool


def _loader(
    samples: Sequence[Sample],
    model_config: CNNConfig,
    train_config: TrainConfig,
    *,
    training: bool,
    epoch: int = 0,
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = FaceDataset(samples, model_config, training=training)
    sampler = balanced_sampler(dataset.samples, train_config.seed + epoch) if training else None
    generator = torch.Generator().manual_seed(train_config.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=train_config.num_workers,
        generator=generator,
        pin_memory=train_config.device.startswith("cuda"),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    amp: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    losses: list[float] = []
    use_amp = amp and device.type == "cuda"
    if training and scaler is None:
        raise RuntimeError("gradient scaler is required while training")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            if optimizer is None:
                raise RuntimeError("optimizer is required while training")
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp),
        ):
            logits = model(images).reshape(-1)
            loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss; checkpoint was not advanced")
        if training:
            if optimizer is None:
                raise RuntimeError("optimizer is required while training")
            if scaler is None:
                raise RuntimeError("gradient scaler is required while training")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite gradient norm; checkpoint was not advanced")
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("data loader produced no batches")
    return float(np.mean(losses))


def _predict(
    model: nn.Module,
    samples: Sequence[Sample],
    model_config: CNNConfig,
    train_config: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    loader = _loader(samples, model_config, train_config, training=False)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for images, _labels in loader:
            scores.append(logits_to_scores(model(images.to(device))).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def _validation_metrics(scores: np.ndarray, samples: Sequence[Sample]) -> tuple[float, float]:
    labels = np.asarray([int(sample.label) for sample in samples], dtype=np.int64)
    types = np.asarray([sample.attack_type.value for sample in samples])
    _eer, threshold = metrics.eer(scores, labels)
    return float(metrics.acer(scores, labels, types, threshold)["acer"]), float(threshold)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: CNNConfig,
    train_config: TrainConfig,
    history: list[EpochRecord],
    best_epoch: int,
    best_val_acer: float,
    epochs_without_improvement: int,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "arch": model_config.arch,
        "cnn_config": asdict(model_config),
        "train_config": asdict(train_config),
        "history": [asdict(record) for record in history],
        "epoch": history[-1].epoch,
        "best_epoch": best_epoch,
        "best_val_acer": best_val_acer,
        "epochs_without_improvement": epochs_without_improvement,
        "scaler": scaler.state_dict(),
        "rng": _rng_state(),
    }


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def fit(
    protocol: Protocol,
    model_config: CNNConfig,
    train_config: TrainConfig,
    checkpoint_dir: Path,
    *,
    resume_from: Path | None = None,
) -> TrainingRun:
    """Fit with validation-ACER early stopping and deterministic single-worker resume."""
    seed_everything(train_config.seed)
    device = torch.device(train_config.device)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.amp and device.type == "cuda")
    history: list[EpochRecord] = []
    best_epoch, best_val_acer, stale = -1, float("inf"), 0
    start_epoch = 0
    if resume_from is not None:
        raw = torch.load(resume_from, map_location=device, weights_only=False)
        if raw.get("cnn_config") != asdict(model_config):
            raise ValueError("resume checkpoint CNN configuration does not match")
        saved_train = raw.get("train_config", {})
        comparable = {k: v for k, v in asdict(train_config).items() if k != "epochs"}
        if {k: saved_train.get(k) for k in comparable} != comparable:
            raise ValueError("resume checkpoint training configuration does not match")
        model.load_state_dict(raw["model"])
        optimizer.load_state_dict(raw["optimizer"])
        scaler.load_state_dict(raw.get("scaler", {}))
        history = [EpochRecord(**record) for record in raw["history"]]
        start_epoch = int(raw["epoch"]) + 1
        best_epoch = int(raw["best_epoch"])
        best_val_acer = float(raw["best_val_acer"])
        stale = int(raw["epochs_without_improvement"])
        _restore_rng(raw["rng"])
    best_path = checkpoint_dir / "best.pt"
    if resume_from is not None and not best_path.exists():
        prior_best = resume_from.parent / "best.pt"
        if not prior_best.exists():
            raise ValueError("resume checkpoint must be accompanied by best.pt")
        _save(
            best_path,
            torch.load(prior_best, map_location=device, weights_only=False),
        )
    for epoch in range(start_epoch, train_config.epochs):
        train_loader = _loader(
            protocol.splits[Split.TRAIN],
            model_config,
            train_config,
            training=True,
            epoch=epoch,
        )
        val_loader = _loader(protocol.splits[Split.VAL], model_config, train_config, training=False)
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=train_config.amp,
            scaler=scaler,
        )
        val_loss = _run_epoch(model, val_loader, None, device, amp=False)
        val_scores = _predict(model, protocol.splits[Split.VAL], model_config, train_config, device)
        val_acer, threshold = _validation_metrics(val_scores, protocol.splits[Split.VAL])
        history.append(EpochRecord(epoch, train_loss, val_loss, val_acer, threshold))
        if val_acer < best_val_acer - 1e-12:
            best_epoch, best_val_acer, stale = epoch, val_acer, 0
            _save(
                best_path,
                _checkpoint(
                    model,
                    optimizer,
                    model_config,
                    train_config,
                    history,
                    best_epoch,
                    best_val_acer,
                    stale,
                    scaler,
                ),
            )
        else:
            stale += 1
        _save(
            checkpoint_dir / "last.pt",
            _checkpoint(
                model,
                optimizer,
                model_config,
                train_config,
                history,
                best_epoch,
                best_val_acer,
                stale,
                scaler,
            ),
        )
        if stale >= train_config.patience:
            break
    if best_epoch < 0:
        raise RuntimeError("training did not produce a valid validation checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    model.cpu().eval()
    return TrainingRun(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_acer=best_val_acer,
        stopped_early=len(history) < train_config.epochs,
    )


def train(
    protocol: Protocol,
    config: CNNConfig,
    epochs: int,
    seed: int,
    device: str = "cpu",
) -> tuple[nn.Module, list[float], float]:
    """Compatibility wrapper for small callers; production runs should use :func:`fit`."""
    import tempfile

    train_config = TrainConfig(epochs=epochs, patience=epochs, seed=seed, device=device)
    with tempfile.TemporaryDirectory() as tmp:
        run = fit(protocol, config, train_config, Path(tmp))
    return run.model, [record.train_loss for record in run.history], run.history[-1].val_loss


def _scored(
    model: nn.Module,
    samples: Sequence[Sample],
    model_config: CNNConfig,
    train_config: TrainConfig,
) -> ScoredSplit:
    scores = _predict(model, samples, model_config, train_config, torch.device("cpu"))
    return ScoredSplit(
        scores=scores,
        labels=np.asarray([int(sample.label) for sample in samples], dtype=np.int64),
        attack_types=np.asarray([sample.attack_type.value for sample in samples]),
        subject_keys=[sample.subject_key for sample in samples],
        sample_ids=[sample.path.name for sample in samples],
    )


def _protocol(dataset: str, data_root: Path, seed: int, out_dir: Path) -> Protocol:
    if dataset in {"synthetic", "synthetic_b"}:
        domain = "B" if dataset == "synthetic_b" else "A"
        samples = generate_synthetic_dataset(
            out_dir / dataset, n_subjects=30, seed=seed, domain=domain
        )
        return make_subject_disjoint_protocol(f"{dataset}-cnn-v1", samples, seed=seed)
    name = "replay_attack-official" if dataset == "replay_attack" else f"{dataset}-intra"
    return load_manifest_dataset(dataset, data_root).protocol(name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["synthetic", *REAL_DATASETS], default="synthetic")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--target-dataset", choices=["synthetic_b", *REAL_DATASETS], default=None)
    parser.add_argument("--target-data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/cnn"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--face-margins", type=float, nargs="+", default=[0.2])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--arch", default="mobilenetv3_small_050")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume each seed/margin run from its checkpoints/last.pt",
    )
    args = parser.parse_args(argv)
    if len(set(args.seeds)) < 3:
        parser.error("at least three distinct seeds are required for reported CNN runs")

    data_kind = "synthetic_smoke" if args.dataset == "synthetic" else "real"
    if args.target_dataset is not None and (args.dataset == "synthetic") != (
        args.target_dataset == "synthetic_b"
    ):
        parser.error("synthetic source/target cannot be mixed with a real dataset")
    for margin in args.face_margins:
        seed_results: list[dict[str, Any]] = []
        for seed in args.seeds:
            run_dir = args.out_dir / f"margin_{margin:g}" / f"seed_{seed}"
            protocol = _protocol(args.dataset, args.data_root, seed, run_dir)
            model_config = CNNConfig(
                arch=args.arch,
                image_size=args.image_size,
                face_margin=margin,
                pretrained=args.pretrained,
            )
            train_config = TrainConfig(
                epochs=args.epochs,
                patience=args.patience,
                seed=seed,
                device=args.device,
                amp=not args.no_amp,
            )
            checkpoint_dir = run_dir / "checkpoints"
            resume_from = checkpoint_dir / "last.pt" if args.resume else None
            if resume_from is not None and not resume_from.exists():
                parser.error(f"resume checkpoint does not exist: {resume_from}")
            run = fit(
                protocol,
                model_config,
                train_config,
                checkpoint_dir,
                resume_from=resume_from,
            )
            val = _scored(run.model, protocol.splits[Split.VAL], model_config, train_config)
            test = _scored(run.model, protocol.splits[Split.TEST], model_config, train_config)
            report_args = {
                "protocol_name": protocol.name,
                "protocol_manifest_hash": protocol.manifest_hash(),
                "model_name": f"{args.arch}-margin-{margin:g}",
                "data_kind": data_kind,
                "seeds_info": {
                    "seed": seed,
                    "best_epoch": run.best_epoch,
                    "best_val_acer": run.best_val_acer,
                },
                "bootstrap_seed": seed,
            }
            if args.target_dataset is None:
                result = evaluate_protocol(val, test, **report_args)
                failure_samples = protocol.splits[Split.TEST]
                failure_scores = test.scores
            else:
                target_root = args.target_data_root or args.data_root
                target_protocol = _protocol(
                    args.target_dataset, target_root, seed, run_dir / "target"
                )
                target_test = _scored(
                    run.model,
                    target_protocol.splits[Split.TEST],
                    model_config,
                    train_config,
                )
                result = cross_dataset_report(
                    val,
                    test,
                    target_test,
                    target_name=args.target_dataset,
                    **report_args,
                )
                result["target_protocol"] = {
                    "name": target_protocol.name,
                    "manifest_hash": target_protocol.manifest_hash(),
                }
                failure_samples = target_protocol.splits[Split.TEST]
                failure_scores = target_test.scores
            result["training"] = {
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "history": [asdict(record) for record in run.history],
                "stopped_early": run.stopped_early,
            }
            write_results_json(run_dir / "result.json", result)
            if data_kind == "real":
                threshold_source = result["source"] if result["kind"] == "cross_dataset" else result
                threshold = float(threshold_source["thresholds"]["eer_val"])
                failure_grid(
                    failure_samples,
                    failure_scores.tolist(),
                    threshold,
                    run_dir / "false_accepts.png",
                    "false_accept",
                )
                failure_grid(
                    failure_samples,
                    failure_scores.tolist(),
                    threshold,
                    run_dir / "false_rejects.png",
                    "false_reject",
                )
            seed_results.append(result)
        write_results_json(
            args.out_dir / f"cnn_margin_{margin:g}_aggregate.json",
            aggregate_seeds(seed_results),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
