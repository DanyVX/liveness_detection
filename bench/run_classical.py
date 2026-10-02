"""Train the classical baseline and write raw val/test scores (no metrics computed here).

Synthetic runs are a pipeline smoke test only and are labelled "synthetic_smoke".
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import sklearn  # type: ignore[import-untyped]

from liveness.data import Split, make_subject_disjoint_protocol
from liveness.data.registry import load_dataset
from liveness.data.synthetic import DATASET_NAME, generate_synthetic_dataset
from liveness.models.classical import ClassicalPAD


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _split_payload(model: ClassicalPAD, samples: tuple[Any, ...]) -> dict[str, Any]:
    scores = model.decision_scores(samples)
    return {
        "sample_ids": [f"{s.dataset}/{s.path.name}" for s in samples],
        "scores": [float(x) for x in scores],
        "labels": [int(s.label) for s in samples],
        "attack_types": [s.attack_type.value for s in samples],
        "subject_keys": [list(s.subject_key) for s in samples],
    }


def run(args: argparse.Namespace) -> Path:
    out_dir: Path = args.out_dir
    with tempfile.TemporaryDirectory() as tmp:
        if args.data_root is None:
            if args.dataset != DATASET_NAME:
                raise SystemExit("--data-root is required for non-synthetic datasets")
            root = Path(tmp) / "synthetic"
            generate_synthetic_dataset(root, n_subjects=args.n_subjects, seed=args.seed)
        else:
            root = args.data_root
        samples = load_dataset(args.dataset, root)
        protocol = make_subject_disjoint_protocol(
            f"{args.dataset}_subject_disjoint", samples, seed=args.seed
        )
        model = ClassicalPAD(args.mode, random_state=args.seed)
        model.fit(protocol.splits[Split.TRAIN])
        payload: dict[str, Any] = {
            "data_kind": "synthetic_smoke" if args.dataset == DATASET_NAME else "real",
            "feature_mode": args.mode,
            "dataset": args.dataset,
            "protocol": protocol.name,
            "protocol_manifest_sha256": protocol.manifest_hash(),
            "score_convention": "higher = more bona fide",
            "splits": {
                split.value: _split_payload(model, protocol.splits[split])
                for split in (Split.VAL, Split.TEST)
            },
            "n_face_detection_fallbacks": model.n_fallback,
            "env": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "git_commit": _git_commit(),
            },
        }
    out = out_dir / f"classical_{args.mode.replace('+', '_')}_scores.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["lbp", "fft", "lbp+fft"], default="lbp+fft")
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--n-subjects", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    print(run(p.parse_args(argv)))


if __name__ == "__main__":
    main()
