"""End-to-end pipeline smoke check on the synthetic fixture (NOT a real result).

Trains the classical baseline on synthetic domain A, evaluates intra-domain and A->B
cross-domain, and writes results JSON labelled `synthetic_smoke`. It exists to prove the
protocol -> model -> metrics -> report chain works in CI-sized time.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from liveness.data import Sample, Split, make_subject_disjoint_protocol
from liveness.data.synthetic import generate_synthetic_dataset
from liveness.eval import (
    ScoredSplit,
    cross_dataset_report,
    evaluate_protocol,
    write_results_json,
)
from liveness.models import ClassicalPAD

DATA_KIND = "synthetic_smoke"


def _scored(model: ClassicalPAD, samples: list[Sample]) -> ScoredSplit:
    import numpy as np

    return ScoredSplit(
        scores=model.decision_scores(samples),
        labels=np.array([int(s.label) for s in samples]),
        attack_types=np.array([s.attack_type.value for s in samples]),
        subject_keys=np.array([f"{s.dataset}:{s.subject_id}" for s in samples]),
        sample_ids=np.array([s.path.name for s in samples]),
    )


def run(out_dir: Path, n_subjects: int, seed: int, mode: str, n_boot: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        a = generate_synthetic_dataset(Path(tmp) / "a", n_subjects, seed, domain="A")
        b = generate_synthetic_dataset(Path(tmp) / "b", n_subjects, seed + 1, domain="B")
        proto_a = make_subject_disjoint_protocol("synthetic-a-v0", a, seed=seed)
        proto_b = make_subject_disjoint_protocol("synthetic-b-v0", b, seed=seed)
        model = ClassicalPAD(mode, random_state=seed).fit(list(proto_a.splits[Split.TRAIN]))  # type: ignore[arg-type]
        val_a = _scored(model, list(proto_a.splits[Split.VAL]))
        test_a = _scored(model, list(proto_a.splits[Split.TEST]))
        test_b = _scored(model, list(proto_b.splits[Split.TEST]))
        common = {"model_name": f"classical-{mode}", "data_kind": DATA_KIND, "n_boot": n_boot}

        intra = evaluate_protocol(
            val_a,
            test_a,
            protocol_name=proto_a.name,
            protocol_manifest_hash=proto_a.manifest_hash(),
            **common,  # type: ignore[arg-type]
        )
        path = out_dir / f"smoke_classical_{mode.replace('+', '_')}_intra.json"
        write_results_json(path, intra)
        written.append(path)

        cross = cross_dataset_report(
            val_a,
            test_a,
            test_b,
            protocol_name="synthetic A->B",
            protocol_manifest_hash=proto_a.manifest_hash(),
            target_name="synthetic_b",
            **common,  # type: ignore[arg-type]
        )
        path = out_dir / f"smoke_classical_{mode.replace('+', '_')}_cross_a_to_b.json"
        write_results_json(path, cross)
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-subjects", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="lbp+fft", choices=["lbp", "fft", "lbp+fft"])
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    for path in run(args.out_dir, args.n_subjects, args.seed, args.mode, args.n_boot):
        print(path)


if __name__ == "__main__":
    main()
