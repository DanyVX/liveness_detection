"""Evaluation protocols: which subjects are in train / val / test.

A `Protocol` cannot be constructed if any subject appears in two splits, so leakage is
impossible by construction rather than by convention. Subject identity is
`(dataset, subject_id)`. Whether the *same person* appears under different IDs in two
datasets is not knowable from the data and is documented in docs/PROTOCOLS.md.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from liveness.data.base import Label, Sample


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ProtocolError(ValueError):
    """The protocol is malformed (e.g. an empty split or a split with a single class)."""


class LeakageError(ProtocolError):
    """A subject or file appears in more than one split."""


def _subjects(samples: Iterable[Sample]) -> set[tuple[str, str]]:
    return {s.subject_key for s in samples}


@dataclass(frozen=True)
class Protocol:
    name: str
    splits: Mapping[Split, tuple[Sample, ...]]

    def __post_init__(self) -> None:
        missing = [s.value for s in Split if s not in self.splits]
        if missing:
            raise ProtocolError(f"protocol {self.name!r} is missing splits: {missing}")
        for split, samples in self.splits.items():
            labels = {s.label for s in samples}
            if labels != {Label.ATTACK, Label.BONA_FIDE}:
                raise ProtocolError(
                    f"split {split.value!r} of {self.name!r} must contain both bona fide "
                    f"and attack samples (has {sorted(label.name for label in labels)})"
                )
        self._check_disjoint()

    def _check_disjoint(self) -> None:
        splits = list(self.splits.items())
        for i, (name_a, a) in enumerate(splits):
            for name_b, b in splits[i + 1 :]:
                shared_subjects = _subjects(a) & _subjects(b)
                if shared_subjects:
                    raise LeakageError(
                        f"{len(shared_subjects)} subject(s) in both {name_a.value} and "
                        f"{name_b.value}: {sorted(shared_subjects)[:5]}"
                    )
                shared_files = {s.path for s in a} & {s.path for s in b}
                if shared_files:
                    raise LeakageError(
                        f"{len(shared_files)} file(s) in both {name_a.value} and {name_b.value}"
                    )

    def subjects(self, split: Split) -> list[tuple[str, str]]:
        return sorted(_subjects(self.splits[split]))

    def manifest(self) -> dict[str, object]:
        """Subject-level manifest: auditable, diff-able, and independent of file paths."""
        return {
            "name": self.name,
            "splits": {
                split.value: [list(key) for key in self.subjects(split)] for split in Split
            },
        }

    def manifest_hash(self) -> str:
        blob = json.dumps(self.manifest(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def write_manifest(self, path: Path) -> None:
        path.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n")


def protocol_from_manifest(manifest_path: Path, samples: Sequence[Sample]) -> Protocol:
    """Rebuild a protocol from a committed subject manifest and a loaded sample list."""
    raw = json.loads(manifest_path.read_text())
    by_split: dict[Split, tuple[Sample, ...]] = {}
    for split in Split:
        keys = {(d, s) for d, s in raw["splits"][split.value]}
        by_split[split] = tuple(s for s in samples if s.subject_key in keys)
    return Protocol(name=raw["name"], splits=by_split)


def make_subject_disjoint_protocol(
    name: str,
    samples: Sequence[Sample],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 1337,
) -> Protocol:
    """Split by subject (never by file), deterministically for a given seed."""
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive and sum to 1")
    subjects = sorted(_subjects(samples))
    if len(subjects) < 3:
        raise ProtocolError("need at least 3 subjects for a train/val/test split")
    random.Random(seed).shuffle(subjects)

    n = len(subjects)
    n_val = max(1, round(n * fractions[1]))
    n_test = max(1, round(n * fractions[2]))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ProtocolError("not enough subjects left for the train split")
    groups = {
        Split.TRAIN: set(subjects[:n_train]),
        Split.VAL: set(subjects[n_train : n_train + n_val]),
        Split.TEST: set(subjects[n_train + n_val :]),
    }
    return Protocol(
        name=name,
        splits={
            split: tuple(s for s in samples if s.subject_key in keys)
            for split, keys in groups.items()
        },
    )
