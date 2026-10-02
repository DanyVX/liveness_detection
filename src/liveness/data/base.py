"""Unified sample definition shared by every dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path


class Label(IntEnum):
    """Binary PAD label. BONA_FIDE is the positive ("live") class."""

    ATTACK = 0
    BONA_FIDE = 1


class AttackType(StrEnum):
    NONE = "none"  # bona fide presentations only
    PRINT = "print"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class Sample:
    """One image or video presentation.

    `subject_id` is unique only within `dataset`; the pair is the identity used for
    split-leakage checks (see `subject_key`).
    """

    dataset: str
    subject_id: str
    path: Path
    label: Label
    attack_type: AttackType

    def __post_init__(self) -> None:
        if not self.dataset or not self.subject_id:
            raise ValueError("dataset and subject_id must be non-empty")
        is_bona_fide = self.label is Label.BONA_FIDE
        if is_bona_fide != (self.attack_type is AttackType.NONE):
            raise ValueError(
                f"label={self.label.name} is inconsistent with attack_type={self.attack_type.value}"
            )

    @property
    def subject_key(self) -> tuple[str, str]:
        return (self.dataset, self.subject_id)
