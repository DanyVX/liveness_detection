from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liveness.data import (
    AttackType,
    Label,
    LeakageError,
    Protocol,
    ProtocolError,
    Sample,
    Split,
    make_subject_disjoint_protocol,
)
from liveness.data.protocols import protocol_from_manifest


def _mk(dataset: str, subject: str, name: str, attack: bool) -> Sample:
    return Sample(
        dataset,
        subject,
        Path(f"{dataset}/{subject}_{name}.png"),
        Label.ATTACK if attack else Label.BONA_FIDE,
        AttackType.PRINT if attack else AttackType.NONE,
    )


def _both(dataset: str, subject: str) -> list[Sample]:
    return [_mk(dataset, subject, "bf", False), _mk(dataset, subject, "atk", True)]


def test_split_is_subject_disjoint(synthetic_samples: list[Sample]) -> None:
    proto = make_subject_disjoint_protocol("synthetic-v0", synthetic_samples, seed=1)
    train, val, test = (set(proto.subjects(s)) for s in (Split.TRAIN, Split.VAL, Split.TEST))
    assert train and val and test
    assert not (train & val) and not (train & test) and not (val & test)
    total = sum(len(v) for v in proto.splits.values())
    assert total == len(synthetic_samples)


def test_leakage_is_rejected() -> None:
    a, b, c = _both("d", "s1"), _both("d", "s2"), _both("d", "s3")
    leaked = _mk("d", "s1", "other", False)  # s1 now also in test
    with pytest.raises(LeakageError, match="subject"):
        Protocol("bad", {Split.TRAIN: tuple(a), Split.VAL: tuple(b), Split.TEST: (*c, leaked)})


def test_same_file_in_two_splits_is_rejected() -> None:
    a, b, c = _both("d", "s1"), _both("d", "s2"), _both("d", "s3")
    dup = replace(c[0], path=a[0].path)
    with pytest.raises(LeakageError, match="file"):
        Protocol("bad", {Split.TRAIN: tuple(a), Split.VAL: tuple(b), Split.TEST: (dup, c[1])})


def test_same_subject_id_in_different_datasets_is_not_leakage() -> None:
    proto = Protocol(
        "cross",
        {
            Split.TRAIN: tuple(_both("A", "001")),
            Split.VAL: tuple(_both("A", "002")),
            Split.TEST: tuple(_both("B", "001")),
        },
    )
    assert proto.subjects(Split.TEST) == [("B", "001")]


def test_single_class_split_is_rejected() -> None:
    a, b = _both("d", "s1"), _both("d", "s2")
    only_bf = (_mk("d", "s3", "bf", False),)
    with pytest.raises(ProtocolError, match="both bona fide and attack"):
        Protocol("bad", {Split.TRAIN: tuple(a), Split.VAL: tuple(b), Split.TEST: only_bf})


def test_missing_split_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="missing splits"):
        Protocol("bad", {Split.TRAIN: tuple(_both("d", "s1"))})


def test_split_is_deterministic_and_seed_sensitive(synthetic_samples: list[Sample]) -> None:
    h1 = make_subject_disjoint_protocol("p", synthetic_samples, seed=7).manifest_hash()
    h2 = make_subject_disjoint_protocol("p", synthetic_samples, seed=7).manifest_hash()
    h3 = make_subject_disjoint_protocol("p", synthetic_samples, seed=8).manifest_hash()
    assert h1 == h2
    assert h1 != h3


def test_manifest_roundtrip(synthetic_samples: list[Sample], tmp_path: Path) -> None:
    proto = make_subject_disjoint_protocol("p", synthetic_samples, seed=3)
    manifest = tmp_path / "p.json"
    proto.write_manifest(manifest)
    rebuilt = protocol_from_manifest(manifest, synthetic_samples)
    assert rebuilt.manifest_hash() == proto.manifest_hash()
    assert rebuilt.splits == proto.splits


def test_too_few_subjects() -> None:
    samples = _both("d", "s1") + _both("d", "s2")
    with pytest.raises(ProtocolError, match="at least 3"):
        make_subject_disjoint_protocol("p", samples)


@pytest.mark.parametrize("fractions", [(0.5, 0.5, 0.5), (0.8, 0.2, 0.0), (1.0, 0.0, 0.0)])
def test_bad_fractions(fractions: tuple[float, float, float]) -> None:
    samples = [s for i in range(5) for s in _both("d", f"s{i}")]
    with pytest.raises(ValueError, match="fractions"):
        make_subject_disjoint_protocol("p", samples, fractions=fractions)


@settings(max_examples=50, deadline=None)
@given(n_subjects=st.integers(min_value=3, max_value=60), seed=st.integers(0, 2**31))
def test_property_no_subject_in_two_splits(n_subjects: int, seed: int) -> None:
    samples = [s for i in range(n_subjects) for s in _both("d", f"s{i:03d}")]
    proto = make_subject_disjoint_protocol("p", samples, seed=seed)
    seen: set[tuple[str, str]] = set()
    for split in Split:
        subjects = set(proto.subjects(split))
        assert subjects
        assert not (seen & subjects)
        seen |= subjects
    assert len(seen) == n_subjects
