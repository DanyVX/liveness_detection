import math
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from liveness.fusion import Decision, TemporalFuser


def make(**kw: object) -> TemporalFuser:
    params: dict[str, object] = {
        "accept_threshold": 0.6,
        "reject_threshold": 0.4,
        "window": 8,
        "min_valid_frames": 4,
    }
    params.update(kw)
    return TemporalFuser(**params)  # type: ignore[arg-type]


def test_window_shorter_than_needed_is_insufficient_evidence() -> None:
    f = make()
    for _ in range(3):
        r = f.update(0.99)
    assert r.decision is Decision.INSUFFICIENT_EVIDENCE
    assert r.aggregate is None
    assert f.update(0.99).decision is Decision.LIVE


def test_no_face_frames_do_not_count_as_evidence() -> None:
    f = make()
    results = [f.update(s) for s in [0.9, None, 0.9, None, 0.9, None, 0.9]]
    assert results[-1].n_valid == 4
    assert results[-1].decision is Decision.LIVE
    f.reset()
    r = f.fuse([0.9, None, None, None, None, 0.9, None, None])
    assert r.decision is Decision.INSUFFICIENT_EVIDENCE


def test_no_face_frames_age_out_old_evidence() -> None:
    f = make(window=4, min_valid_frames=2)
    f.fuse([0.95, 0.95])
    assert f.state is Decision.LIVE
    for _ in range(3):
        r = f.update(None)
    assert r.decision is Decision.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1.5, -0.1])
def test_non_finite_or_out_of_range_score_rejected(bad: float) -> None:
    f = make()
    f.update(0.5)
    with pytest.raises(ValueError, match="score"):
        f.update(bad)
    assert f.window_size == 1


def test_hysteresis_prevents_flapping_on_noisy_scores_near_threshold() -> None:
    # Raw per-frame thresholding at 0.5 flips on every frame; the fuser must not.
    noisy = [0.52, 0.48, 0.55, 0.45, 0.5, 0.53, 0.47, 0.51, 0.49, 0.54, 0.46, 0.5] * 3
    fuser = make(window=4, min_valid_frames=2, accept_threshold=0.6, reject_threshold=0.4)
    states = [fuser.update(s).decision for s in noisy]
    raw_flips = sum(1 for a, b in pairwise(noisy) if (a >= 0.5) != (b >= 0.5))
    flips = sum(1 for a, b in pairwise(states) if a != b)
    assert raw_flips > 10
    assert flips == 0


def test_hysteresis_keeps_state_between_thresholds_then_switches() -> None:
    f = make(window=3, min_valid_frames=3)
    assert f.fuse([0.9, 0.9, 0.9]).decision is Decision.LIVE
    for _ in range(3):
        assert f.update(0.5).decision is Decision.LIVE  # aggregate 0.5 sits between thresholds
    assert f.update(0.1).decision is Decision.SPOOF  # aggregate 0.37 <= reject
    for _ in range(3):
        assert f.update(0.5).decision is Decision.SPOOF  # does not flip back until >= accept


def test_ambiguous_without_history_is_insufficient_evidence() -> None:
    r = make(window=3, min_valid_frames=3).fuse([0.5, 0.5, 0.5])
    assert r.decision is Decision.INSUFFICIENT_EVIDENCE
    assert r.aggregate == pytest.approx(0.5)


def test_median_ignores_single_outlier_frame() -> None:
    scores = [0.7, 0.7, 0.7, 0.0]
    assert make(aggregation="median").fuse(scores).decision is Decision.LIVE
    assert make(aggregation="mean").fuse(scores).decision is Decision.INSUFFICIENT_EVIDENCE


def test_thresholds_are_constructor_config() -> None:
    scores = [0.7] * 4
    assert make(accept_threshold=0.6).fuse(scores).decision is Decision.LIVE
    assert make(accept_threshold=0.8, reject_threshold=0.75).fuse(scores).decision is (
        Decision.SPOOF
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"reject_threshold": 0.7},
        {"window": 0},
        {"min_valid_frames": 9},
        {"min_valid_frames": 0},
        {"aggregation": "mode"},
        {"accept_threshold": math.nan},
    ],
)
def test_invalid_config_rejected(kw: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make(**kw)


@given(st.lists(st.one_of(st.none(), st.floats(0.0, 1.0)), max_size=200))
def test_memory_bounded_and_state_valid(scores: list[float | None]) -> None:
    f = make(window=5, min_valid_frames=2)
    for s in scores:
        r = f.update(s)
        assert f.window_size <= 5
        assert r.n_frames <= 5
        assert r.n_valid <= r.n_frames
    assert f.state in set(Decision)


@given(st.lists(st.floats(0.6, 1.0), min_size=4, max_size=50))
def test_all_confident_scores_end_live(scores: list[float]) -> None:
    assert make().fuse(scores).decision is Decision.LIVE
