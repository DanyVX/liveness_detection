import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from liveness import metrics

SCORES = np.array([0.9, 0.8, 0.4, 0.7, 0.2, 0.3, 0.5, 0.1])
LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0])
TYPES = np.array(["none"] * 4 + ["print", "print", "replay", "replay"])


def test_bpcer_hand_computed() -> None:
    assert metrics.bpcer(SCORES[:4], 0.5) == 0.25
    assert metrics.bpcer(SCORES[:4], 0.4) == 0.0  # score == threshold is accepted


def test_apcer_per_type_max_mean_and_tie_accepts() -> None:
    a = metrics.apcer(SCORES[4:], TYPES[4:], 0.5)
    assert a == {"print": 0.0, "replay": 0.5, "max": 0.5, "mean": 0.25}


def test_acer_worst_and_mean_variants() -> None:
    r = metrics.acer(SCORES, LABELS, TYPES, 0.5)
    assert r["bpcer"] == 0.25
    assert r["acer"] == pytest.approx((0.5 + 0.25) / 2)
    assert r["acer_mean"] == pytest.approx((0.25 + 0.25) / 2)


def test_hter_and_confusion_hand_computed() -> None:
    assert metrics.hter(SCORES, LABELS, 0.5) == 0.25
    assert metrics.confusion_matrix(SCORES, LABELS, 0.5) == {"tp": 3, "fn": 1, "fp": 1, "tn": 3}


def test_eer_and_threshold_hand_computed() -> None:
    value, thr = metrics.eer(SCORES, LABELS)
    assert (value, thr) == (0.25, 0.5)


def test_auc_hand_computed_matches_sklearn() -> None:
    assert metrics.auc(SCORES, LABELS) == pytest.approx(15 / 16)
    assert metrics.auc(SCORES, LABELS) == pytest.approx(roc_auc_score(LABELS, SCORES))


def test_auc_ties_get_half_credit() -> None:
    assert metrics.auc([0.5, 0.5], [1, 0]) == 0.5


def test_roc_and_det_points() -> None:
    roc = metrics.roc_curve(SCORES, LABELS)
    assert roc["fpr"][0] == 1.0 and roc["tpr"][0] == 1.0  # lowest threshold accepts all
    assert roc["fpr"][-1] == 0.0 and roc["tpr"][-1] == 0.0  # +inf threshold rejects all
    det = metrics.det_curve(SCORES, LABELS)
    assert det["frr"][-1] == 1.0 and det["far"][-1] == 0.0
    assert len(det["threshold"]) == len(det["far"]) == len(det["frr"])


def test_threshold_at_bpcer_hand_computed() -> None:
    bona = np.array([0.4, 0.7, 0.8, 0.9])
    assert metrics.threshold_at_bpcer(bona, 0.01) == 0.4
    assert metrics.threshold_at_bpcer(bona, 0.25) == 0.7
    assert metrics.bpcer(bona, 0.7) == 0.25
    with pytest.raises(ValueError, match="target_bpcer"):
        metrics.threshold_at_bpcer(bona, 1.0)


def test_threshold_at_bpcer_with_ties_never_exceeds_target() -> None:
    bona = np.array([0.5, 0.5, 0.5, 0.9])
    thr = metrics.threshold_at_bpcer(bona, 0.25)
    assert metrics.bpcer(bona, thr) <= 0.25


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics.bpcer([], 0.5),
        lambda: metrics.bpcer([0.1, float("nan")], 0.5),
        lambda: metrics.bpcer([0.1, float("inf")], 0.5),
        lambda: metrics.auc([0.1, 0.2], [1, 1]),
        lambda: metrics.auc([0.1, 0.2], [0, 0]),
        lambda: metrics.eer([0.1, 0.2], [1, 1]),
        lambda: metrics.roc_curve([0.1, 0.2], [0, 0]),
        lambda: metrics.hter([0.1, 0.2], [1, 1], 0.5),
        lambda: metrics.auc([0.1, 0.2], [1]),
        lambda: metrics.auc([0.1, 0.2], [1, 2]),
        lambda: metrics.auc([[0.1], [0.2]], [1, 0]),
        lambda: metrics.apcer([0.1, 0.2], ["print"], 0.5),
        lambda: metrics.acer([0.1, 0.2], [1, 0], ["none"], 0.5),
    ],
)
def test_degenerate_inputs_raise(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        call()


def test_single_class_error_message_is_clear() -> None:
    with pytest.raises(ValueError, match="single class"):
        metrics.auc([0.1, 0.2], [1, 1])


def test_bootstrap_resamples_subjects_not_samples() -> None:
    n = 10
    scores = np.concatenate([np.arange(n) / 100 + 0.5, np.arange(n) / 100])
    labels = np.array([1] * n + [0] * n)
    subjects = list(range(n)) + list(range(n))
    types = np.array(["none"] * n + ["print"] * n)
    violations = 0

    def stat(s: np.ndarray, lab: np.ndarray, t: np.ndarray) -> float:
        nonlocal violations
        bona = np.sort(np.round((s[lab == 1] - 0.5) * 100))
        att = np.sort(np.round(s[lab == 0] * 100))
        violations += int(not np.array_equal(bona, att))  # each subject drags both samples
        return float(np.mean(s))

    ci = metrics.bootstrap_ci(stat, scores, labels, types, subjects, n_boot=200, seed=1)
    assert violations == 0
    assert ci["n_subjects"] == n and ci["lower"] <= ci["upper"]


def test_bootstrap_is_seeded_and_ci_contains_estimate_for_separable_data() -> None:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(1, 0.3, 40), rng.normal(0, 0.3, 40)])
    labels = np.array([1] * 40 + [0] * 40)
    types = np.array(["none"] * 40 + ["print"] * 40)
    subj = list(range(40)) * 2

    def stat(s: np.ndarray, lab: np.ndarray, t: np.ndarray) -> float:
        return metrics.auc(s, lab)

    a = metrics.bootstrap_ci(stat, scores, labels, types, subj, n_boot=100, seed=3)
    b = metrics.bootstrap_ci(stat, scores, labels, types, subj, n_boot=100, seed=3)
    assert a == b
    assert a["lower"] <= a["estimate"] <= a["upper"]


def test_bootstrap_errors() -> None:
    def stat(s: np.ndarray, lab: np.ndarray, t: np.ndarray) -> float:
        return 0.0

    with pytest.raises(ValueError, match="length"):
        metrics.bootstrap_ci(stat, [0.1, 0.2], [1, 0], ["none", "print"], [1], n_boot=5)
    with pytest.raises(ValueError, match="n_boot"):
        metrics.bootstrap_ci(stat, [0.1, 0.2], [1, 0], ["none", "print"], [1, 2], n_boot=0)
    # one subject holds both classes: every replicate contains both -> fine;
    # two single-class subjects, so many replicates lack a class but not all.
    metrics.bootstrap_ci(stat, [0.1, 0.2], [1, 0], ["none", "print"], [1, 2], n_boot=50)
    with pytest.raises(ValueError, match="lacked a class"):
        metrics.bootstrap_ci(stat, [0.1, 0.2], [1, 0], ["none", "print"], [1, 2], n_boot=1, seed=0)


_pair = st.tuples(st.floats(-5, 5, allow_nan=False), st.integers(0, 1))


@st.composite
def _datasets(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = draw(st.lists(_pair, min_size=2, max_size=40))
    labels = np.array([p[1] for p in pairs] + [0, 1])
    scores = np.array([p[0] for p in pairs] + [draw(st.floats(-5, 5)), draw(st.floats(-5, 5))])
    types = np.where(labels == 1, "none", np.where(np.arange(labels.size) % 2 == 0, "a", "b"))
    return scores, labels, types


@settings(max_examples=60, deadline=None)
@given(_datasets(), st.floats(-6, 6))
def test_prop_acer_in_unit_interval(data, thr) -> None:  # type: ignore[no-untyped-def]
    scores, labels, types = data
    r = metrics.acer(scores, labels, types, thr)
    assert 0 <= r["acer"] <= 1 and r["acer_mean"] <= r["acer"] + 1e-12


@settings(max_examples=60, deadline=None)
@given(_datasets(), st.floats(-6, 6), st.floats(0, 3))
def test_prop_threshold_monotonicity(data, thr, delta) -> None:  # type: ignore[no-untyped-def]
    scores, labels, types = data
    lo = metrics.acer(scores, labels, types, thr)
    hi = metrics.acer(scores, labels, types, thr + delta)
    assert hi["bpcer"] >= lo["bpcer"]
    assert hi["apcer_max"] <= lo["apcer_max"]
    assert hi["apcer_mean"] <= lo["apcer_mean"]


@settings(max_examples=60, deadline=None)
@given(st.integers(1, 15), st.integers(1, 15), st.floats(0.01, 5))
def test_prop_perfect_separation_gives_zero(n_b, n_a, gap) -> None:  # type: ignore[no-untyped-def]
    scores = np.concatenate([np.linspace(gap, gap + 1, n_b), np.linspace(-1, 0, n_a)])
    labels = np.array([1] * n_b + [0] * n_a)
    types = np.array(["none"] * n_b + ["print"] * n_a)
    value, thr = metrics.eer(scores, labels)
    assert value == 0.0
    assert metrics.acer(scores, labels, types, thr)["acer"] == 0.0
    assert metrics.auc(scores, labels) == 1.0


@settings(max_examples=80, deadline=None)
@given(_datasets())
def test_prop_auc_matches_sklearn_and_eer_bounds(data) -> None:  # type: ignore[no-untyped-def]
    scores, labels, _ = data
    assert metrics.auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))
    e, _ = metrics.eer(scores, labels)
    assert 0 <= e <= 1
