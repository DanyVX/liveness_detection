"""ISO/IEC 30107-3 style PAD metrics as pure numpy functions.

Convention: a higher score means more bona fide, and a presentation is *accepted* when
``score >= threshold``. Label 1 (bona fide) is the positive class.

- APCER (per attack type): fraction of that type's attack presentations accepted.
- BPCER: fraction of bona fide presentations rejected (``score < threshold``).
- ACER = (APCER + BPCER) / 2, where APCER is the *worst* per-type APCER (the ISO reading).
  The variant that averages APCER over attack types is reported alongside as ``acer_mean``.
- HTER = (FAR + FRR) / 2 with FAR the pooled attack acceptance rate (type-agnostic).

Degenerate inputs (empty class, NaN/inf scores, a single class where both are needed,
mismatched lengths) raise ``ValueError`` instead of returning a misleading number.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

_F = NDArray[np.float64]


def _scores(x: ArrayLike, name: str = "scores") -> _F:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _labels(scores: _F, labels: ArrayLike) -> NDArray[np.int64]:
    lab = np.asarray(labels)
    if lab.shape != scores.shape:
        raise ValueError(f"labels shape {lab.shape} does not match scores shape {scores.shape}")
    if not np.all((lab == 0) | (lab == 1)):
        raise ValueError("labels must be 0 (attack) or 1 (bona fide)")
    return lab.astype(np.int64)


def _split(scores: ArrayLike, labels: ArrayLike) -> tuple[_F, _F]:
    """Return (bona_fide_scores, attack_scores); both classes must be present."""
    s = _scores(scores)
    lab = _labels(s, labels)
    bona, attack = s[lab == 1], s[lab == 0]
    if bona.size == 0:
        raise ValueError("no bona fide samples: metric undefined with a single class")
    if attack.size == 0:
        raise ValueError("no attack samples: metric undefined with a single class")
    return bona, attack


def bpcer(bona_fide_scores: ArrayLike, threshold: float) -> float:
    """Fraction of bona fide presentations rejected (score < threshold)."""
    s = _scores(bona_fide_scores, "bona_fide_scores")
    return float(np.mean(s < threshold))


def apcer(
    attack_scores: ArrayLike, attack_types: Sequence[str] | NDArray[np.str_], threshold: float
) -> dict[str, float]:
    """APCER per attack type plus ``"max"`` (worst type) and ``"mean"`` (over types)."""
    s = _scores(attack_scores, "attack_scores")
    types = np.asarray(attack_types, dtype=str)
    if types.shape != s.shape:
        raise ValueError(f"attack_types shape {types.shape} does not match scores {s.shape}")
    out: dict[str, float] = {}
    for t in sorted(set(types.tolist())):
        out[t] = float(np.mean(s[types == t] >= threshold))
    per_type = list(out.values())
    out["max"] = max(per_type)
    out["mean"] = float(np.mean(per_type))
    return out


def acer(
    scores: ArrayLike,
    labels: ArrayLike,
    attack_types: Sequence[str] | NDArray[np.str_],
    threshold: float,
) -> dict[str, float]:
    """ACER with worst-type APCER (``acer``) and mean-type APCER (``acer_mean``)."""
    s = _scores(scores)
    lab = _labels(s, labels)
    bona, attack = _split(s, lab)
    types = np.asarray(attack_types, dtype=str)
    if types.shape != s.shape:
        raise ValueError(f"attack_types shape {types.shape} does not match scores {s.shape}")
    a = apcer(attack, types[lab == 0], threshold)
    b = bpcer(bona, threshold)
    return {
        "apcer_max": a["max"],
        "apcer_mean": a["mean"],
        "bpcer": b,
        "acer": (a["max"] + b) / 2,
        "acer_mean": (a["mean"] + b) / 2,
    }


def hter(scores: ArrayLike, labels: ArrayLike, threshold: float) -> float:
    """Half total error rate with pooled (type-agnostic) false accept / false reject rates."""
    bona, attack = _split(scores, labels)
    return (float(np.mean(attack >= threshold)) + float(np.mean(bona < threshold))) / 2


def confusion_matrix(scores: ArrayLike, labels: ArrayLike, threshold: float) -> dict[str, int]:
    """Counts at a threshold; positive = bona fide accepted. tp/fn: bona fide, fp/tn: attack."""
    bona, attack = _split(scores, labels)
    return {
        "tp": int(np.sum(bona >= threshold)),
        "fn": int(np.sum(bona < threshold)),
        "fp": int(np.sum(attack >= threshold)),
        "tn": int(np.sum(attack < threshold)),
    }


def _rates(bona: _F, attack: _F) -> tuple[_F, _F, _F]:
    """Thresholds (ascending, ending with +inf = reject all), FAR and FRR at each."""
    finite = np.unique(np.concatenate([bona, attack]))
    thr = np.append(finite, np.inf)
    b, a = np.sort(bona), np.sort(attack)
    frr = np.searchsorted(b, thr, side="left") / b.size
    far = 1.0 - np.searchsorted(a, thr, side="left") / a.size
    return thr, far, frr


def det_curve(scores: ArrayLike, labels: ArrayLike) -> dict[str, list[float]]:
    """DET points: pooled APCER (far) against BPCER (frr). The last threshold is +inf."""
    bona, attack = _split(scores, labels)
    thr, far, frr = _rates(bona, attack)
    return {"threshold": thr.tolist(), "far": far.tolist(), "frr": frr.tolist()}


def roc_curve(scores: ArrayLike, labels: ArrayLike) -> dict[str, list[float]]:
    """ROC points: false accept rate (attack accepted) against true accept rate (bona fide)."""
    bona, attack = _split(scores, labels)
    thr, far, frr = _rates(bona, attack)
    return {"threshold": thr.tolist(), "fpr": far.tolist(), "tpr": (1.0 - frr).tolist()}


def eer(scores: ArrayLike, labels: ArrayLike) -> tuple[float, float]:
    """Equal error rate and the threshold achieving it.

    With discrete scores FAR and FRR rarely cross exactly; the candidate threshold with the
    smallest |FAR - FRR| is used and the EER is the mean of the two rates there.
    """
    bona, attack = _split(scores, labels)
    thr, far, frr = _rates(bona, attack)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def _average_ranks(x: _F) -> _F:
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg = ends - (counts - 1) / 2.0
    return avg[inv].astype(np.float64)


def auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get half credit)."""
    bona, attack = _split(scores, labels)
    ranks = _average_ranks(np.concatenate([bona, attack]))
    n_pos, n_neg = bona.size, attack.size
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def threshold_at_bpcer(val_bona_fide_scores: ArrayLike, target_bpcer: float) -> float:
    """Highest threshold whose BPCER on the given (validation) bona fide scores is <= target.

    Pass validation scores only; the result is then applied unchanged to test data.
    """
    if not 0.0 <= target_bpcer < 1.0:
        raise ValueError(f"target_bpcer must be in [0, 1), got {target_bpcer}")
    s = np.sort(_scores(val_bona_fide_scores, "val_bona_fide_scores"))
    k = int(np.floor(target_bpcer * s.size + 1e-9))
    return float(s[k])


def bootstrap_ci(
    statistic: Callable[[_F, NDArray[np.int64], NDArray[np.str_]], float],
    scores: ArrayLike,
    labels: ArrayLike,
    attack_types: Sequence[str] | NDArray[np.str_],
    subject_keys: Sequence[object],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile CI from resampling *subjects* (with all their samples), not samples.

    Replicates missing either class are skipped and counted in ``n_skipped``; if every
    replicate is skipped a ValueError is raised.
    """
    s = _scores(scores)
    lab = _labels(s, labels)
    types = np.asarray(attack_types, dtype=str)
    if types.shape != s.shape or len(subject_keys) != s.size:
        raise ValueError("attack_types and subject_keys must match scores in length")
    if n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("n_boot must be >= 1 and alpha in (0, 1)")
    _split(s, lab)
    keys = np.array([repr(k) for k in subject_keys])
    uniq, inverse = np.unique(keys, return_inverse=True)
    groups = [np.flatnonzero(inverse == g) for g in range(uniq.size)]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([groups[g] for g in pick])
        if lab[idx].min() == lab[idx].max():
            continue
        values.append(float(statistic(s[idx], lab[idx], types[idx])))
    if not values:
        raise ValueError("every bootstrap replicate lacked a class; too few subjects")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(statistic(s, lab, types)),
        "lower": float(lo),
        "upper": float(hi),
        "n_boot": n_boot,
        "n_skipped": n_boot - len(values),
        "n_subjects": int(uniq.size),
        "alpha": alpha,
    }
