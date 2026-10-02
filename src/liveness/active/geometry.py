"""Pure-numpy landmark geometry for active liveness.

Compact landmark layout (21 points, pixel coordinates, shape (21, 2) or (21, 3)):

    0-5    eye A, six points p1..p6 (p1/p4 corners, p2/p3 upper lid, p5/p6 lower lid)
    6-11   eye B, same ordering
    12-17  mouth, six points with the same ordering as an eye
    18     nose tip
    19, 20 face edges (left/right extremes of the face; order does not matter)

A and B are simply the two eyes as they appear in the delivered image; EAR and yaw use both
symmetrically, so the assignment never matters.

Mirroring: selfie cameras usually deliver a horizontally flipped image. Only the sign of the
yaw is affected. `yaw_estimate` therefore takes an explicit `mirrored` flag and always returns
the sign in the *subject's* frame.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

EYE_A = slice(0, 6)
EYE_B = slice(6, 12)
MOUTH = slice(12, 18)
NOSE_TIP = 18
EDGE_A = 19
EDGE_B = 20
N_LANDMARKS = 21

_EPS = 1e-9


def _points(a: ArrayLike, n: int, name: str) -> NDArray[np.float64]:
    pts = np.asarray(a, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] != n or pts.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape ({n}, 2) or ({n}, 3), got {pts.shape}")
    return pts[:, :2]


def _six_point_ratio(pts: NDArray[np.float64], name: str) -> float:
    horizontal = float(np.linalg.norm(pts[0] - pts[3]))
    if horizontal < _EPS:
        raise ValueError(f"degenerate {name}: corner points coincide")
    vertical = float(np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4]))
    return vertical / (2.0 * horizontal)


def eye_aspect_ratio(eye: ArrayLike) -> float:
    """Six-point EAR: (|p2-p6| + |p3-p5|) / (2 |p1-p4|). Invariant to scale and translation."""
    return _six_point_ratio(_points(eye, 6, "eye"), "eye")


def mouth_aspect_ratio(mouth: ArrayLike) -> float:
    """Same construction as EAR over the six mouth points (corners are p1 and p4)."""
    return _six_point_ratio(_points(mouth, 6, "mouth"), "mouth")


def yaw_estimate(
    nose_tip: ArrayLike, edge_a: ArrayLike, edge_b: ArrayLike, *, mirrored: bool
) -> float:
    """Normalised horizontal nose offset, roughly in [-1, 1]. Not an angle.

    Sign convention (subject's frame): positive = the subject turned toward THEIR OWN LEFT,
    negative = toward their own right, 0 = frontal. For an unmirrored camera the subject's left
    is image-right, so the nose moves toward larger x. If `mirrored` the image is flipped and
    the sign is inverted so the result still refers to the subject.
    """
    nose = np.asarray(nose_tip, dtype=np.float64)
    xa = float(np.asarray(edge_a, dtype=np.float64)[0])
    xb = float(np.asarray(edge_b, dtype=np.float64)[0])
    half_width = abs(xb - xa) / 2.0
    if half_width < _EPS:
        raise ValueError("degenerate face: edge landmarks coincide")
    raw = (float(nose[0]) - (xa + xb) / 2.0) / half_width
    return -raw if mirrored else raw


def landmarks_valid(landmarks: object) -> bool:
    if not isinstance(landmarks, np.ndarray):
        return False
    if landmarks.ndim != 2 or landmarks.shape[0] != N_LANDMARKS or landmarks.shape[1] not in (2, 3):
        return False
    return bool(np.all(np.isfinite(landmarks)))


def ear_from_landmarks(landmarks: NDArray[np.float64]) -> float:
    """Mean EAR of both eyes."""
    return 0.5 * (eye_aspect_ratio(landmarks[EYE_A]) + eye_aspect_ratio(landmarks[EYE_B]))


def mar_from_landmarks(landmarks: NDArray[np.float64]) -> float:
    return mouth_aspect_ratio(landmarks[MOUTH])


def yaw_from_landmarks(landmarks: NDArray[np.float64], *, mirrored: bool) -> float:
    return yaw_estimate(
        landmarks[NOSE_TIP], landmarks[EDGE_A], landmarks[EDGE_B], mirrored=mirrored
    )
