import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from active_helpers import make_landmarks
from liveness.active.geometry import (
    N_LANDMARKS,
    ear_from_landmarks,
    eye_aspect_ratio,
    landmarks_valid,
    mar_from_landmarks,
    mouth_aspect_ratio,
    yaw_estimate,
    yaw_from_landmarks,
)
from liveness.active.landmarks import MESH_INDICES, LandmarkResult, LandmarkSource, to_compact

coord = st.floats(-500, 500, allow_nan=False, allow_infinity=False)


@given(
    pts=st.lists(st.tuples(coord, coord), min_size=6, max_size=6),
    scale=st.floats(0.1, 20),
    tx=coord,
    ty=coord,
)
def test_ear_invariant_to_scale_and_translation(
    pts: list[tuple[float, float]], scale: float, tx: float, ty: float
) -> None:
    eye = np.asarray(pts)
    assume(np.linalg.norm(eye[0] - eye[3]) > 1.0)
    moved = eye * scale + np.array([tx, ty])
    assert eye_aspect_ratio(moved) == pytest.approx(eye_aspect_ratio(eye), rel=1e-6, abs=1e-9)


def test_ear_known_value_and_3d_input() -> None:
    lm = make_landmarks(eye_open=1.0)
    assert ear_from_landmarks(lm) == pytest.approx(0.3)
    lm3 = np.hstack([lm, np.full((N_LANDMARKS, 1), 7.0)])
    assert ear_from_landmarks(lm3) == pytest.approx(0.3)
    assert ear_from_landmarks(make_landmarks(eye_open=0.0)) == pytest.approx(0.0)


def test_mouth_aspect_ratio() -> None:
    assert mar_from_landmarks(make_landmarks(mouth_open=0.7)) == pytest.approx(0.7)
    assert mouth_aspect_ratio(make_landmarks()[12:18]) == pytest.approx(0.0)


def test_degenerate_and_bad_shapes_raise() -> None:
    with pytest.raises(ValueError, match="degenerate eye"):
        eye_aspect_ratio(np.zeros((6, 2)))
    with pytest.raises(ValueError, match="shape"):
        eye_aspect_ratio(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="degenerate face"):
        yaw_estimate([0, 0], [1, 0], [1, 0], mirrored=False)


def test_yaw_sign_convention_unmirrored() -> None:
    left = make_landmarks(yaw=0.5)  # subject turns to their own left: nose toward image-right
    right = make_landmarks(yaw=-0.5)
    assert yaw_from_landmarks(left, mirrored=False) == pytest.approx(0.5)
    assert yaw_from_landmarks(right, mirrored=False) == pytest.approx(-0.5)
    assert yaw_from_landmarks(make_landmarks(), mirrored=False) == pytest.approx(0.0)


def test_mirrored_swaps_left_and_right() -> None:
    for yaw in (-0.6, -0.2, 0.3, 0.7):
        plain = make_landmarks(yaw=yaw)
        flipped = make_landmarks(yaw=yaw, mirrored=True)
        # The same physical turn gives the same subject-frame sign once `mirrored` is stated.
        assert yaw_from_landmarks(plain, mirrored=False) == pytest.approx(yaw)
        assert yaw_from_landmarks(flipped, mirrored=True) == pytest.approx(yaw)
        # Ignoring the flag swaps left and right.
        assert yaw_from_landmarks(flipped, mirrored=False) == pytest.approx(-yaw)


def test_landmarks_valid() -> None:
    lm = make_landmarks()
    assert landmarks_valid(lm)
    bad = lm.copy()
    bad[3, 0] = np.nan
    assert not landmarks_valid(bad)
    assert not landmarks_valid(lm[:5])
    assert not landmarks_valid([1, 2])


def test_mesh_mapping_and_protocol() -> None:
    assert len(MESH_INDICES) == N_LANDMARKS
    mesh = np.zeros((468, 3))
    mesh[:, 0] = np.arange(468) / 468.0
    out = to_compact(mesh, 200, 100)
    assert out.shape == (N_LANDMARKS, 2)
    assert out[0, 0] == pytest.approx(MESH_INDICES[0] / 468.0 * 200)

    class Stub:
        def extract(self, image_rgb: np.ndarray) -> LandmarkResult:  # type: ignore[type-arg]
            return LandmarkResult(make_landmarks(), 1, 1.0)

    src: LandmarkSource = Stub()
    assert src.extract(np.zeros((4, 4, 3), dtype=np.uint8)).num_faces == 1
