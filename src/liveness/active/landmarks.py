"""Landmark sources. Tests and other backends plug in through `LandmarkSource`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

# Compact layout order (see geometry.py): eye A (6), eye B (6), mouth (6), nose tip, edge, edge.
# NOTE: these MediaPipe Face Mesh (468-point) indices are from memory of commonly used lists and
# MUST be verified against the MediaPipe documentation / canonical face model before real use.
# Eye order is p1 corner, p2/p3 upper lid, p4 corner, p5/p6 lower lid. Mouth uses the inner lip
# ring: corners, two upper-lip points, two lower-lip points.
MESH_INDICES: tuple[int, ...] = (
    33, 160, 158, 133, 153, 144,  # eye A
    362, 385, 387, 263, 373, 380,  # eye B
    78, 82, 312, 308, 317, 87,  # mouth
    1,  # nose tip
    234, 454,  # face edges
)  # fmt: skip


@dataclass(frozen=True)
class LandmarkResult:
    landmarks: NDArray[np.float64] | None
    num_faces: int
    confidence: float


class LandmarkSource(Protocol):
    def extract(self, image_rgb: NDArray[np.uint8]) -> LandmarkResult: ...


def to_compact(mesh_xy: NDArray[np.float64], width: int, height: int) -> NDArray[np.float64]:
    """Select the compact set from a (468, >=2) array of normalised coords; return pixels."""
    pts = np.asarray(mesh_xy, dtype=np.float64)[list(MESH_INDICES), :2]
    return pts * np.array([width, height], dtype=np.float64)


class FaceMeshAdapter:  # pragma: no cover
    """MediaPipe Face Mesh backend. UNTESTED: mediapipe is not available in CI.

    mediapipe is imported lazily on first `extract`. The API calls below (legacy
    `mediapipe.solutions.face_mesh`) and the index lists in `MESH_INDICES` are unverified and
    must be checked against the installed MediaPipe version. Face Mesh may not provide a
    per-landmark visibility score, so `confidence` is the constant `assumed_confidence`; wire in
    a real signal if the backend offers one.
    """

    def __init__(
        self,
        max_faces: int = 2,
        min_detection_confidence: float = 0.5,
        assumed_confidence: float = 1.0,
    ) -> None:
        self._max_faces = max_faces
        self._min_det = min_detection_confidence
        self._assumed_confidence = assumed_confidence
        self._mesh: Any = None

    def _ensure(self) -> Any:
        if self._mesh is None:
            try:
                import mediapipe as mp  # type: ignore[import-not-found,unused-ignore]
            except ImportError as exc:
                raise ImportError(
                    "mediapipe is required for FaceMeshAdapter: install the 'landmarks' extra"
                ) from exc
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self._max_faces,
                min_detection_confidence=self._min_det,
            )
        return self._mesh

    def extract(self, image_rgb: NDArray[np.uint8]) -> LandmarkResult:
        height, width = image_rgb.shape[:2]
        res = self._ensure().process(image_rgb)
        faces = res.multi_face_landmarks or []
        if not faces:
            return LandmarkResult(None, 0, 0.0)
        mesh = np.array([[p.x, p.y, p.z] for p in faces[0].landmark], dtype=np.float64)
        return LandmarkResult(to_compact(mesh, width, height), len(faces), self._assumed_confidence)
