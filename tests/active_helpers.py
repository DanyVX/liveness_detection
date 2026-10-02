"""Synthetic landmark generators for active-liveness tests (compact 21-point layout)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from liveness.active.challenge import Challenge, ChallengeKind, ChallengeStep
from liveness.active.session import FrameObservation

OPEN_EAR = 0.3


def make_landmarks(
    eye_open: float = 1.0,
    yaw: float = 0.0,
    mouth_open: float = 0.0,
    *,
    mirrored: bool = False,
    scale: float = 100.0,
    center: tuple[float, float] = (320.0, 240.0),
) -> NDArray[np.float64]:
    """EAR = 0.3 * eye_open, MAR = mouth_open, yaw in the subject frame (+ = subject's left)."""
    cx, cy = center
    pts: list[tuple[float, float]] = []
    for ex in (cx - 0.4 * scale, cx + 0.4 * scale):
        w = 0.3 * scale
        v = OPEN_EAR * w * eye_open
        y = cy - 0.3 * scale
        pts += [
            (ex - w / 2, y),
            (ex - w / 4, y - v / 2),
            (ex + w / 4, y - v / 2),
            (ex + w / 2, y),
            (ex + w / 4, y + v / 2),
            (ex - w / 4, y + v / 2),
        ]
    mw = 0.5 * scale
    mv = mouth_open * mw
    my = cy + 0.5 * scale
    pts += [
        (cx - mw / 2, my),
        (cx - mw / 4, my - mv / 2),
        (cx + mw / 4, my - mv / 2),
        (cx + mw / 2, my),
        (cx + mw / 4, my + mv / 2),
        (cx - mw / 4, my + mv / 2),
    ]
    pts.append((cx + yaw * scale, cy))
    pts += [(cx - scale, cy), (cx + scale, cy)]
    arr = np.asarray(pts, dtype=np.float64)
    if mirrored:
        arr[:, 0] = 2 * cx - arr[:, 0]
    return arr


def blink_profile(t: float, start: float, duration: float) -> float:
    """Eye openness in [0.05, 1]: 30% closing, 40% closed, 30% opening."""
    u = (t - start) / duration
    if u < 0 or u > 1:
        return 1.0
    if u < 0.3:
        return 1 - 0.95 * (u / 0.3)
    if u < 0.7:
        return 0.05
    return 0.05 + 0.95 * ((u - 0.7) / 0.3)


@dataclass
class Script:
    duration_ms: float = 6000.0
    fps: float = 30.0
    blinks: list[tuple[float, float]] = field(default_factory=list)  # (start_ms, duration_ms)
    yaw_keys: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0)])
    mouth_keys: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0)])
    mirrored: bool = False
    t0: float = 0.0

    def landmarks_at(self, t: float) -> NDArray[np.float64]:
        openness = min([1.0] + [blink_profile(t, s, d) for s, d in self.blinks])
        yaw = float(np.interp(t, *zip(*self.yaw_keys, strict=True)))
        mouth = float(np.interp(t, *zip(*self.mouth_keys, strict=True)))
        return make_landmarks(openness, yaw, mouth, mirrored=self.mirrored)

    def times(self) -> list[float]:
        n = int(self.duration_ms * self.fps / 1000.0)
        return [i * 1000.0 / self.fps for i in range(n)]

    def landmark_frames(self) -> list[tuple[float, NDArray[np.float64]]]:
        return [(self.t0 + t, self.landmarks_at(t)) for t in self.times()]

    def frames(self) -> list[FrameObservation]:
        return [FrameObservation(t, lm, 1) for t, lm in self.landmark_frames()]


def make_challenge(
    steps: list[ChallengeStep] | list[ChallengeKind],
    *,
    nonce: str = "a" * 64,
    issued_at: float = 1000.0,
    ttl_s: float = 30.0,
) -> Challenge:
    seq = tuple(s if isinstance(s, ChallengeStep) else ChallengeStep(s) for s in steps)
    return Challenge(nonce, issued_at, issued_at + ttl_s, seq)
