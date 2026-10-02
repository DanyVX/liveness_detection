"""Uniform local binary patterns with spatial-grid histograms.

LBP encodes micro-texture (print halftone, paper grain, screen pixel structure, blur)
and is compared over a grid so coarse spatial layout is retained. Implemented in numpy
because scikit-image is not a dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from liveness.features.preprocess import to_gray

_P = 8


def _uniform_table() -> NDArray[np.intp]:
    """Map each 8-bit code to one of 59 bins: 58 uniform patterns plus one catch-all."""
    table = np.empty(256, dtype=np.intp)
    next_bin = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(_P)]
        transitions = sum(bits[i] != bits[(i + 1) % _P] for i in range(_P))
        if transitions <= 2:
            table[code] = next_bin
            next_bin += 1
        else:
            table[code] = -1
    table[table == -1] = next_bin
    return table


_TABLE = _uniform_table()
N_BINS = int(_TABLE.max()) + 1  # 59


def lbp_codes(gray: NDArray[np.float32], radius: int = 1) -> NDArray[np.intp]:
    """Uniform-LBP bin index per interior pixel (P=8 neighbours on a square ring).

    Neighbours are the 8 integer-offset points on the square ring at Chebyshev distance
    `radius` (exactly the classic 3x3 neighbourhood for radius 1); larger radii are a
    coarse approximation of the circular sampling and do not interpolate.
    """
    if radius < 1:
        raise ValueError("radius must be >= 1")
    h, w = gray.shape
    if h <= 2 * radius or w <= 2 * radius:
        raise ValueError("image too small for this LBP radius")
    r = radius
    centre = gray[r : h - r, r : w - r]
    offsets = [(-r, -r), (-r, 0), (-r, r), (0, r), (r, r), (r, 0), (r, -r), (0, -r)]
    code = np.zeros(centre.shape, dtype=np.intp)
    for bit, (dy, dx) in enumerate(offsets):
        neigh = gray[r + dy : h - r + dy, r + dx : w - r + dx]
        code |= (neigh >= centre).astype(np.intp) << bit
    return _TABLE[code]


def lbp_grid_histogram(
    img: NDArray[np.generic], radius: int = 1, grid: tuple[int, int] = (3, 3)
) -> NDArray[np.float64]:
    """Concatenated per-cell histograms, each cell L1-normalised (sums to 1)."""
    bins = lbp_codes(to_gray(img), radius)
    rows = np.array_split(np.arange(bins.shape[0]), grid[0])
    cols = np.array_split(np.arange(bins.shape[1]), grid[1])
    hists = []
    for r_idx in rows:
        for c_idx in cols:
            cell = bins[r_idx[0] : r_idx[-1] + 1, c_idx[0] : c_idx[-1] + 1]
            counts = np.bincount(cell.ravel(), minlength=N_BINS).astype(np.float64)
            hists.append(counts / counts.sum())
    return np.concatenate(hists)


def lbp_features(
    img: NDArray[np.generic],
    radii: Sequence[int] = (1,),
    grid: tuple[int, int] = (3, 3),
) -> NDArray[np.float64]:
    """Uniform LBP (P=8) at one or several radii; length = len(radii) * cells * 59."""
    if not radii:
        raise ValueError("radii must be non-empty")
    return np.concatenate([lbp_grid_histogram(img, r, grid) for r in radii])
