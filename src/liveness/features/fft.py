"""Frequency-domain features on a fixed-size grayscale image.

The image is mean-removed and Hann-windowed to limit spectral leakage, then the 2-D
power spectrum is summarised radially.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from liveness.features.preprocess import to_gray

HIGH_BAND_START = 0.5  # fraction of Nyquist
_MIN_RADIUS = 0.04
_EPS = 1e-12


def power_spectrum(img: NDArray[np.generic]) -> NDArray[np.float64]:
    """Centred 2-D power spectrum of the mean-removed, Hann-windowed grayscale image."""
    gray = to_gray(img).astype(np.float64)
    gray = gray - gray.mean()
    win = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    spec = np.fft.fftshift(np.fft.fft2(gray * win))
    return np.asarray(np.abs(spec) ** 2, dtype=np.float64)


def _radius(shape: tuple[int, int]) -> NDArray[np.float64]:
    """Radial frequency as a fraction of Nyquist (1.0 = Nyquist on the axes)."""
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0])) / 0.5
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1])) / 0.5
    return np.asarray(np.hypot(fy[:, None], fx[None, :]), dtype=np.float64)


def fft_features(img: NDArray[np.generic], n_bands: int = 8) -> NDArray[np.float64]:
    """Feature vector of length n_bands + 2.

    - n_bands log-spaced radial band energies, as log10 of the band's share of the total
      energy: the spectral slope. Blur (print, defocus) steepens it; sharpening and
      re-capture noise flatten it.
    - high-band ratio: energy beyond HIGH_BAND_START of Nyquist over total energy;
      detail loss (blur) lowers it and screen/halftone texture raises it.
    - high-band peak prominence: log10 of the strongest bin over the median bin in the
      high band; a periodic pattern such as moire or a pixel grid shows up as an isolated
      peak even when it adds little total energy.
    """
    if n_bands < 2:
        raise ValueError("n_bands must be >= 2")
    spec = power_spectrum(img)
    rad = _radius(spec.shape)
    inside = (rad >= _MIN_RADIUS) & (rad <= 1.0)
    total = spec[inside].sum() + _EPS
    edges = np.logspace(np.log10(_MIN_RADIUS), 0.0, n_bands + 1)
    bands = np.empty(n_bands)
    for i in range(n_bands):
        hi_inclusive = i == n_bands - 1
        mask = (rad >= edges[i]) & ((rad <= edges[i + 1]) if hi_inclusive else (rad < edges[i + 1]))
        bands[i] = np.log10(spec[mask].sum() / total + _EPS)
    high = (rad >= HIGH_BAND_START) & (rad <= 1.0)
    high_ratio = spec[high].sum() / total
    vals = spec[high]
    prominence = np.log10(np.max(vals, initial=0.0) + _EPS) - np.log10(np.median(vals) + _EPS)
    return np.concatenate([bands, [high_ratio, prominence]])


def high_band_ratio(img: NDArray[np.generic]) -> float:
    """Convenience accessor for the high-band energy ratio."""
    return float(fft_features(img)[-2])
