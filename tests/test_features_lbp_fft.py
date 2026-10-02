import numpy as np
import pytest
from PIL import Image

from liveness.data.synthetic import render_bona_fide, render_print, render_replay
from liveness.features.fft import fft_features, high_band_ratio, power_spectrum
from liveness.features.lbp import N_BINS, lbp_codes, lbp_features, lbp_grid_histogram
from liveness.features.preprocess import normalize_resolution, to_gray

SIZE = 96


def _face(seed: int = 0, size: int = 64) -> np.ndarray:
    return np.asarray(render_bona_fide(np.random.default_rng(seed), size))


def _at(img: np.ndarray, native: int) -> np.ndarray:
    """The same picture stored at another native resolution, then normalised."""
    big = np.asarray(Image.fromarray(img).resize((native, native), Image.Resampling.BICUBIC))
    return normalize_resolution(big, SIZE)


def test_lbp_n_bins_is_59() -> None:
    assert N_BINS == 59


def test_lbp_histograms_sum_to_one_per_cell_and_deterministic() -> None:
    img = normalize_resolution(_face(), SIZE)
    a = lbp_grid_histogram(img, grid=(3, 3))
    b = lbp_grid_histogram(img, grid=(3, 3))
    assert np.array_equal(a, b)
    cells = a.reshape(9, N_BINS)
    assert np.allclose(cells.sum(axis=1), 1.0)
    assert (cells >= 0).all()


def test_lbp_multiscale_length() -> None:
    img = normalize_resolution(_face(), SIZE)
    assert lbp_features(img, radii=(1, 2, 3)).shape == (3 * 9 * N_BINS,)
    assert lbp_features(img).shape == (9 * N_BINS,)


def test_lbp_known_pattern() -> None:
    flat = np.full((5, 5), 7.0, dtype=np.float32)
    codes = lbp_codes(flat)
    assert codes.shape == (3, 3) and len(set(codes.ravel().tolist())) == 1


def test_lbp_errors() -> None:
    img = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="radius"):
        lbp_codes(img, 0)
    with pytest.raises(ValueError, match="small"):
        lbp_codes(np.zeros((4, 4), dtype=np.float32), 2)
    with pytest.raises(ValueError, match="radii"):
        lbp_features(img, radii=())


def test_grayscale_and_rgb_give_same_shaped_features() -> None:
    rgb = normalize_resolution(_face(), SIZE)
    gray = rgb[..., 0]
    assert lbp_features(gray).shape == lbp_features(rgb).shape
    assert fft_features(gray).shape == fft_features(rgb).shape
    gray3 = np.repeat(gray[..., None], 3, axis=2)
    assert np.allclose(lbp_features(gray), lbp_features(gray3), atol=0.02)


@pytest.mark.parametrize("native", [128, 192, 256])
def test_native_resolution_is_not_a_shortcut(native: int) -> None:
    img = _face(3, 64)
    ref, other = _at(img, 64), _at(img, native)
    assert np.abs(lbp_features(ref) - lbp_features(other)).sum() / 9 < 0.3
    assert np.allclose(fft_features(ref)[-2], fft_features(other)[-2], atol=0.02)
    assert np.allclose(fft_features(ref)[:-1], fft_features(other)[:-1], atol=1.0)


def test_fft_high_band_ratio_larger_with_grating() -> None:
    plain = normalize_resolution(_face(1), SIZE)
    xx = np.arange(SIZE)[None, :]
    grating = 25 * np.sin(2 * np.pi * xx / 3.0)
    noisy = np.clip(plain.astype(np.float64) + grating[..., None], 0, 255).astype(np.uint8)
    assert high_band_ratio(noisy) > 5 * high_band_ratio(plain)
    assert fft_features(noisy)[-1] > fft_features(plain)[-1]  # peak prominence


def test_fft_features_shape_and_finite() -> None:
    for n_bands in (4, 8):
        f = fft_features(normalize_resolution(_face(), SIZE), n_bands=n_bands)
        assert f.shape == (n_bands + 2,) and np.isfinite(f).all()
    with pytest.raises(ValueError, match="n_bands"):
        fft_features(_face(), n_bands=1)
    assert power_spectrum(_face()).shape == (64, 64)


def test_blur_lowers_high_band_ratio_synthetic_sanity() -> None:
    rng = np.random.default_rng(2)
    bona = normalize_resolution(np.asarray(render_bona_fide(rng, 64)), SIZE)
    blurred = normalize_resolution(np.asarray(render_print(np.random.default_rng(2), 64)), SIZE)
    assert high_band_ratio(blurred) < high_band_ratio(bona)
    assert np.isfinite(
        fft_features(normalize_resolution(np.asarray(render_replay(rng, 64)), SIZE))
    ).all()
    assert to_gray(bona).shape == (SIZE, SIZE)
