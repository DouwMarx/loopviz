import numpy as np
import pytest

from loopviz.render import (block_grams, energy_image, mean_image,
                                pool_rows, render, save_png)
from loopviz.zspace import PARAM_NAMES, theta_to_params


@pytest.fixture
def params():
    return theta_to_params(np.zeros(len(PARAM_NAMES)))


def _dense_reference(L, R, P):
    """Ground truth by materializing A (small D only)."""
    A = L @ R.T
    D = A.shape[0]
    k = D // P
    A = A[: P * k, : P * k]
    blocks = A.reshape(P, k, P, k)
    return blocks.mean(axis=(1, 3)), (blocks**2).mean(axis=(1, 3))


def test_streamed_images_match_dense():
    rng = np.random.default_rng(0)
    D, r, P = 240, 5, 16
    L = rng.standard_normal((D, r))
    R = rng.standard_normal((D, r))
    mean_ref, energy_ref = _dense_reference(L, R, P)
    assert np.allclose(mean_image(L, R, P), mean_ref, atol=1e-10)
    assert np.allclose(energy_image(L, R, P), energy_ref, atol=1e-10)


def test_strided_images_approximate_exact():
    # smooth factors (the realistic case: audio and 1/f noise are correlated
    # over the ~hundreds of samples inside one block)
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(1)
    D, r, P = 6400, 4, 16
    L = gaussian_filter1d(rng.standard_normal((D, r)), sigma=20, axis=0)
    R = gaussian_filter1d(rng.standard_normal((D, r)), sigma=20, axis=0)
    exact = energy_image(L, R, P)
    approx = energy_image(L, R, P, stride=4)
    # correlation with exact must stay high; stride is an ES-loop shortcut
    c = np.corrcoef(exact.ravel(), approx.ravel())[0, 1]
    assert c > 0.95


def test_pool_rows_block_mean():
    F = np.arange(12, dtype=float).reshape(12, 1)
    out = pool_rows(F, 3)  # blocks of 4: means 1.5, 5.5, 9.5
    assert np.allclose(out.ravel(), [1.5, 5.5, 9.5])


def test_block_grams_psd():
    rng = np.random.default_rng(1)
    G = block_grams(rng.standard_normal((100, 4)), 10)
    for p in range(10):
        eig = np.linalg.eigvalsh(G[p])
        assert eig.min() > -1e-12


def test_render_shape_and_range(params):
    rng = np.random.default_rng(2)
    L = rng.standard_normal((512, 6))
    R = rng.standard_normal((512, 6))
    img = render(L, R, 64, params)
    assert img.shape == (64, 64)
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert np.isfinite(img).all()


def test_render_deterministic(params):
    rng = np.random.default_rng(3)
    L = rng.standard_normal((256, 4))
    R = rng.standard_normal((256, 4))
    assert np.array_equal(render(L, R, 32, params), render(L, R, 32, params))


def test_resolution_exceeding_dim_raises(params):
    with pytest.raises(ValueError):
        render(np.ones((16, 2)), np.ones((16, 2)), 32, params)


@pytest.mark.parametrize("bits", [8, 16])
def test_save_png_roundtrip(tmp_path, bits):
    from PIL import Image

    img = np.random.default_rng(4).random((20, 20))
    path = tmp_path / f"out{bits}.png"
    save_png(img, path, bit_depth=bits)
    loaded = np.asarray(Image.open(path))
    assert loaded.shape == (20, 20)
    maxval = 65535 if loaded.dtype == np.uint16 else 255
    assert np.allclose(loaded / maxval, img, atol=1.5 / 255)


def test_save_png_rejects_rgb(tmp_path):
    with pytest.raises(ValueError):
        save_png(np.zeros((8, 8, 3)), tmp_path / "x.png")
