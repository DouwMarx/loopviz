import numpy as np
import pytest

from loopviz.paper import metrics as M


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_feature_vector_shape(rng):
    L = rng.random((128, 128))
    phi = M.features(L)
    assert phi.shape == (M.N_METRICS,)
    assert np.isfinite(phi).all()


def test_beta_white_noise_near_zero(rng):
    # white noise has flat spectrum -> beta ~ 0
    L = rng.random((256, 256))
    assert abs(M.beta_slope(L)) < 0.5


def test_beta_smooth_field_high(rng):
    # heavily smoothed noise -> steep spectral slope
    from scipy.ndimage import gaussian_filter
    L = gaussian_filter(rng.random((256, 256)), sigma=8)
    assert M.beta_slope(L) > 2.0


def test_entropy_bounds(rng):
    assert M.entropy(np.full((64, 64), 0.5)) == pytest.approx(0.0, abs=1e-9)
    assert M.entropy(rng.random((256, 256))) > 7.0  # near-uniform histogram


def test_symmetry_mirror_image(rng):
    half = rng.random((64, 32))
    L = np.concatenate([half, half[:, ::-1]], axis=1)
    assert M.symmetry(L) > 0.99
    assert M.symmetry(rng.random((64, 64))) < 0.3


def test_gini_bounds(rng):
    # constant gradient field -> gini ~ 0; single spike -> gini ~ 1
    ramp = np.outer(np.arange(64.0), np.ones(64))
    assert M.gradient_gini(ramp / 64) < 0.2
    spike = np.zeros((64, 64))
    spike[32, 32] = 1.0
    assert M.gradient_gini(spike) > 0.95


def test_edge_orient_entropy_extremes(rng):
    # single orientation (vertical edges) -> low; isotropic noise -> high
    stripes = np.tile(np.arange(64) % 2, (64, 1)).astype(float)
    assert M.edge_orient_entropy(stripes) < 0.5
    assert M.edge_orient_entropy(rng.random((128, 128))) > 0.9


def test_compress_complexity_ordering(rng):
    flat = np.full((128, 128), 0.5)
    noise = rng.random((128, 128))
    assert M.compress_complexity(flat) < 0.05
    assert M.compress_complexity(noise) > 0.8


def test_lum_skewness_sign():
    x = np.full((64, 64), 0.2)
    x[:8, :8] = 1.0  # a few bright pixels -> positive skew
    assert M.lum_skewness(x) > 1.0
    assert M.lum_skewness(np.full((16, 16), 0.3)) == 0.0


def test_balance_centered_vs_corner():
    centered = np.zeros((65, 65))
    centered[30:35, 30:35] = 1.0
    corner = np.zeros((65, 65))
    corner[:5, :5] = 1.0
    assert M.balance_dcm(centered) < 0.1
    assert M.balance_dcm(corner) > 0.4


def test_fractal_dim_range(rng):
    # any real edge map should land in (0, 2]
    d = M.fractal_dim(rng.random((128, 128)))
    assert 0.0 < d <= 2.1


def test_two_scale_averaging(rng):
    L = rng.random((256, 256))
    phi1 = M.features(L, two_scale=False)
    phi2 = M.features(L, two_scale=True)
    assert not np.allclose(phi1, phi2)  # downsample changes at least one metric
