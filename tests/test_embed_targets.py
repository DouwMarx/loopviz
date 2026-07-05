import numpy as np
import pytest

from playlistviz.embed import display, embed_image, lift
from playlistviz.operator import PlaylistOperator
from playlistviz.render import mean_image, pool_rows
from playlistviz.targets import (N_PARAMS_2D, RANGES_2D, generate_target,
                                 theta2d_to_params)


@pytest.fixture
def op():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8000, 5))
    X /= np.linalg.norm(X, axis=0)
    return PlaylistOperator.from_songs(X)


# -- targets ------------------------------------------------------------------

def test_target_generator_shape_range_determinism():
    p = theta2d_to_params(np.zeros(N_PARAMS_2D))
    a = generate_target(p, 128)
    b = generate_target(p, 128)
    assert a.shape == (128, 128)
    assert a.min() >= 0 and a.max() <= 1
    assert np.array_equal(a, b)


def test_target_theta_ranges():
    p = theta2d_to_params(np.full(N_PARAMS_2D, 30.0))
    for name, val in p.items():
        assert val == pytest.approx(RANGES_2D[name][1], rel=1e-3)
    with pytest.raises(ValueError):
        theta2d_to_params(np.zeros(3))


def _neutral():
    return dict(theta2d_to_params(np.zeros(N_PARAMS_2D)),
                fig_mix=0.0, lic_mix=0.0,
                warp_amp=0.0, ridge_amount=0.0, gamma=1.0, vignette=0.0)


def test_target_beta_controls_smoothness():
    smooth = dict(_neutral(), beta=3.2, mix=0.0)
    rough = dict(smooth, beta=1.2)
    g_s = np.abs(np.diff(generate_target(smooth, 128), axis=0)).mean()
    g_r = np.abs(np.diff(generate_target(rough, 128), axis=0)).mean()
    assert g_s < g_r


def test_flow_smear_changes_field():
    base = dict(_neutral(), beta=1.5)
    smeared = dict(base, lic_mix=0.9, lic_len=0.04)
    a = generate_target(base, 128)
    b = generate_target(smeared, 128)
    assert not np.allclose(a, b)
    # smearing averages along streamlines: reduces gradient magnitude
    assert np.abs(np.diff(b, axis=0)).mean() < np.abs(np.diff(a, axis=0)).mean()


def test_figure_ground_layer_changes_field():
    base = _neutral()
    fig = dict(base, fig_mix=0.9, fig_thresh=0.5, fig_feather=0.02)
    assert not np.allclose(generate_target(base, 128), generate_target(fig, 128))


# -- embed ---------------------------------------------------------------------

def test_lift_is_exact_pooling_right_inverse():
    rng = np.random.default_rng(1)
    D, P, q = 4000, 40, 3
    target = rng.standard_normal((P, q))
    assert np.allclose(pool_rows(lift(target, D, P), P), target, atol=1e-10)


def test_embed_matches_target(op):
    rng = np.random.default_rng(2)
    P = 40
    from scipy.ndimage import gaussian_filter
    T = gaussian_filter(rng.random((P, P)), 3)
    T = (T - T.min()) / (T.max() - T.min())
    res = embed_image(op, T, rank=P)
    assert res.rel_error < 0.02
    # achieved pooled image really is what a fresh render of A produces
    L, R = op.factors(U=res.U, Vp=res.Vp, scale=res.scale)
    assert np.allclose(mean_image(L, R, P), res.achieved, atol=1e-8)


def test_embed_preserves_playback_exactly(op):
    rng = np.random.default_rng(3)
    T = rng.random((32, 32))
    res = embed_image(op, T, rank=32)
    assert op.playback_error(U=res.U, Vp=res.Vp, scale=res.scale) < 1e-8


def test_embed_rank_truncation_degrades_gracefully(op):
    rng = np.random.default_rng(4)
    T = rng.random((40, 40))  # white noise target: hardest case for low rank
    full = embed_image(op, T, rank=40)
    trunc = embed_image(op, T, rank=10)
    assert full.rel_error < trunc.rel_error
    assert trunc.rel_error < 1.0


def test_embed_rejects_bad_targets(op):
    with pytest.raises(ValueError):
        embed_image(op, np.ones((16, 16)))  # constant
    with pytest.raises(ValueError):
        embed_image(op, np.zeros((16, 8)))  # not square


def test_display_normalizes():
    img = display(np.random.default_rng(5).standard_normal((20, 20)) * 7 + 3)
    assert img.min() >= 0 and img.max() <= 1
    assert img.std() > 0.1
