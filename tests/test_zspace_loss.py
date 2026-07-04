import numpy as np
import pytest

from playlistviz.config import OptConfig, ZConfig
from playlistviz.loss import (barrier, equal_weights, loss_vector,
                              sample_weights, scalar_loss)
from playlistviz.metrics import METRIC_NAMES, N_METRICS, TARGETS
from playlistviz.operator import PlaylistOperator
from playlistviz.optimize import run_es
from playlistviz.zspace import (N_PARAMS, PARAM_RANGES, generate_Z,
                                song_envelopes, theta_to_params)


# -- zspace ---------------------------------------------------------------

def test_theta_mapping_ranges():
    params = theta_to_params(np.zeros(N_PARAMS))
    for name, val in params.items():
        lo, hi = PARAM_RANGES[name]
        assert lo <= val <= hi
    # extreme thetas saturate near the bounds
    hi_params = theta_to_params(np.full(N_PARAMS, 20.0))
    for name, val in hi_params.items():
        assert val == pytest.approx(PARAM_RANGES[name][1], rel=1e-3)


def test_theta_wrong_shape_raises():
    with pytest.raises(ValueError):
        theta_to_params(np.zeros(3))


def test_generate_Z_deterministic():
    params = theta_to_params(np.zeros(N_PARAMS))
    cfg = ZConfig(rank=4, seed=1)
    z1 = generate_Z(params, 600, cfg)
    z2 = generate_Z(params, 600, cfg)
    assert np.array_equal(z1.U, z2.U)
    assert np.array_equal(z1.Vp, z2.Vp)


def test_generate_Z_theta_sensitivity():
    cfg = ZConfig(rank=4, seed=1)
    a = generate_Z(theta_to_params(np.zeros(N_PARAMS)), 600, cfg)
    b = generate_Z(theta_to_params(np.full(N_PARAMS, 2.0)), 600, cfg)
    assert not np.allclose(a.U, b.U)


def test_generate_Z_projected_is_invisible():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((800, 6))
    op = PlaylistOperator.from_songs(X)
    params = theta_to_params(rng.standard_normal(N_PARAMS))
    zf = generate_Z(params, op.D, ZConfig(rank=4), envelopes=song_envelopes(X),
                    project_perp=op.project_perp)
    assert op.playback_error(U=zf.U, Vp=zf.Vp, scale=zf.scale) < 1e-8


def test_song_envelopes_shape_and_mean():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5000, 3))
    env = song_envelopes(X, smooth=100)
    assert env.shape == X.shape
    assert np.allclose(env.mean(axis=0), 1.0, atol=0.05)
    assert (env >= 0).all()


# -- loss -------------------------------------------------------------------

def test_loss_zero_at_targets():
    assert np.allclose(loss_vector(TARGETS.copy()), 0.0)


def test_scalar_loss_weighting():
    phi = TARGETS.copy()
    phi[0] += 0.4  # one scale unit off on beta -> squared residual 1
    w = np.zeros(N_METRICS)
    w[0] = 1.0
    assert scalar_loss(phi, w) == pytest.approx(1.0, abs=1e-6)
    w_other = np.zeros(N_METRICS)
    w_other[1] = 1.0
    assert scalar_loss(phi, w_other) == pytest.approx(0.0, abs=1e-6)


def test_barrier_activates_on_flat_image_stats():
    phi = TARGETS.copy()
    assert barrier(phi) == 0.0
    phi[METRIC_NAMES.index("rms_contrast")] = 0.01  # washed out
    assert barrier(phi) > 10.0


def test_dirichlet_weights_on_simplex():
    rng = np.random.default_rng(0)
    for alpha in (0.5, 2.0, 8.0, float("inf")):
        w = sample_weights(alpha, rng)
        assert w.shape == (N_METRICS,)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()
    assert np.allclose(sample_weights(float("inf"), rng), equal_weights())


# -- ES ---------------------------------------------------------------

def test_es_improves_toy_objective():
    """ES should approach the minimum of a shifted quadratic."""
    target = np.full(N_PARAMS, 0.7)

    class R:
        def __init__(self, theta, loss):
            self.theta, self.loss = theta, loss

    def objective(theta):
        return R(theta, float(((theta - target) ** 2).sum()))

    cfg = OptConfig(generations=20, population=8, seed=1, subspace_rank=0)
    best, hist = run_es(objective, cfg)
    assert hist.best_loss[-1] < hist.best_loss[0] * 0.2
    assert hist.evaluations == 1 + 20 * 8


def test_es_subspace_search_runs():
    def objective(theta):
        class R:
            pass
        r = R()
        r.theta, r.loss = theta, float((theta**2).sum())
        return r

    cfg = OptConfig(generations=3, population=4, seed=0, subspace_rank=8)
    best, _ = run_es(objective, cfg)
    assert best.theta.shape == (N_PARAMS,)
