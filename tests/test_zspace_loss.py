import numpy as np
import pytest

from loopviz.config import OptConfig, ZConfig
from loopviz.loss import (barrier, equal_weights, loss_vector,
                              loss_vector_from_phi_dict, sample_weights,
                              scalar_loss)
from loopviz.metrics import METRIC_NAMES, N_METRICS, TARGETS
from loopviz.operator import PlaylistOperator
from loopviz.optimize import run_es
from loopviz.zspace import (N_PARAMS, PARAM_RANGES, generate_Z,
                                theta_to_params)


# -- zspace ---------------------------------------------------------------

def test_theta_mapping_ranges():
    params = theta_to_params(np.zeros(N_PARAMS))
    for name, val in params.items():
        lo, hi = PARAM_RANGES[name]
        assert lo <= val <= hi
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


def test_locality_localizes_columns():
    """High locality must concentrate column energy; low must spread it."""
    cfg = ZConfig(rank=4, seed=2)
    D = 4000
    base = dict(theta_to_params(np.zeros(N_PARAMS)), ridge_amount=0.0)
    p_global = dict(base, locality=0.0)
    p_local = dict(base, locality=0.97, log_width=-2.5, width_spread=0.0)
    zg = generate_Z(p_global, D, cfg)
    zl = generate_Z(p_local, D, cfg)

    def support_frac(U):
        # fraction of samples holding 90% of the energy, averaged over cols
        fracs = []
        for k in range(U.shape[1]):
            e = np.sort(U[:, k] ** 2)[::-1]
            c = np.cumsum(e) / e.sum()
            fracs.append(np.searchsorted(c, 0.9) / U.shape[0])
        return np.mean(fracs)

    assert support_frac(zl.U) < 0.5 * support_frac(zg.U)


def test_generate_Z_projected_is_invisible():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((800, 6))
    op = PlaylistOperator.from_songs(X)
    params = theta_to_params(rng.standard_normal(N_PARAMS))
    zf = generate_Z(params, op.D, ZConfig(rank=4),
                    project_perp=op.project_perp)
    assert op.playback_error(U=zf.U, Vp=zf.Vp, scale=zf.scale) < 1e-8


# -- loss -------------------------------------------------------------------

def test_loss_zero_at_targets():
    assert np.allclose(loss_vector(TARGETS.copy()), 0.0)


def test_loss_vector_from_phi_dict_ignores_stale_names():
    phi = {name: float(t) for name, t in zip(METRIC_NAMES, TARGETS)}
    phi["colorfulness"] = 55.0  # stale metric from an older candidate file
    assert np.allclose(loss_vector_from_phi_dict(phi), 0.0)


def test_scalar_loss_weighting():
    phi = TARGETS.copy()
    phi[0] += 0.4  # one scale unit off on beta -> squared residual 1
    w = np.zeros(N_METRICS)
    w[0] = 1.0
    assert scalar_loss(phi, w) == pytest.approx(1.0, abs=1e-6)
    w_other = np.zeros(N_METRICS)
    w_other[1] = 1.0
    assert scalar_loss(phi, w_other) == pytest.approx(0.0, abs=1e-6)


def test_chebyshev_scalarization():
    phi = TARGETS.copy()
    phi[0] += 0.4   # loss 1 on metric 0
    phi[1] += 0.15  # loss 1 on metric 1
    w = np.full(N_METRICS, 1.0 / N_METRICS)
    cheb = scalar_loss(phi, w, scalarization="chebyshev")
    # max term = 1/N + augmentation 0.05 * 2
    assert cheb == pytest.approx(1.0 / N_METRICS + 0.05 * 2.0, abs=1e-6)
    with pytest.raises(ValueError):
        scalar_loss(phi, w, scalarization="nope")


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
    target = np.full(N_PARAMS, 0.7)

    class R:
        def __init__(self, theta, loss):
            self.theta, self.loss = theta, loss

    def objective(theta):
        return R(theta, float(((theta - target) ** 2).sum()))

    cfg = OptConfig(generations=25, population=10, seed=1, subspace_rank=0)
    best, hist = run_es(objective, cfg)
    assert hist.best_loss[-1] < hist.best_loss[0] * 0.25
    assert hist.evaluations == 1 + 25 * 10


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
