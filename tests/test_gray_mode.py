import numpy as np
import pytest

from playlistviz import bt
from playlistviz.loss import (GRAY_METRICS, equal_weights, metric_mask,
                              sample_weights)
from playlistviz.metrics import METRIC_NAMES, N_METRICS
from playlistviz.render import render
from playlistviz.zspace import N_PARAMS, theta_to_params

COLOR_METRICS = {"colorfulness", "hue_dispersion", "mean_saturation"}


def test_gray_mask_excludes_color_metrics():
    mask = metric_mask(gray=True)
    for i, name in enumerate(METRIC_NAMES):
        assert mask[i] == (name not in COLOR_METRICS)
    assert metric_mask(gray=False).all()
    assert set(GRAY_METRICS) == set(METRIC_NAMES) - COLOR_METRICS


def test_masked_equal_weights():
    mask = metric_mask(gray=True)
    w = equal_weights(mask)
    assert w.sum() == pytest.approx(1.0)
    assert (w[~mask] == 0).all()
    assert np.allclose(w[mask], 1.0 / mask.sum())


def test_masked_dirichlet_weights():
    mask = metric_mask(gray=True)
    rng = np.random.default_rng(0)
    for alpha in (0.5, 2.0, float("inf")):
        w = sample_weights(alpha, rng, mask)
        assert w.sum() == pytest.approx(1.0)
        assert (w[~mask] == 0).all()


def test_gray_render_channels_equal():
    rng = np.random.default_rng(1)
    L = rng.standard_normal((512, 6))
    R = rng.standard_normal((512, 6))
    params = theta_to_params(rng.standard_normal(N_PARAMS))
    img = render(L, R, 64, params, gray=True)
    assert np.array_equal(img[..., 0], img[..., 1])
    assert np.array_equal(img[..., 1], img[..., 2])
    # and non-degenerate
    assert img.std() > 0


def test_bt_fit_with_active_mask_zeroes_inactive():
    rng = np.random.default_rng(2)
    mask = metric_mask(gray=True)
    loss_vectors = {f"c{i}": rng.random(N_METRICS) * 2 for i in range(12)}
    w_true = np.zeros(N_METRICS)
    w_true[0] = 5.0
    comps = []
    ids = sorted(loss_vectors)
    for _ in range(200):
        i, j = rng.choice(len(ids), 2, replace=False)
        a, b = ids[i], ids[j]
        z = w_true @ (loss_vectors[b] - loss_vectors[a])
        winner, loser = (a, b) if rng.random() < 1 / (1 + np.exp(-z)) else (b, a)
        comps.append(bt.Comparison(winner=winner, loser=loser))
    fit = bt.fit_bt(comps, loss_vectors, active=mask)
    assert np.allclose(fit.w_raw[~mask], 0.0, atol=1e-6)
    assert np.argmax(fit.w_simplex) == 0
