"""Interpolation reproduces the cell values at cell centres for every
method, the surface lands in [t - relief, t], slopes are linear in the
relief and fit_relief returns the analytic value for a ramp."""

import numpy as np
import pytest

from loopviz.print3d.plate import PlateParams
from loopviz.print3d.slab import (
    ON_EDGE,
    mesh_metrics,
    overhang_angles,
    slab_mesh,
    slab_mesh_tagged,
    transform,
)
from loopviz.print3d.surface import (
    METHODS,
    cell_values,
    critical_relief,
    facet_angle,
    fit_relief,
    interpolate,
    slope_stats,
    surface,
    tri_gradients,
)


def test_cell_values_clip_and_smooth():
    A = np.zeros((5, 5))
    A[2, 2] = 10.0
    A[0, 0] = -1.0
    v = cell_values(A, clip_pct=100.0)
    assert v[2, 2] == 1.0 and v[0, 0] == -0.1
    v = cell_values(A, clip_pct=50.0)         # median of |A| is 0 -> everything clips
    assert set(np.unique(v)) <= {-1.0, 0.0, 1.0}
    vs = cell_values(A, clip_pct=100.0, sigma_cells=1.0)
    assert vs[2, 2] == 1.0 and np.abs(vs).max() == 1.0       # renormalised after the blur
    assert vs[2, 1] > 0.3 and vs[0, 0] < 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("subdiv", [2, 4])
def test_interpolation_hits_cell_centres(method, subdiv):
    rng = np.random.default_rng(0)
    v = rng.standard_normal((6, 9))
    Z = interpolate(v, subdiv, method)
    assert Z.shape == (6 * subdiv + 1, 9 * subdiv + 1)
    ii = np.arange(6) * subdiv + subdiv // 2          # even subdiv: nodes at centres
    jj = np.arange(9) * subdiv + subdiv // 2
    assert np.allclose(Z[np.ix_(ii, jj)], v, atol=1e-12)


def test_nearest_and_linear_stay_in_range_and_flat_is_flat():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((7, 5))
    for m in ("nearest", "linear"):
        s = surface(A, 2.0, thickness_mm=10.0, relief_mm=4.0, subdiv=3, interp=m)
        assert s.Z.min() >= 6.0 - 1e-12 and s.Z.max() <= 10.0 + 1e-12
        assert s.x[-1] == pytest.approx(10.0) and s.y[-1] == pytest.approx(14.0)
        assert s.x.size == 16 and s.y.size == 22
    s = surface(np.zeros((4, 4)), 2.0, 10.0, 4.0)
    assert np.allclose(s.Z, 10.0)                     # flat: everything is the highest point


def test_overshoot_is_scaled_not_clipped():
    A = -np.ones((8, 8))
    A[3, 3] = A[5, 2] = 1.0                           # spikes: the cubic overshoots
    s = surface(A, 2.0, 10.0, 6.0, interp="cubic")
    assert s.Z.max() == pytest.approx(10.0) and s.scale == 1.0 and s.relief_mm == 6.0
    assert s.printed_range_mm > 6.0                   # cubic overshoots the data
    assert s.printed_range_mm == pytest.approx(s.Z.max() - s.Z.min())
    c = surface(A, 2.0, 10.0, 6.0, interp="cubic", max_range_mm=6.0)
    assert c.printed_range_mm == pytest.approx(6.0) and c.Z.min() == pytest.approx(4.0)
    assert c.scale == pytest.approx(6.0 / s.printed_range_mm) and c.relief_mm == pytest.approx(6.0 * c.scale)
    assert np.allclose(c.Z - 10.0, (s.Z - 10.0) * c.scale)     # a pure scaling about the top
    lin = surface(A, 2.0, 10.0, 6.0, subdiv=2, interp="linear", max_range_mm=6.0)   # nodes on centres
    assert lin.scale == 1.0 and lin.printed_range_mm == pytest.approx(6.0)


def test_fourier_is_exact_for_an_even_cosine():
    n = 8
    x = np.arange(n) + 0.5
    v = np.cos(np.pi * x / n)[None, :] * np.ones((4, 1))
    Z = interpolate(v, 3, "fourier")
    xs = np.arange(n * 3 + 1) / 3
    assert np.allclose(Z[0], np.cos(np.pi * xs / n), atol=1e-12)


def test_slopes_scale_with_relief():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((8, 8))
    s1 = slope_stats(surface(A, 2.0, 10.0, 1.0))
    s2 = slope_stats(surface(A, 2.0, 10.0, 3.0))
    assert s2["abs_dy"]["q99"] == pytest.approx(3 * s1["abs_dy"]["q99"])
    assert s2["grad"]["max"] == pytest.approx(3 * s1["grad"]["max"])
    assert 0.0 <= s1["frac_dy_over_limit"] <= 1.0
    assert s1["frac_facet_over_limit"] <= s1["frac_dy_over_limit"]   # facet rule is milder
    assert 0.3 < s1["frac_hanging"] < 0.7


def test_fit_relief_ramp_is_analytic():
    # v runs -1..1 over 9 cell pitches = 18 mm: slope (relief/2) * 2/18 = relief/18
    A = np.linspace(-1, 1, 10)[:, None] * np.ones((1, 12))
    for rule in ("y", "facet"):                       # rising in y: both rules agree
        assert fit_relief(A, 2.0, interp="linear", rule=rule) == pytest.approx(18.0)
        assert fit_relief(A, 2.0, max_overhang_deg=60.0, interp="linear",
                          rule=rule) == pytest.approx(18 * np.sqrt(3))
    assert fit_relief(A, 2.0, interp="linear", rule="x") == float("inf")
    assert fit_relief(-A, 2.0, interp="linear", rule="y") == pytest.approx(18.0)
    assert fit_relief(-A, 2.0, interp="linear", rule="facet") == float("inf")   # leans back


def test_facet_rule_matches_mesh_metrics_on_a_plane():
    # unit-relief gradient (a, b) with b > 0: the facet rule says the plane
    # reaches alpha at R with R^2 (b^2 - tan^2 a^2) = tan^2; mesh_metrics on
    # the slab printed on edge must report that angle at that relief
    a, b, alpha = 0.3, 0.5, 45.0                  # b > a tan: it does hang eventually
    rc = critical_relief(np.array([a]), np.array([b]), alpha, "facet")[0]
    assert np.isfinite(rc)
    x, y = np.arange(7) * 1.0, np.arange(9) * 1.0
    top = 20.0 + rc * (a * x[None, :] + b * y[:, None])
    V, F = slab_mesh(top, 0.0, x, y)
    m = mesh_metrics(transform(V, ON_EDGE), F)
    assert m["overhang"]["max_deg"] == pytest.approx(alpha, abs=1e-9)
    assert facet_angle(np.array([rc * a]), np.array([rc * b]))[0] == pytest.approx(alpha)
    # x slope alone never hangs; a steep x slope can keep a facet inside forever
    assert critical_relief(np.array([2.0]), np.array([0.0]), alpha)[0] == np.inf
    assert critical_relief(np.array([2.0]), np.array([1.0]), alpha)[0] == np.inf
    assert critical_relief(np.array([2.0]), np.array([1.0]), alpha, "y")[0] == pytest.approx(1.0)


def test_fit_relief_matches_the_mesh_on_a_random_surface():
    # at the fitted relief, the footprint fraction of relief triangles over
    # the limit on the on-edge mesh equals 1 - q (both count the same facets)
    rng = np.random.default_rng(11)
    A = rng.standard_normal((40, 40))
    q = 0.99
    for rule in ("facet", "y"):
        R = fit_relief(A, 2.0, 45.0, q, rule=rule)
        s = surface(A, 2.0, R + 3.0, R)
        assert s.scale == 1.0
        V, F, tag = slab_mesh_tagged(s.Z, 0.0, s.x, s.y)
        Vb = transform(V, ON_EDGE)
        top = F[tag == 0]
        alpha, _, _ = overhang_angles(Vb, top, skip_floor=False)
        n_y = np.abs(np.cross(Vb[top[:, 1]] - Vb[top[:, 0]],
                              Vb[top[:, 2]] - Vb[top[:, 0]])[:, 1])      # footprint = |n_Y| area
        foot = 0.5 * n_y
        d = s.node_mm
        gx, gy = tri_gradients(s.Z, d, d)
        if rule == "facet":
            over = np.nan_to_num(alpha, nan=0.0) > 45.0 + 1e-9
        else:
            over = np.abs(gy) > 1.0 + 1e-9
        frac = foot[over].sum() / foot.sum()
        assert abs(frac - (1 - q)) < 0.1 * (1 - q), (rule, frac)
        if rule == "facet":                      # same angles, triangle by triangle
            mesh_max = np.nanmax(alpha)
            assert mesh_max == pytest.approx(facet_angle(gx, gy).max(), abs=1e-9)


def test_unknown_method():
    with pytest.raises(ValueError):
        interpolate(np.zeros((3, 3)), 2, "spline")


def test_cell_values_mappings():
    A = np.array([[-8.0, -1.0, -0.5, 0.0], [0.25, 0.5, 1.0, 20.0]])
    # clip: the default, identical to the old behaviour
    v = cell_values(A, clip_pct=75.0, mapping="clip")
    m = np.percentile(np.abs(A), 75.0)
    assert np.allclose(v, np.clip(A / m, -1, 1)) and np.allclose(v, cell_values(A, clip_pct=75.0))
    assert v.min() == -1.0 and v.max() == 1.0
    # tanh: monotone soft knee spanning [-1, 1], no flat top below the extremes
    w = cell_values(A, clip_pct=75.0, mapping="tanh")
    assert np.abs(w).max() == 1.0 and w.max() == 1.0 and -1 < w.min() < -0.99   # peak at |A| max
    assert np.all(np.diff(w.ravel()) > 0)                  # strictly increasing with A
    assert np.allclose(w, np.tanh(A / m) / np.tanh(1) / (np.tanh(20 / m) / np.tanh(1)))
    # none: linear in A, the largest entry takes the range
    u = cell_values(A, mapping="none")
    assert np.allclose(u, A / 20.0) and u.max() == 1.0 and u.min() == -0.4
    with pytest.raises(ValueError):
        cell_values(A, mapping="sigmoid")
    with pytest.raises(ValueError):
        PlateParams(2.0, 1, 1, 6.0, 3.0, mapping="sigmoid")
    assert surface(A, 2.0, 6.0, 3.0, mapping="none").Z.shape == (7, 13)
