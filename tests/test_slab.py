"""slab_mesh is a closed 2-manifold with outward winding and the exact
triangulated volume; labels engrave inside the margin and mirror; the
on-edge transform is a proper rotation; overhang metrics are exact on a
wedge."""

import numpy as np
import pytest

from loopviz.print3d.relief import check_mesh, mesh_volume, read_stl, write_stl
from loopviz.print3d.slab import (
    ON_EDGE,
    concat,
    engrave_text,
    homogeneous,
    mesh_metrics,
    slab_mesh,
    slab_mesh_tagged,
    slab_volume,
    stl_vertices,
    text_mask,
    transform,
    trapezoid_volume,
)


def _grid(ny=5, nx=7, d=0.5):
    return np.arange(nx) * d, np.arange(ny) * d


@pytest.mark.parametrize("flat", [True, False])
def test_slab_is_watertight_and_exact(flat):
    trimesh = pytest.importorskip("trimesh")
    rng = np.random.default_rng(0)
    x, y = _grid()
    top = 6.0 + rng.standard_normal((5, 7))
    bottom = 0.0 if flat else 0.4 * rng.random((5, 7))
    V, F, tag = slab_mesh_tagged(top, bottom, x, y)
    m = trimesh.Trimesh(V, F, process=False)
    assert m.is_watertight and m.is_winding_consistent and m.is_volume
    c = check_mesh(V, F)
    assert c["closed"] and c["oriented"] and c["nonmanifold_edges"] == 0
    # every edge in exactly two faces
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert (cnt == 2).all()
    vol = slab_volume(top, bottom, x, y)
    assert vol > 0 and mesh_volume(V, F) == pytest.approx(vol, rel=1e-12)
    assert m.volume == pytest.approx(vol, rel=1e-9)
    # trapezoid differs only by the corner term dx dy / 12 * (corners)
    h = top - bottom
    corner = 0.5 * 0.5 / 12 * (h[0, 0] - h[0, -1] - h[-1, 0] + h[-1, -1])
    assert trapezoid_volume(top, bottom, x, y) + corner == pytest.approx(vol, rel=1e-12)
    assert set(np.unique(tag)) == {0, 1, 2}
    assert (tag == 0).sum() == 2 * 4 * 6
    assert (tag == 1).sum() == (20 if flat else 48)


def test_slab_rejects_bad_input():
    x, y = _grid()
    with pytest.raises(ValueError):
        slab_mesh(np.zeros((5, 7)), 0.0, x, y)            # top not above bottom
    with pytest.raises(ValueError):
        slab_mesh(np.ones((5, 6)), 0.0, x, y)             # shape mismatch


def test_text_mask_shape_and_glyph():
    m = text_mask("B2", 9)
    plain = text_mask("B2", 9, glyph_up=False)
    assert m.shape[0] == 9 and m.any() and plain.shape[1] < m.shape[1]
    assert np.array_equal(m[:, :plain.shape[1]], plain)   # text first, then the glyph
    g = m[:, plain.shape[1]:].sum(axis=1)                 # triangle: apex up, base down
    assert g[0] <= 2 and g[-1] >= 5 and (np.diff(g) >= 0).all()


def test_engrave_text_is_mirrored_inside_margin_and_recessed():
    x, y = np.arange(40) * 0.5, np.arange(24) * 0.5
    b = engrave_text(0.0, x, y, "A3", 5.0, 0.6, anchor_xy=(2.0, 10.0))
    assert b.shape == (24, 40) and set(np.unique(b)) == {0.0, 0.6}
    assert b[0].max() == 0 and b[-1].max() == 0 and b[:, 0].max() == 0 and b[:, -1].max() == 0
    assert b.max(axis=0).nonzero()[0].min() >= 4           # starts at the anchor column
    bm = engrave_text(0.0, x, y, "A3", 5.0, 0.6, anchor_xy=(2.0, 10.0), mirror_x=False)
    w = max(np.flatnonzero(b.any(axis=0)).max(), np.flatnonzero(bm.any(axis=0)).max()) - 3
    assert np.array_equal(b[:, 4:4 + w], bm[:, 4:4 + w][:, ::-1])
    with pytest.raises(ValueError):
        engrave_text(0.0, x, y, "A3", 5.0, 0.6, anchor_xy=(0.0, 11.5))   # touches the edge


def test_on_edge_is_a_proper_rotation():
    assert np.linalg.det(ON_EDGE) == pytest.approx(1.0)
    assert np.allclose(ON_EDGE @ [1, 2, 3], [1, -3, 2])
    V = np.array([[1.0, 2.0, 3.0]])
    M = homogeneous(ON_EDGE, (10, 20, 30))
    assert np.allclose(transform(V, M), [[11, 17, 32]])
    assert np.allclose(transform(V, ON_EDGE, (10, 20, 30)), [[11, 17, 32]])
    assert np.allclose(transform(transform(V, M), np.linalg.inv(M)), V)


def test_wedge_overhang_area_is_exact():
    th = np.radians(60)
    x, y = np.linspace(0, 10, 3), np.linspace(0, 20, 5)
    top = 5 + np.tan(th) * y[:, None] * np.ones((1, 3))
    V, F = slab_mesh(top, 0.0, x, y)
    m = mesh_metrics(transform(V, ON_EDGE), F)
    expect = 10 * 20 / np.cos(th)
    assert m["overhang"]["area_over_45_mm2"] == pytest.approx(expect)
    assert sum(m["overhang"]["area_mm2"]) == pytest.approx(expect)
    assert m["overhang"]["max_deg"] == pytest.approx(60.0)
    assert m["floor_area_mm2"] == pytest.approx(10 * 5)     # the z = 0 edge on the bed
    assert m["volume_mm3"] == pytest.approx(slab_volume(top, 0.0, x, y))
    assert m["mass_g"] == pytest.approx(m["volume_mm3"] * 1.24e-3)
    flat = mesh_metrics(V, F)                               # flat: nothing hangs
    assert flat["overhang"]["area_over_45_mm2"] == 0.0
    assert flat["floor_area_mm2"] == pytest.approx(200.0)


def test_concat_and_stl_vertices(tmp_path):
    x, y = _grid(3, 4, 1.0)
    V1, F1 = slab_mesh(np.full((3, 4), 2.0), 0.0, x, y)
    V2, F2 = slab_mesh(np.full((3, 4), 3.0), 1.0, x + 10, y)
    V, F = concat([(V1, F1), (V2, F2)])
    assert V.shape[0] == V1.shape[0] + V2.shape[0] and F.max() == V.shape[0] - 1
    assert mesh_volume(V, F) == pytest.approx(mesh_volume(V1, F1) + mesh_volume(V2, F2))
    p = write_stl(V, F, tmp_path / "two.stl")
    P = stl_vertices(p)
    Vw, _ = read_stl(p)
    assert P.shape == Vw.shape
    assert np.allclose(np.sort(P, axis=0), np.sort(Vw, axis=0), atol=1e-3)
