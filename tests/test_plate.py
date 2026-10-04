"""The plate pipeline end to end on small matrices: cuts share seam
nodes, strips are watertight, the bed placement is valid, the files
reassemble to the plate, the demo and CLI run."""

import json
from itertools import pairwise

import numpy as np
import pytest

from loopviz import cli
from loopviz.print3d.plate import (
    Bed,
    PlateParams,
    build_plate,
    col_letter,
    cut_indices,
    cut_plate,
    demo_cases,
    edge_transform,
    f_matrix,
    strip_label,
    verify_build,
)
from loopviz.print3d.relief import check_mesh, read_stl
from loopviz.print3d.slab import transform
from loopviz.print3d.surface import surface


def test_labels_and_cuts():
    assert [col_letter(c) for c in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]
    assert strip_label(1, 0) == "B1"
    assert cut_indices(10, 3) == [(0, 3), (3, 6), (6, 9)]
    assert cut_indices(991, 2) == [(0, 495), (495, 990)]
    with pytest.raises(ValueError):
        cut_indices(3, 4)


def test_edge_transform_places_strip_k():
    bed = Bed(size_mm=340, zmax_mm=325, margin_mm=5, gap_mm=8)
    T = edge_transform(2, x0=100.0, y0=50.0, thickness_mm=10.0, bed=bed)
    # plate strip corners (x0, y0, 0) and (x0 + l, y0 + h, t)
    lo = transform(np.array([[100.0, 50.0, 0.0]]), T)[0]
    hi = transform(np.array([[300.0, 110.0, 10.0]]), T)[0]
    yk = 5 + 10 + 2 * 18
    assert np.allclose(lo, [5, yk, 0])                 # back face at y_k, on the bed
    assert np.allclose(hi, [205, yk - 10, 60])         # relief face at y_k - t
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_cut_strips_share_seams_and_are_watertight():
    trimesh = pytest.importorskip("trimesh")
    rng = np.random.default_rng(0)
    A = rng.standard_normal((11, 14))
    surf = surface(A, 2.0, 8.0, 3.0, subdiv=2)
    P = PlateParams(2.0, 2, 3, 8.0, 3.0, subdiv=2, labels=False)
    strips = cut_plate(surf, P, Bed())
    assert [s.label for s in strips] == ["A1", "A2", "A3", "B1", "B2", "B3"]
    for s in strips:
        m = trimesh.Trimesh(s.V, s.F, process=False)
        assert m.is_watertight and m.is_winding_consistent and m.volume > 0
        assert check_mesh(s.V, s.F)["watertight"]
        lo, hi = s.V.min(axis=0), s.V.max(axis=0)
        assert lo[2] == 0.0 and hi[2] <= 8.0 + 1e-9
        assert (s.V[:, 2] >= 0).all() and set(np.unique(s.tag)) == {0, 1, 2}
    a1, b1 = strips[0], strips[3]
    seam_a = a1.V[(np.abs(a1.V[:, 0] - surf.x[a1.j1]) < 1e-12) & (a1.V[:, 2] > 0)]
    seam_b = b1.V[(np.abs(b1.V[:, 0] - surf.x[b1.j0]) < 1e-12) & (b1.V[:, 2] > 0)]
    assert seam_a.shape == seam_b.shape
    assert np.allclose(np.sort(seam_a, axis=0), np.sort(seam_b, axis=0))
    # on the bed: height equals the plate y extent, nothing overlaps
    boxes = [(s.bed_V.min(axis=0), s.bed_V.max(axis=0)) for s in strips]
    for s, (lo, hi) in zip(strips, boxes):
        assert hi[2] - lo[2] == pytest.approx(surf.y[s.i1] - surf.y[s.i0])
        assert hi[1] - lo[1] <= 8.0 + 1e-9               # back face to this strip's peak
        assert lo[0] == pytest.approx(5.0) and lo[1] >= 5.0 - 1e-9
    assert max(hi[1] - lo[1] for lo, hi in boxes) == pytest.approx(8.0)   # plate peak at t
    ys = sorted((lo[1], hi[1]) for lo, hi in boxes)
    for (_, a_hi), (b_lo, _) in pairwise(ys):
        assert 8.0 - 1e-9 <= b_lo - a_hi <= 8.0 + 8.0     # at least one gap between neighbours


def test_build_plate_writes_everything_and_verifies(tmp_path):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((16, 20))
    P = PlateParams(2.0, 2, 2, 8.0, 3.0, labels=True, label_mm=5.0, label_margin_mm=2.0)
    m = build_plate(A, P, Bed(), tmp_path, render=False, verbose=False)
    for f in ("plate.stl", "bed.stl", "layout.json", "metrics.json",
              "assembly.md", "A.npy", "surface_Z.npy", "strips/A1.stl", "strips/B2.stl"):
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "plate_cut.stl").exists()        # opt-in (write_cut)
    assert m["verify"]["max_dev_mm"] <= 1e-3
    assert m["strips"] == 4 and m["relief_mm"] <= 3.0 and "frac_facet_over_limit" in m["slopes"]
    layout = json.loads((tmp_path / "layout.json").read_text())
    assert layout["params"]["relief_mm"] == m["relief_mm"]
    assert layout["params"]["relief_requested_mm"] == 3.0
    est = m["material_estimate"]
    assert est["volume_mm3"] < m["volume_mm3"] and est["mass_kg"] > 0
    rc = m["relief_choice"]
    assert rc["requested_mm"] == 3.0 and rc["cap_mm"] == 5.0 and rc["scale"] == 1.0
    assert m["printed_range_mm"][1] == pytest.approx(8.0)          # highest node at t
    assert m["printed_range_mm"][2] == pytest.approx(rc["printed_range_mm"])
    assert m["printed_range_mm"][2] <= 5.0
    assert set(m["overhang_bed"]) >= {"edges_deg", "area_mm2", "area_over_45_mm2", "max_deg"}
    layout = json.loads((tmp_path / "layout.json").read_text())
    assert [s["label"] for s in layout["strips"]] == ["A1", "A2", "B1", "B2"]
    md = (tmp_path / "assembly.md").read_text()
    assert "A2    B2" in md and "A1    B1" in md and "Elephant" in md
    # plate.stl is the uncut slab with a flat back: strips carry engraving, so
    # their volume is a little less than the plate's
    V, F = read_stl(tmp_path / "plate.stl")
    plate_vol = check_mesh(V, F)["volume"]
    assert 0 < plate_vol - m["volume_mm3"] < 0.02 * plate_vol
    ver = verify_build(tmp_path)
    assert ver["strips"] == 4 and ver["plate_vertices"] == 49 * 61    # relief nodes only
    # corrupt one strip file and the check must fail
    (tmp_path / "strips" / "A1.stl").write_bytes((tmp_path / "strips" / "B2.stl").read_bytes())
    with pytest.raises(RuntimeError):
        verify_build(tmp_path)
    m2 = build_plate(A, P, Bed(), tmp_path / "cut", render=False, verbose=False, write_cut=True)
    assert (tmp_path / "cut" / "plate_cut.stl").exists() and m2["files_mb"]["strips/"] > 0


def test_flat_orientation_and_bad_params(tmp_path):
    A = np.random.default_rng(4).standard_normal((10, 10))
    P = PlateParams(2.0, 1, 2, 8.0, 3.0, orientation="flat", labels=False)
    m = build_plate(A, P, Bed(), tmp_path, render=False, verbose=False)
    assert not (tmp_path / "bed.stl").exists() and m["verify"]["strips"] == 2
    with pytest.raises(ValueError):
        PlateParams(2.0, 1, 1, 5.0, 3.0, fit_range=False)  # explicit relief, body under 3 mm
    big = PlateParams(2.0, 1, 1, 5.0, 3.0, labels=False)    # auto: scaled to fit instead
    mb = build_plate(A, big, Bed(), tmp_path / "big", render=False, verbose=False)
    assert mb["relief_choice"]["scaled_to_fit"] and mb["relief_choice"]["scale"] < 1.0
    assert mb["printed_range_mm"][2] == pytest.approx(2.0) and mb["body_min_actual_mm"] == pytest.approx(3.0)
    assert mb["printed_range_mm"][1] == pytest.approx(5.0)
    with pytest.raises(ValueError):
        PlateParams(2.0, 1, 1, 8.0, 3.0, interp="nope")
    with pytest.raises(RuntimeError):                   # 20 mm / 4 mm = 5 > k = 2
        build_plate(A, PlateParams(2.0, 1, 1, 4.0, 1.0, labels=False, k_slender=2.0),
                    Bed(), tmp_path / "k", render=False, verbose=False)
    with pytest.raises(RuntimeError):                   # brim wider than the margin
        build_plate(A, PlateParams(2.0, 1, 1, 8.0, 3.0, labels=False),
                    Bed(brim_mm=6.0), tmp_path / "brim", render=False, verbose=False)
    with pytest.raises(ValueError):                     # label does not fit the strip
        build_plate(A, PlateParams(2.0, 1, 4, 8.0, 3.0, label_mm=6.0), Bed(),
                    tmp_path / "lab", render=False, verbose=False)


def test_f_matrix_is_asymmetric():
    A = f_matrix()
    assert A.shape == (18, 24)
    assert not np.array_equal(A, A[:, ::-1]) and not np.array_equal(A, A[::-1])


def test_demo_cases_and_cli(tmp_path):
    cases = demo_cases(tmp_path / "demo", render=False)
    assert set(cases) == {"F", "F_cut", "spike_nearest", "spike_cubic", "ridge_strip"}
    assert cases["F_cut"]["bed.stl"].exists() and (tmp_path / "demo" / "cases.json").exists()
    for name, files in cases.items():
        assert verify_build(tmp_path / "demo" / name)["max_dev_mm"] <= 1e-3
        assert (files["plate_strips"] / "A1.stl").exists() and (files["strips"] / "A1.stl").exists()
    assert (cases["F_cut"]["plate_strips"] / "B3.stl").exists()
    ridge = json.loads(cases["ridge_strip"]["metrics.json"].read_text())
    assert ridge["overhang_relief_face"]["max_deg"] > 45.0
    spike = json.loads(cases["spike_nearest"]["metrics.json"].read_text())
    assert spike["overhang_relief_face"]["max_deg"] > 45.0
    cli.main(["plate", "demo", "--out", str(tmp_path / "cli"), "--no-render"])
    assert (tmp_path / "cli" / "ridge_strip" / "bed.stl").exists()
