"""Visual tests: tiny distinctive geometry rendered to PNG, a numeric
assertion, and a headless Claude judgement of the picture.

Offline by default (pyproject addopts "-m 'not visual'"); run them with
    uv run pytest -m visual
Model via LOOPVIZ_VISUAL_MODEL (default sonnet). Every test makes one judge
call. Renders and judge logs land in runs/visual/<test>/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
from visual_judge import RUNS, judge, judge_passes

from loopviz.print3d.relief import check_mesh, heightfield_mesh, mesh_volume, read_stl
from loopviz.print3d.reliefviz import mesh_views

pytestmark = pytest.mark.visual

FRONT = [("high", 60, -90), ("front-left", 35, -60)]


def run_dir(name: str) -> Path:
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- harness tests on the old stepped mesh (no plate modules needed) -----------

def letter_L() -> np.ndarray:
    H = np.full((5, 5), 2.0)
    H[1:4, 1] = 4.0          # stem: column 1, rows 1..3 (y up)
    H[1, 1:4] = 4.0          # foot: row 1, columns 1..3 (to the right)
    return H


def test_raised_L_is_not_mirrored():
    out = run_dir("raised_L")
    H = letter_L()
    V, F = heightfield_mesh(H, 2.0)
    assert check_mesh(V, F, H, 2.0)["volume_ok"]
    png = mesh_views(V, F, out / "L.png", title="raised L", views=FRONT)
    judge_passes([png], "A raised block letter L sitting inside a flat margin on a slab "
                 "(the L does not touch the slab edges; that is expected). Within the L, "
                 "the vertical stem is its LEFT part and the short foot at its BOTTOM "
                 "extends to the RIGHT of the stem (x increases to the right, y increases "
                 "upward in the 'high' view). Fail if the foot extends to the left of the "
                 "stem (mirrored), if the foot is at the top, or if the raised shape is "
                 "not an L.", test_name="raised_L")


def test_judge_rejects_a_wrong_expectation():
    """The pipeline must be able to say no: the same L judged as a T."""
    out = run_dir("raised_L_wrong")
    V, F = heightfield_mesh(letter_L(), 2.0)
    png = mesh_views(V, F, out / "L.png", title="raised L", views=FRONT)
    v = judge([png], "A raised block letter T on a flat slab: a horizontal bar "
              "across the TOP with a vertical stem hanging down from its centre. "
              "Fail if the raised shape is any other letter.", test_name="raised_L_wrong")
    print(f"[raised_L_wrong] observed: {v['observed']} | reason: {v['reason']}")
    assert v["pass"] is False, f"judge accepted an L as a T: {v['observed']} | {v['reason']}"


def test_staircase_rises_left_to_right():
    out = run_dir("staircase")
    H = np.tile(np.array([2.0, 3.0, 4.0, 5.0, 6.0]), (3, 1))
    V, F = heightfield_mesh(H, 2.0)
    assert check_mesh(V, F, H, 2.0)["volume_ok"]
    assert np.all(np.diff(H[0]) > 0)
    png = mesh_views(V, F, out / "stairs.png", title="staircase",
                     views=[("front", 20, -90), ("front-left", 35, -60)])
    judge_passes([png], "A staircase of five equal steps that rises from left to "
                 "right: the lowest step is at the left (x = 0) and the highest at "
                 "the right. Fail if it descends to the right or is not a staircase.",
                 test_name="staircase")


# -- plate tests (loopviz.print3d.plate / slab from the core agent) ----------------------

@pytest.fixture(scope="module")
def demo():
    """plate.demo_cases(out) -> {case: {name: Path}}, built once per module
    (about 10 s): F, F_cut, spike_nearest, spike_cubic, ridge_strip, with PNGs."""
    from loopviz.print3d import plate

    return plate.demo_cases(RUNS / "plate_demo", render=True)


@pytest.fixture(scope="module")
def slab():
    from loopviz.print3d import slab

    return slab


def _plate_frame_strip(case: dict, label: str, slab):
    """strips/<label>.stl read back and returned to the plate frame."""
    layout = json.loads(Path(case["layout.json"]).read_text())
    r = next(r for r in layout["strips"] if r["label"] == label)
    V, F = read_stl(Path(case["strips"]) / f"{label}.stl")
    return slab.transform(V, np.linalg.inv(np.asarray(r["transform"]))), F, r, layout


def test_F_plate_front_is_correct_and_reassembles(demo, slab):
    from loopviz.print3d.plate import verify_build

    case = demo["F_cut"]
    # numeric: every bed strip brought back by its inverse transform equals plate_cut.stl
    rep = verify_build(Path(case["plate.stl"]).parent)
    assert rep["max_dev_mm"] < 1e-3 and rep["strips"] == 6
    Vc, _ = read_stl(Path(case["plate_cut.stl"]))
    layout = json.loads(Path(case["layout.json"]).read_text())
    back = np.concatenate([_plate_frame_strip(case, r["label"], slab)[0]
                           for r in layout["strips"]])
    assert cKDTree(Vc).query(back)[0].max() < 1e-3     # STL is float32
    assert cKDTree(back).query(Vc)[0].max() < 1e-3
    judge_passes([demo["F"]["front_png"], case["reassembled_png"]],
                 "Both images show the same relief: a raised block letter F on a flat "
                 "rectangular slab, correctly oriented and NOT mirrored. In the 'front' "
                 "panel (x to the right, y up) the F has its vertical stem on the LEFT, "
                 "a long bar along the TOP extending to the RIGHT, and a shorter bar in "
                 "the middle also extending to the right, nothing at the bottom right. "
                 "Fail if either image shows the bars extending to the left, an upside "
                 "down F, or a different shape.", test_name="F_plate")


def test_bed_layout_six_upright_strips(demo):
    case = demo["F_cut"]
    layout = json.loads(Path(case["layout.json"]).read_text())
    bed, strips = layout["bed"], layout["strips"]
    assert len(strips) == 6 and [s["label"] for s in strips] == ["A1", "A2", "A3", "B1", "B2", "B3"]
    boxes = [np.asarray(s["bed_bbox"], float) for s in strips]
    for a, (lo, hi) in enumerate(boxes):
        assert np.all(lo >= -1e-6) and np.all(hi[:2] <= bed["size_mm"] + 1e-6)
        assert hi[2] <= bed["zmax_mm"] + 1e-6
        plo, phi = np.asarray(strips[a]["plate_bbox"], float)
        assert hi[2] - lo[2] == pytest.approx(phi[1] - plo[1], abs=1e-6)   # height = plate y extent
        for lo2, hi2 in boxes[a + 1:]:
            assert np.any(hi <= lo2 + 1e-9) or np.any(hi2 <= lo + 1e-9)  # no bbox overlap
    judge_passes([case["bed_png"]],
                 "The oblique panel shows exactly six separate thin rectangular slabs "
                 "standing upright on their long edges (tall and thin, not lying flat), "
                 "side by side in a row with visible gaps so none touch. The slabs carry "
                 "a relief on one face and every relief is on the SAME side (in the "
                 "top-down panel the wavy edge of every bar is on the same side; a slab "
                 "cut from plain background may be flat, that is fine). Fail if any slab "
                 "lies flat on the bed, if any two touch, if a relief faces the opposite "
                 "side from the others, or if the count is not six.", test_name="bed_layout")


def test_labels_read_from_behind(demo, slab):
    case = demo["F_cut"]
    for label in ("B2", "A3"):
        Vp, _, r, layout = _plate_frame_strip(case, label, slab)
        P = layout["params"]
        lo, hi = np.asarray(r["plate_bbox"], float)
        eng = Vp[(Vp[:, 2] > 1e-6) & (Vp[:, 2] < P["engrave_mm"] + 0.05)]
        assert len(eng) > 0 and np.allclose(eng[:, 2], P["engrave_mm"])  # recessed into the back
        assert Vp[:, 2].min() == pytest.approx(0.0, abs=1e-6)             # back face stays z = 0
        m = P["label_margin_mm"] - P["pitch_mm"] / P["subdiv"]                # snapped to the grid
        assert eng[:, 0].min() >= lo[0] + m and eng[:, 0].max() <= hi[0] - m
        assert eng[:, 1].min() >= lo[1] + m and eng[:, 1].max() <= hi[1] - m
        assert eng[:, 1].mean() > (lo[1] + hi[1]) / 2                         # upper half of the strip
    judge_passes([case["back_B2_png"], case["back_A3_png"]],
                 "Each image shows the back face of a strip with a label engraved into it "
                 "(the engraving shows as a slightly darker outline). Read the engraved "
                 "characters: image 1 must read exactly 'B2' and image 2 exactly 'A3', "
                 "reading normally left to right (not mirrored, not upside down). Next to "
                 "each label is a triangle; it must point UP (apex at the top). Fail on any "
                 "other character, mirrored text, or a triangle whose apex is not at the top.",
                 test_name="labels")


def test_nearest_is_stepped_and_cubic_is_smooth(demo):
    from scipy.ndimage import maximum_filter

    Vn, _ = read_stl(Path(demo["spike_nearest"]["plate.stl"]))
    Vc, _ = read_stl(Path(demo["spike_cubic"]["plate.stl"]))
    assert len(np.unique(np.round(Vn[:, 2], 6))) == 3       # back, slab top, spike top
    assert len(np.unique(np.round(Vc[:, 2], 6))) > 20       # a continuous bump
    # cubic: one dominant peak at the centre; the ringing lobes around it stay
    # small relative to the background level (secondary maxima under 5 % of
    # the relief above it, the dip under 15 % below it)
    Z = np.load(Path(demo["spike_cubic"]["plate.stl"]).parent / "surface_Z.npy")
    base, relief = float(np.median(Z)), float(Z.max() - Z.min())
    peaks = Z[(Z == maximum_filter(Z, size=3)) & (Z > Z.min() + 1e-6)]
    assert np.isclose(peaks.max(), Z.max()) and Z.max() > base + 0.8 * relief
    assert np.sort(peaks)[:-3].max() < base + 0.05 * relief   # the peak spans up to 3 nodes
    assert Z.min() > base - 0.15 * relief
    judge_passes([demo["spike_nearest"]["views_png"], demo["spike_cubic"]["views_png"]],
                 "Image 1 (nearest) must show a single square flat-topped column with "
                 "steep straight walls rising from a flat slab: a sharp step up and a "
                 "flat top, no rounding. Image 2 (cubic) must show one dominant smooth "
                 "bump (a rounded or pointed hill with continuously sloping sides) and no "
                 "terraces or steps; a faint ring-shaped dip and faint ripples around the "
                 "hill are expected interpolation ringing and are acceptable. Fail if "
                 "image 1 looks rounded, if image 2 shows flat terraces, a flat-topped "
                 "column, or a second hill of comparable height to the main one.",
                 test_name="smoothness")


def test_overhang_colouring(demo, slab):
    # numeric: a 60 deg wedge has one overhanging face of known area
    V, F = _wedge(60.0)
    assert mesh_volume(V, F) > 0
    m = slab.mesh_metrics(V, F)["overhang"]
    assert m["max_deg"] == pytest.approx(60.0, abs=1e-6)
    assert m["area_over_45_mm2"] == pytest.approx(100.0 * np.hypot(1.0, np.tan(np.radians(60.0))))
    case = demo["ridge_strip"]
    Vs, Fs = read_stl(Path(case["bed.stl"]))
    ms = slab.mesh_metrics(Vs, Fs)["overhang"]
    assert 45.0 < ms["max_deg"] < 90.0 and ms["area_over_45_mm2"] > 0
    # own render: end-on so the ridge profile and its colour bands are unambiguous
    from loopviz.print3d.plate import overhang_colors, preview_faces

    Fp = preview_faces(Vs, Fs)
    png = mesh_views(Vs, Fp, run_dir("overhang") / "ridge_overhang.png",
                     title="ridge strip on edge, red = overhang > 45 deg",
                     views=[("end-on profile", 5, -15), ("under side", -20, -50)],
                     face_colors=overhang_colors(Vs, Fp))
    judge_passes([png],
                 "A thin slab standing on edge; its front face bulges out in one horizontal "
                 "ridge whose crest is at mid height. Faces are coloured by overhang angle: "
                 "grey = facing up or vertical, green < 30 deg, yellow 30 to 45 deg, red "
                 "> 45 deg, all measured on DOWNWARD-facing faces. Expected: the flank above "
                 "the crest and the flat upper part of the slab are grey; below the crest "
                 "(the under side of the bulge) the bands run green, yellow, RED, yellow, "
                 "green going downward, so red is one band in the lower half of the bulge. "
                 "Pass if red appears only on the under side of the bulge. Fail if red "
                 "appears above the crest, on the top edge, on the end faces, or on the "
                 "flat upper part of the slab.", test_name="overhang")


def _wedge(deg: float):
    """Closed triangular prism (10 mm along x) whose sloped face leans
    outward by `deg` from vertical, so it is a downward-facing overhang."""
    t = np.tan(np.radians(deg))
    yz = np.array([[0.0, 0.0], [0.0, 10.0], [10.0 * t, 10.0]])      # (y, z)
    V = np.array([[x, y, z] for x in (0.0, 10.0) for y, z in yz])    # 0..2 at x=0, 3..5 at x=10
    F = np.array([[0, 1, 2], [3, 5, 4],                              # end caps
                  [0, 4, 1], [0, 3, 4],                              # y = 0 wall (normal -y)
                  [1, 5, 2], [1, 4, 5],                              # top z = 10
                  [2, 3, 0], [2, 5, 3]])                             # sloped underside
    return V, F
