"""The strip packer must reproduce the hand-over's worked result, respect
every constraint (body under the relief, zmax, bed slots) and give a
monotone Pareto frontier."""

import json
import math

import pytest

from loopviz.print3d.strippack import (
    Bed,
    Material,
    best_layout,
    best_per_t,
    enumerate_layouts,
    main,
    make_layout,
    pareto,
    recommend,
    strip_material,
)

BED = Bed()                    # 340 bed, margin 5 -> L 330, zmax 325, gap 8
# the hand-over has no body rule, and its t = 5 row (2 x 13) needs 13 rows
HANDOVER = {"relief_mm": 0.0, "body_min_mm": 0.0, "max_rows": 13}


def by_t(layouts):
    return {lay.thickness_mm: lay for lay in best_per_t(layouts)}


# -- hand-over reproduction ---------------------------------------------------

def test_slots_match_handover():
    assert [BED.slots(t) for t in (5, 8, 10, 15, 40)] == [26, 21, 18, 14, 7]


def test_handover_table_and_strip_size():
    best = by_t(enumerate_layouts(BED, **HANDOVER))
    assert [best[t].side_mm for t in (5.0, 8.0, 10.0)] == [520, 640, 660]
    lay = best[10.0]
    assert (lay.cols, lay.rows, lay.n_strips, lay.slots) == (2, 9, 18, 18)
    assert lay.strip_len_mm == 330
    # the hand-over quotes 330 x 80 strips; 80 is h_max, the square plate
    # only needs 660 / 9
    assert lay.strip_height_mm == pytest.approx(660 / 9)
    assert lay.binding == "len"


def test_global_optimum_3x3_at_29_5():
    lays = enumerate_layouts(BED, **HANDOVER)
    top = best_layout(lays)
    assert (top.cols, top.rows, top.thickness_mm, top.side_mm) == (3, 3, 29.5, 708)
    # the same on the hand-over's 0.1 mm grid
    fine = best_layout(enumerate_layouts(BED, t_step=0.1, **HANDOVER))
    assert (fine.cols, fine.rows, fine.thickness_mm, fine.side_mm) == (3, 3, 29.5, 708)


def test_plateau_at_2L_except_the_3x3_window():
    best = by_t(enumerate_layouts(BED, **HANDOVER))
    for t, lay in best.items():
        if t >= 10:
            if 27.5 < t <= 29.5:
                assert lay.side_mm > 660 and (lay.cols, lay.rows) == (3, 3)
            else:
                assert lay.side_mm == 660 and lay.binding == "len"


def test_600_target_fits_2x8_at_t10():
    lay = make_layout(BED, 2, 8, 10.0, 8.0, 0.0, Material(), side_mm=600.0)
    assert lay is not None
    assert (lay.strip_len_mm, lay.strip_height_mm) == (300, 75)


# -- constraints ----------------------------------------------------------------

def test_thickness_holds_relief_plus_body():
    lays = enumerate_layouts(BED, relief_mm=4.0, body_min_mm=3.0)
    assert min(lay.thickness_mm for lay in lays) == 7.0
    assert all(lay.thickness_mm >= lay.relief_mm + 3.0 for lay in lays)
    lays = enumerate_layouts(BED, relief_mm=6.0, body_min_mm=5.0, t_min=5, t_max=12)
    assert {lay.thickness_mm for lay in lays} == {11.0, 11.5, 12.0}


def test_zmax_binds_for_thick_strips():
    lays = enumerate_layouts(BED, t_min=30, t_max=60, t_step=1.0)
    assert all(lay.strip_height_mm <= BED.zmax_mm + 1e-9 for lay in lays)
    tall = [lay for lay in lays if lay.binding == "zmax"]
    assert tall and all(lay.thickness_mm * 8 >= BED.zmax_mm for lay in tall)
    assert all(lay.binding != "zmax" for lay in lays if lay.thickness_mm * 8 < BED.zmax_mm)
    assert max(lay.strip_height_mm for lay in tall) == BED.zmax_mm


def test_strips_fit_the_bed_without_overlap():
    for lay in enumerate_layouts(BED, max_per_row=2):
        assert lay.n_strips <= lay.slots * lay.per_row
        rows = math.ceil(lay.n_strips / lay.per_row)
        assert rows * lay.thickness_mm + (rows - 1) * BED.gap_mm <= BED.usable_mm + 1e-9
        assert (lay.per_row * lay.strip_len_mm + (lay.per_row - 1) * BED.gap_mm
                <= BED.usable_mm + 1e-9)
        assert lay.strip_height_mm <= min(BED.zmax_mm, 8 * lay.thickness_mm) + 1e-9
        assert lay.side_mm == pytest.approx(lay.cols * lay.strip_len_mm)
        assert lay.side_mm == pytest.approx(lay.rows * lay.strip_height_mm)


def test_two_per_row_beats_1d_on_this_bed():
    one = best_layout(enumerate_layouts(BED))
    two = best_layout(enumerate_layouts(BED, max_per_row=2))
    assert one.side_mm == 708
    assert (two.cols, two.rows, two.per_row, two.thickness_mm) == (5, 4, 2, 25.5)
    assert two.side_mm == pytest.approx(805)
    assert two.strip_len_mm == pytest.approx(161)


# -- material, time, slenderness ----------------------------------------------

def test_material_model_is_a_shell_plus_infill_core():
    mat = Material()
    m = strip_material(330, 660 / 9, 10, 4, mat)
    w = 3 * 0.45
    face = 330 * 660 / 9
    shell = w * (1.4 * face + face + 2 * (330 + 660 / 9) * 10)
    solid = face * (10 - 2)
    infill = (solid - shell) * 0.15
    assert m["volume_mm3"] == pytest.approx(shell + infill)
    assert m["mass_kg"] == pytest.approx((shell + infill) * 1.24e-6)
    assert m["hours"] == pytest.approx((shell + infill) / 6 / 3600)
    # a thin strip is all shell: no infill, never more than the solid volume
    thin = strip_material(100, 50, 3, 1, mat)
    assert thin["infill_mm3"] == 0 and thin["volume_mm3"] == pytest.approx(100 * 50 * 2.5)
    # the 660 plate at t = 10: 18 strips, close to the sliced 2.1 kg (relief 4 here)
    lay = by_t(enumerate_layouts(BED))[10.0]
    assert 2.0 < lay.mass_kg < 2.6
    assert lay.mass_kg == pytest.approx(18 * m["mass_kg"])


def test_slenderness_and_frequency():
    lay = make_layout(BED, 2, 9, 10.0, 8.0, 0.0, Material())
    assert lay.slenderness == pytest.approx(660 / 9 / 10)
    assert lay.f1_hz == pytest.approx(270 * 0.01 / (0.66 / 9) ** 2)
    # 270 t / h^2 is the PLA cantilever constant (1.875^2 / 2 pi) sqrt(E / 12 rho)
    c = 1.875 ** 2 / (2 * math.pi) * math.sqrt(3.5e9 / (12 * 1240))
    assert c == pytest.approx(270, rel=0.01)
    # at k = 8 a strip only falls under 100 Hz when t > 2.7 / 64 m = 42 mm
    assert all(lay.f1_hz > 100 for lay in enumerate_layouts(BED))
    assert lay.seams == 9 and lay.seam_len_mm == 9 * 660


# -- frontier and recommendation ----------------------------------------------

def test_pareto_is_monotone_and_dominates():
    lays = enumerate_layouts(BED)
    cands = best_per_t(lays)
    front = pareto(cands)
    sides = [lay.side_mm for lay in front]
    kgs = [lay.mass_kg for lay in front]
    assert sides == sorted(sides) and len(set(sides)) == len(sides)
    assert kgs == sorted(kgs) and len(set(kgs)) == len(kgs)
    for c in cands:
        assert any(f.side_mm >= c.side_mm and f.mass_kg <= c.mass_kg for f in front)
    assert front[-1].side_mm == 708
    assert pareto(lays)[-1].side_mm == 708


def test_recommendation_jumps_over_kinks():
    front = pareto(best_per_t(enumerate_layouts(BED)))
    rec = recommend(front, 5.0)
    assert (rec.side_mm, rec.cols, rec.rows, rec.thickness_mm) == (660, 2, 10, 8.5)
    assert recommend(front, 1e9) is front[-1]
    assert recommend(front, 0.0) is front[0]


# -- CLI ------------------------------------------------------------------------

def test_cli_plan_json(tmp_path, capsys):
    out = tmp_path / "plan.json"
    main(["plan", "--side", "630", "--json", str(out)])
    text = capsys.readouterr().out
    assert "29.5 3x3" in text and "<- max S" in text and "recommended: t = 8.5" in text
    res = json.loads(out.read_text())
    assert res["best"]["side_mm"] == 708
    assert res["t_infeasible"] == [5.0, 5.5, 6.0, 6.5]
    assert res["side"]["side_mm"] == 630
    assert all(d["side_mm"] == 630 for d in res["side"]["layouts"])
    assert res["side"]["lightest_per_grid"][0]["cols"] == 2


def test_cli_env_override(monkeypatch, capsys):
    monkeypatch.setenv("LOOPVIZ_BED_MM", "260")
    monkeypatch.setenv("LOOPVIZ_ZMAX_MM", "250")
    main(["plan"])
    assert "bed 260 mm" in capsys.readouterr().out
