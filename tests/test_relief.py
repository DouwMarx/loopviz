"""The relief mesh must be a closed, outward-oriented solid whose volume is
exactly the sum of the columns, for every height configuration, and the
quantization must round-trip through the STL."""

import numpy as np
import pytest

from loopviz.relief import (
    Printer,
    check_mesh,
    heightfield_mesh,
    heights,
    printer,
    quantized_matrix,
    read_stl,
    relief_plan,
    signed_levels,
    write_stl,
)

CASES = {
    "2x2_all_different": np.array([[1.0, 2.0], [3.0, 4.0]]),
    "2x2_flat": np.full((2, 2), 2.5),
    "3x3_checker_diagonal_touch": np.array([[3.0, 1.0, 3.0],
                                            [1.0, 3.0, 1.0],
                                            [3.0, 1.0, 3.0]]),
    "3x3_four_heights_at_a_corner": np.array([[1.0, 2.0, 1.0],
                                              [4.0, 3.0, 1.0],
                                              [1.0, 1.0, 1.0]]),
    "1x5_strip": np.array([[1.0, 3.0, 2.0, 3.0, 1.0]]),
    "5x1_strip": np.array([[1.0], [3.0], [2.0], [3.0], [1.0]]),
    "1x1": np.array([[2.0]]),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_mesh_closed_oriented_exact_volume(name):
    H = CASES[name]
    V, F = heightfield_mesh(H, 1.5)
    c = check_mesh(V, F, H, 1.5)
    assert c["closed"], c
    assert c["oriented"], c
    assert c["volume_ok"], c
    assert c["volume"] > 0


def test_random_heights_many_levels():
    rng = np.random.default_rng(0)
    H = 2.0 + 0.1 * rng.integers(0, 40, (23, 17))
    V, F = heightfield_mesh(H, 1.2)
    c = check_mesh(V, F, H, 1.2)
    assert c["watertight"] and c["volume_ok"], c
    # bounding box is exactly the footprint and the tallest column
    assert np.allclose(V.min(axis=0), 0.0)
    assert np.allclose(V.max(axis=0), [17 * 1.2, 23 * 1.2, H.max()])


def test_trimesh_agrees_on_volume_and_winding():
    trimesh = pytest.importorskip("trimesh")
    rng = np.random.default_rng(1)
    H = 2.0 + 0.1 * rng.integers(0, 40, (12, 12))
    V, F = heightfield_mesh(H, 1.0)
    tm = trimesh.Trimesh(V, F, process=False)
    assert tm.is_winding_consistent
    assert abs(tm.volume - float(H.sum())) < 1e-6
    # the only defects trimesh may see are voxel edges (4 faces), never holes
    e = tm.edges_sorted
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert set(np.unique(cnt)) <= {2, 4}


def test_stl_roundtrip(tmp_path):
    H = CASES["3x3_four_heights_at_a_corner"]
    V, F = heightfield_mesh(H, 2.0)
    path = write_stl(V, F, tmp_path / "t.stl")
    assert path.stat().st_size == 84 + 50 * F.shape[0]
    V2, F2 = read_stl(path)
    c = check_mesh(V2, F2, H, 2.0)
    assert c["watertight"] and c["volume_ok"], c


def test_signed_levels_symmetric_and_clipped():
    A = np.array([[-10.0, -1.0, 0.0, 1.0, 10.0]])
    q = signed_levels(A, levels=5, clip_pct=100.0)
    assert q.tolist() == [[0, 2, 2, 2, 4]]          # 10 spans the range
    q = signed_levels(A, levels=5, clip_pct=50.0)   # clip at |A| = 1
    assert q.tolist() == [[0, 0, 2, 4, 4]]
    Aq = quantized_matrix(A, levels=5, clip_pct=50.0)
    assert np.allclose(Aq, [[-1, -1, 0, 1, 1]])


def test_plan_geometry_and_levels():
    pr = Printer("t", bed_mm=305.0, layer_mm=0.1, step_layers=2, margin_mm=2.5,
                 max_aspect=2.5, max_relief_mm=6.0)
    rp = relief_plan(T=13.5, pr=pr, pitch_mm=1.5, rho=0.95)
    assert rp.n == 200                       # floor(300 / 1.5)
    assert rp.side_mm == 300.0
    assert rp.relief_mm == pytest.approx(3.6)   # 2.5 * 1.5 = 3.75 -> 18 steps of 0.2
    assert rp.levels == 19
    assert rp.plan.N == 190
    assert rp.plan.f == pytest.approx(190 * 200 / 13.5)
    H = heights(np.linspace(-1, 1, 200 * 200).reshape(200, 200), rp)
    assert H.min() == pytest.approx(rp.base_mm)
    assert H.max() == pytest.approx(rp.base_mm + rp.relief_mm)
    steps = np.unique(np.round((H - rp.base_mm) / pr.layer_mm, 6))
    assert np.allclose(steps, np.round(steps))    # tops on layer boundaries
    assert np.allclose(steps % 2, 0)             # and two layers apart


def test_max_protrusion_is_local_not_range():
    from loopviz.relief import max_protrusion

    ramp = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert max_protrusion(ramp) == 1.0          # 5 mm range, 1 mm steps
    spike = np.array([[1.0, 1.0, 1.0], [1.0, 6.0, 1.0], [1.0, 1.0, 1.0]])
    assert max_protrusion(spike) == 5.0
    assert max_protrusion(np.full((3, 3), 2.0)) == 0.0


def test_printer_env_override(monkeypatch):
    monkeypatch.setenv("LOOPVIZ_BED_MM", "250")
    monkeypatch.setenv("LOOPVIZ_LAYER_MM", "0.2")
    monkeypatch.setenv("LOOPVIZ_STEP_LAYERS", "1")
    pr = printer("fdm04", margin_mm=5.0)
    assert pr.bed_mm == 250.0 and pr.layer_mm == 0.2 and pr.margin_mm == 5.0
    assert pr.side_mm == 240.0 and pr.step_layers == 1
    # an explicit flag beats the environment instead of crashing
    assert printer("fdm04", bed_mm=40.0).bed_mm == 40.0


def test_plan_floor_and_tiles_and_validation():
    pr = Printer("t", bed_mm=305.0, margin_mm=2.5)
    assert relief_plan(13.5, pr, 0.8).n == 375          # 300 / 0.8 exactly
    assert relief_plan(13.5, pr, 1.2, side_mm=14.4).n == 12
    rp = relief_plan(13.5, pr, 1.9, tiles=2)
    assert rp.n == 2 * 157 and rp.n * 1.9 / 2 <= pr.side_mm   # each tile fits
    with pytest.raises(ValueError):
        relief_plan(13.5, pr, 2.0, relief_mm=0.0)
    with pytest.raises(ValueError):
        relief_plan(13.5, pr, 2.0, base_mm=0.0)


def test_cli_sweep_probe_and_step_layers(tmp_path, capsys):
    import soundfile as sf

    from loopviz.cli import main

    sr = 4000
    t = np.arange(int(2.0 * sr)) / sr
    x = np.sin(2 * np.pi * 110 * t) + 0.2 * np.random.default_rng(0).standard_normal(t.size)
    wav = tmp_path / "s.wav"
    sf.write(str(wav), x, sr)
    base = ["relief", "sweep", "--audio", str(wav), "--start", "0.2", "--end", "1.7",
            "--bed", "40", "--margin", "2", "--pitches", "2.0,3.0", "--rho", "0.7"]
    main(base + ["--probe", "--json", str(tmp_path / "sw.json")])
    out = capsys.readouterr().out
    assert "f_max" in out and (tmp_path / "sw.json").exists()
    main(base + ["--step-layers", "1"])
    one = capsys.readouterr().out
    main(base + ["--step-layers", "2"])
    two = capsys.readouterr().out
    lv = lambda s: int(s.splitlines()[-2].split("|")[7])   # levels column, 3.0 mm row
    assert lv(one) == 2 * (lv(two) - 1) + 1                  # 61 vs 31


def test_cli_build_end_to_end(tmp_path):
    """The documented user path: a wav in, STL + previews + json out."""
    import json

    import soundfile as sf

    from loopviz.cli import main

    sr = 4000
    t = np.arange(int(2.0 * sr)) / sr
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 110 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    x += 0.2 * rng.standard_normal(x.size)
    wav = tmp_path / "s.wav"
    sf.write(str(wav), x, sr)
    out = tmp_path / "out"
    main(["relief", "build", "--audio", str(wav), "--start", "0.2", "--end", "1.7",
          "--pitch", "2.0", "--bed", "40", "--margin", "2", "--rho", "0.7",
          "--tiles", "2", "--out", str(out)])
    meta = json.loads((out / "relief.json").read_text())
    assert meta["n"] == 36 and meta["mesh"]["watertight"]     # 2 x floor(36 / 2)
    assert len(list((out / "tiles").glob("tile_r*_c*.stl"))) == 4
    assert sum(meta["level_histogram"]) == 36 * 36
    assert meta["exact_drift_per_pass"] < 1e-6
    for f in ("relief.stl", "hillshade.png", "heightmap16.png", "mesh_views.png",
              "loop.wav", "A.npy", "H_mm.npy"):
        assert (out / f).exists(), f
    V, F = read_stl(out / "relief.stl")
    H = np.load(out / "H_mm.npy")
    assert check_mesh(V, F, H, 2.0)["volume_ok"]


def test_tiles_partition_the_height_map(tmp_path):
    from loopviz.relief_cli import write_tiles

    rng = np.random.default_rng(2)
    H = 2.0 + 0.1 * rng.integers(0, 20, (10, 8))
    paths = write_tiles(H, 1.0, 2, tmp_path)
    assert len(paths) == 4
    with pytest.raises(ValueError):
        write_tiles(H[:9], 1.0, 2, tmp_path)
    vol = sum(check_mesh(*read_stl(p))["volume"] for p in paths)
    assert abs(vol - float(H.sum())) < 1e-6      # tiles cover every cell once
