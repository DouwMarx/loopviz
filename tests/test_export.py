"""`loopviz plate export` on tiny builds: the package file list, the spec
sheet keys, the audio recipe. Offline, no slicer."""

import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from loopviz.cli import main
from loopviz.loopspec import LoopSpec
from loopviz.print3d import export
from loopviz.print3d.plate import demo_cases
from loopviz.songmatrix import Plan, build, materialize, windows

SHEET_KEYS = {"build", "loop", "operator", "plate", "relief", "labels", "bed", "overhang", "mass",
              "slicer", "files_mb", "audio", "provenance"}
RELIEF_KEYS = {"mode", "rule", "fit_facet_mm", "fit_y_mm", "max_overhang_deg", "area_quantile",
               "requested_mm", "chosen_mm", "scale", "printed_range_mm", "body_min_mm", "cap_mm",
               "interp", "subdiv", "clip_pct", "mapping", "sigma_cells"}


def test_play_is_the_resampled_song_for_the_exact_operator():
    sr, T = 2000, 1.5
    t = np.arange(int(T * sr)) / sr
    sig = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 123 * t)
    pl = Plan(n=20, N=14, f=14 * 20 / T, T=T)
    raw = windows(sig, pl)
    norms = np.linalg.norm(raw, axis=0)
    op, W = build(sig, pl)
    x, info = export.play(materialize(op), W, norms, passes=3)
    assert x.size == 3 * 14 * 20 and info["diverged_at_window"] is None
    one = raw.T.ravel()                              # the dithered resampled song
    assert np.abs(x[:one.size] - one).max() < 1e-4   # pass 1 is the song itself
    assert np.abs(x[one.size:2 * one.size] - one).max() < 1e-3
    Aq, levels = export.printed_matrix(materialize(op), 5.0, 99.5)
    assert levels == 101 and Aq.shape == (20, 20)
    assert export.step_error(Aq, W)["max"] > export.step_error(materialize(op), W)["max"]
    # an unstable operator is cut to silence instead of overflowing
    y, info = export.play(2.0 * np.eye(20), W, norms, passes=3)
    assert info["diverged_at_window"] == 7 and np.all(y[7 * 20:] == 0) and np.isfinite(y).all()


def test_export_demo_build_no_audio(tmp_path):
    cases = demo_cases(tmp_path / "demo", render=False)
    build_dir = cases["F_cut"]["layout.json"].parent
    out = tmp_path / "export" / "F_cut"
    r = export.export_build(build_dir, out, design_study=tmp_path / "none", zip_it=True,
                            verbose=False)
    files = set(r["files"])
    assert {"README.md", "spec_sheet.md", "spec_sheet.json", "stl/plate_full.stl",
            "stl/bed_print_as_is.stl", "analysis/metrics.json", "analysis/layout.json",
            "analysis/assembly.md"} <= files
    assert {f"stl/strips/{lab}.stl" for lab in ("A1", "A2", "A3", "B1", "B2", "B3")} <= files
    assert not any(f.startswith("audio/") for f in files)
    assert not any(f.startswith("renders/") for f in files)      # render=False: no previews
    sheet = json.loads((out / "spec_sheet.json").read_text())
    assert set(sheet) == SHEET_KEYS and set(sheet["relief"]) == RELIEF_KEYS
    assert sheet["loop"] is None and sheet["operator"] is None and sheet["audio"] is None
    assert sheet["plate"]["cols"] == 2 and sheet["plate"]["rows"] == 3 and sheet["plate"]["strips"] == 6
    assert sheet["provenance"]["loopviz_version"] and "built" in sheet["provenance"]
    md = (out / "spec_sheet.md").read_text()
    assert "## Relief" in md and "not sliced" in md
    assert "demo plate" in (out / "README.md").read_text()
    z = tmp_path / "export" / "F_cut.zip"
    assert r["zip"] == str(z) and z.exists()
    with zipfile.ZipFile(z) as zf:
        assert "F_cut/stl/bed_print_as_is.stl" in zf.namelist()


def test_export_real_build_with_audio_and_cli(tmp_path):
    sr = 4000
    t = np.arange(int(2.0 * sr)) / sr
    x = np.sin(2 * np.pi * 110 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
    x += 0.2 * np.random.default_rng(0).standard_normal(x.size)
    sf.write(str(tmp_path / "s.wav"), x, sr)
    LoopSpec(tmp_path / "s.wav", 0.2, 1.7, "demo", title="Demo loop").save(tmp_path / "demo.json")
    build_dir = tmp_path / "build"
    main(["plate", "build", "--loop", str(tmp_path / "demo.json"), "--pitch", "2", "--side", "40",
          "--cols", "1", "--rows", "2", "--thickness", "8", "--rho", "0.7", "--no-labels",
          "--out", str(build_dir)])
    (build_dir / "slice_metrics.json").write_text(json.dumps(
        {"preset": "generic04", "total": {"n": 2, "time_h": 3.5, "filament_g": 40.0,
                                          "filament_cm3": 32.0, "all_manifold": True}}))
    study = tmp_path / "study"
    study.mkdir()
    (study / "RESULTS.md").write_text("# study\n")
    (study / "q1_x.png").write_bytes(b"png")
    out = tmp_path / "export" / "demo"
    main(["plate", "export", "--build", str(build_dir), "--out", str(out),
          "--design-study", str(study)])
    files = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert {"audio/loop_clean.wav", "audio/loop_clean_x3.wav", "audio/loop_reconstructed_x3.wav",
            "audio/loop_as_printed_x3.wav", "analysis/preview_plate.png", "analysis/preview_bed.png",
            "analysis/preview_overhang.png", "renders/preview_plate.png",
            "analysis/design_study/RESULTS.md", "analysis/design_study/q1_x.png",
            "stl/strips/A1.stl", "stl/strips/A2.stl"} <= files
    sheet = json.loads((out / "spec_sheet.json").read_text())
    assert sheet["loop"]["name"] == "demo" and sheet["operator"]["n"] == 20
    assert sheet["operator"]["N"] == 14 and sheet["operator"]["nyquist_hz"] == pytest.approx(
        sheet["operator"]["f_hz"] / 2)
    assert len(sheet["operator"]["drift_per_pass"]) == 3
    assert sheet["slicer"]["slice_metrics.json"]["time_h"] == 3.5
    a = sheet["audio"]
    clean, r0 = sf.read(str(out / "audio" / "loop_clean.wav"))
    x3, _ = sf.read(str(out / "audio" / "loop_clean_x3.wav"))
    rec, r1 = sf.read(str(out / "audio" / "loop_reconstructed_x3.wav"))
    prt, r2 = sf.read(str(out / "audio" / "loop_as_printed_x3.wav"))
    assert r0 == sr and clean.size == 6000 and x3.size == 18000
    assert r1 == r2 == round(14 * 20 / 1.5) and rec.size == prt.size == 3 * 14 * 20
    assert a["loop_reconstructed_x3"]["step_error"]["max"] < 1e-6
    assert a["loop_as_printed_x3"]["levels"] == export.printed_levels(sheet["relief"]["chosen_mm"])
    assert "degraded" in a["loop_as_printed_x3"]["note"]
    md = (out / "spec_sheet.md").read_text()
    assert "## Loop" in md and "Demo loop" in md and "## Audio" in md and "generic04" in md
    readme = (out / "README.md").read_text()
    assert "loop_as_printed_x3.wav" in readme and "## How to print" in readme
