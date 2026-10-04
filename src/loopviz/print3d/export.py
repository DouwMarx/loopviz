"""`loopviz plate export`: a self-contained package from a finished build.

  loopviz plate export --build runs/plate/armed_man_p2 --out export/armed_man_p2 [--zip]

Package layout:

  README.md                  what each file is, how to print, what the object is
  spec_sheet.md, .json       every parameter that produced the build, provenance
  audio/loop_clean.wav       the excerpt at the source rate
  audio/loop_clean_x3.wav    the excerpt three times
  audio/loop_reconstructed_x3.wav
                             the exact operator played: seed window 1, N steps
                             per pass, 3 passes, window norms restored, at round(f) Hz
  audio/loop_as_printed_x3.wav
                             the same with the operator the plate carries (clipped,
                             heights in 0.05 mm steps); degraded by design
  analysis/                  previews, metrics.json, layout.json, assembly.md, the
                             design study (RESULTS.md and figures) and renders if present
  stl/plate_full.stl         the uncut plate (reference geometry)
  stl/bed_print_as_is.stl    every strip placed on the bed: print this
  stl/strips/<label>.stl     the same, one file per strip
  renders/                   render/*.png if present, else the previews

Builds without audio (`loopviz plate demo`) get no audio/ and no operator
section; everything else is exported.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import __version__
from ..loopspec import LoopSpec
from ..songmatrix import Plan, build, loop_degradation, materialize, windows
from .relief import quantized_matrix

PRINTED_STEP_MM = 0.05      # height resolution of the printed relief
PASSES = 3
PREVIEWS = ("preview_overhang.png", "preview_plate.png", "preview_bed.png")
ANALYSIS_FILES = PREVIEWS + ("metrics.json", "layout.json", "assembly.md")


# -- the operator played ----------------------------------------------------------------

def play(A: np.ndarray, W: np.ndarray, norms: np.ndarray, passes: int = PASSES,
         blowup: float = 100.0) -> tuple[np.ndarray, dict]:
    """Seed with window 1, iterate A for N steps per pass, `passes` passes;
    every played window is scaled back by the norm of the window it stands
    for. Returns passes * N * n samples at the operator's rate f and a
    record of where the iteration diverged, if it did: an unstable
    operator (the quantised one usually is) grows exponentially, so once
    the state exceeds `blowup` times a unit window the rest is silence
    rather than overflow."""
    N, n = W.shape[1], W.shape[0]
    x = W[:, 0].copy()
    out = np.zeros(passes * N * n)
    info = {"diverged_at_window": None, "diverged_at_s": None}
    for step in range(passes * N):
        k = step % N
        nx = float(np.linalg.norm(x))
        if not np.isfinite(nx) or nx > blowup:
            info = {"diverged_at_window": step, "diverged_at_s": None}
            break
        out[step * n:(step + 1) * n] = x * norms[k]
        x = A @ x
    return out, info


def printed_levels(relief_mm: float, step_mm: float = PRINTED_STEP_MM) -> int:
    return round(relief_mm / step_mm) + 1


def printed_matrix(A: np.ndarray, relief_mm: float, clip_pct: float,
                   step_mm: float = PRINTED_STEP_MM) -> tuple[np.ndarray, int]:
    """The operator as the plate carries it: |A| clipped at the clip_pct
    percentile (the entries above it are flattened to full relief) and the
    height quantised to step_mm over the relief range."""
    levels = printed_levels(relief_mm, step_mm)
    return quantized_matrix(A, levels, clip_pct), levels


def step_error(A: np.ndarray, W: np.ndarray) -> dict:
    Xn = np.roll(W, -1, axis=1)
    err = np.linalg.norm(A @ W - Xn, axis=0) / np.linalg.norm(Xn, axis=0)
    return {"max": float(err.max()), "mean": float(err.mean())}


def write_wav(path: Path, x: np.ndarray, rate: int) -> dict:
    """16-bit PCM; scaled down when the peak exceeds 1 so an unstable
    operator does not clip into noise. Returns what was written."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    scale = 1.0 / peak if peak > 1.0 else 1.0
    sf.write(str(path), x * scale, rate, subtype="PCM_16")
    return {"file": path.name, "rate_hz": rate, "samples": int(x.size),
            "seconds": x.size / rate, "peak": peak, "scaled_by": scale}


def export_audio(spec: LoopSpec, n: int, N: int, relief_mm: float, clip_pct: float,
                 out: Path, loop_wav: Path | None = None) -> dict:
    """The four wavs; returns the audio section of the spec sheet."""
    out.mkdir(parents=True, exist_ok=True)
    signal, T, sr = spec.signal()
    if loop_wav is not None and loop_wav.exists():
        shutil.copy2(loop_wav, out / "loop_clean.wav")
    else:
        sf.write(str(out / "loop_clean.wav"), signal, sr)
    sf.write(str(out / "loop_clean_x3.wav"), np.tile(signal, PASSES), sr)

    pl = Plan(n=n, N=N, f=N * n / T, T=T)
    norms = np.linalg.norm(windows(signal, pl), axis=0)
    op, W = build(signal, pl)
    A = materialize(op)
    rate = round(pl.f)
    x, info = play(A, W, norms)
    exact = write_wav(out / "loop_reconstructed_x3.wav", x, rate)
    exact.update(info)
    exact["drift_per_pass"] = loop_degradation(A, W, loops=PASSES)
    exact["step_error"] = step_error(A, W)

    Aq, levels = printed_matrix(A, relief_mm, clip_pct)
    x, info = play(Aq, W, norms)
    if info["diverged_at_window"] is not None:
        info["diverged_at_s"] = info["diverged_at_window"] * n / pl.f
    printed = write_wav(out / "loop_as_printed_x3.wav", x, rate)
    printed.update({**info, "levels": levels, "step_mm": PRINTED_STEP_MM, "clip_pct": clip_pct,
                    "drift_per_pass": loop_degradation(Aq, W, loops=PASSES),
                    "step_error": step_error(Aq, W),
                    "note": "degraded by design: the clip flattens the largest entries, which "
                            "carry playback, and the height steps quantise the rest; an unstable result grows "
                            "exponentially and is cut to silence once it exceeds 100 unit windows "
                            "(doc/relief.md, 'Does the printed object still play?')"})
    return {"source_hz": sr, "T_s": T, "plan": {"n": n, "N": N, "f_hz": pl.f, "rho": pl.rho,
                                                "nyquist_hz": pl.f / 2},
            "loop_clean": {"file": "loop_clean.wav", "rate_hz": sr, "seconds": T},
            "loop_clean_x3": {"file": "loop_clean_x3.wav", "rate_hz": sr, "seconds": PASSES * T},
            "loop_reconstructed_x3": exact, "loop_as_printed_x3": printed}


# -- provenance ----------------------------------------------------------------------------

def git_state(cwd: Path | None = None) -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": rev, "dirty": dirty}


def slicer_metrics(build_dir: Path) -> dict:
    """slice_metrics.json, then any slice_<preset>.json, as written by
    scripts/slice_metrics.py; {} when the build was never sliced."""
    found = {}
    for p in sorted(build_dir.glob("slice*.json")):
        d = json.loads(p.read_text())
        t = d.get("total", d)
        found[p.name] = {"preset": d.get("preset"), "time_h": t.get("time_h"),
                         "filament_g": t.get("filament_g"), "filament_cm3": t.get("filament_cm3"),
                         "all_manifold": t.get("all_manifold"), "files": t.get("n")}
    return found


def loop_of(metrics: dict, build_dir: Path) -> LoopSpec | None:
    """The loop the build was made from: the recorded spec, else the bare
    audio/start/end of older builds, else None (demo plates)."""
    if "loop" in metrics:
        return LoopSpec(**metrics["loop"])
    if metrics.get("audio") and metrics.get("start_s") is not None:
        return LoopSpec(Path(metrics["audio"]), metrics["start_s"], metrics["end_s"],
                        name=build_dir.name)
    return None


# -- spec sheet ---------------------------------------------------------------------------

def spec_sheet(metrics: dict, layout: dict, build_dir: Path, audio: dict | None,
               spec: LoopSpec | None) -> dict:
    P = layout["params"]
    bed = layout["bed"]
    rc = metrics.get("relief_choice", {})
    ov = metrics["overhang_relief_face"]
    worst = max(layout["strips"], key=lambda r: r["overhang_relief_face"]["max_deg"])
    sizes = sorted({tuple(round(v, 1) for v in r["size_mm"]) for r in layout["strips"]})
    built = dt.datetime.fromtimestamp((build_dir / "metrics.json").stat().st_mtime,
                                      tz=dt.UTC).astimezone()
    sheet = {
        "build": str(build_dir),
        "loop": spec.to_dict() if spec else None,
        "operator": None if "N" not in metrics else {
            "n": metrics["n"], "N": metrics["N"], "rho": metrics["rho"], "f_hz": metrics["f_hz"],
            "nyquist_hz": metrics["f_hz"] / 2, "T_s": metrics["T_s"],
            "source_hz": metrics["source_hz"], "gram_cond": metrics.get("gram_cond"),
            "drift_per_pass_at_build": metrics.get("drift"),
            "drift_per_pass": audio["loop_reconstructed_x3"]["drift_per_pass"] if audio else None,
            "transpose": metrics.get("transpose", False)},
        "plate": {
            "pitch_mm": P["pitch_mm"], "side_mm": metrics["side_mm"], "n": metrics["n"],
            "cols": P["cols"], "rows": P["rows"], "thickness_mm": P["thickness_mm"],
            "strips": metrics["strips"], "strip_sizes_mm": [list(s) for s in sizes],
            "orientation": P["orientation"], "nodes": metrics["nodes"],
            "faces_total": metrics["faces_total"]},
        "relief": {
            "mode": rc.get("mode"), "rule": rc.get("rule"),
            "fit_facet_mm": rc.get("fit_facet_mm"), "fit_y_mm": rc.get("fit_y_mm"),
            "max_overhang_deg": rc.get("max_overhang_deg"), "area_quantile": rc.get("area_quantile"),
            "requested_mm": rc.get("requested_mm", P.get("relief_requested_mm")),
            "chosen_mm": metrics["relief_mm"], "scale": rc.get("scale"),
            "printed_range_mm": metrics["printed_range_mm"],
            "body_min_mm": P["body_min_mm"], "cap_mm": rc.get("cap_mm"),
            "interp": P["interp"], "subdiv": P["subdiv"], "clip_pct": P["clip_pct"],
            "mapping": P.get("mapping", "clip"), "sigma_cells": P["sigma_cells"]},
        "labels": {"on": P["labels"], "label_mm": P["label_mm"], "engrave_mm": P["engrave_mm"],
                   "label_margin_mm": P["label_margin_mm"]},
        "bed": {"size_mm": bed["size_mm"], "zmax_mm": bed["zmax_mm"], "margin_mm": bed["margin_mm"],
                "gap_mm": bed["gap_mm"], "brim_mm": bed["brim_mm"],
                "y_extent_mm": layout["bed_usage"]["y_extent_mm"], "k_slender": P["k_slender"]},
        "overhang": {
            "relief_face_over_45_mm2": ov["area_over_45_mm2"],
            "relief_face_mm2": ov["relief_area_mm2"],
            "relief_face_over_45_frac": ov["frac_over_45_of_relief"],
            "max_deg": ov["max_deg"], "worst_strip": worst["label"],
            "bed_over_45_mm2": metrics["overhang_bed"]["area_over_45_mm2"]},
        "mass": {"solid_g": metrics["mass_g"], "volume_mm3": metrics["volume_mm3"],
                 "shell_estimate_kg": metrics.get("material_estimate", {}).get("mass_kg"),
                 "crude_hours": metrics.get("material_estimate", {}).get("hours")},
        "slicer": slicer_metrics(build_dir),
        "files_mb": metrics["files_mb"],
        "audio": audio,
        "provenance": {"loopviz_version": __version__, **git_state(),
                       "built": built.isoformat(timespec="seconds"),
                       "build_seconds": metrics.get("build_seconds"),
                       "exported": dt.datetime.now(dt.UTC).astimezone().isoformat(timespec="seconds"),
                       "verify_max_dev_mm": metrics.get("verify", {}).get("max_dev_mm")},
    }
    return sheet


def _g(v, fmt=":g"):
    return "n/a" if v is None else format(v, fmt.strip(":"))


def spec_sheet_md(s: dict) -> str:
    L = [f"# Spec sheet: {Path(s['build']).name}", ""]
    if s["loop"]:
        lp = s["loop"]
        L += ["## Loop", "", f"- name: {lp['name']}"]
        if lp.get("title"):
            L.append(f"- title: {lp['title']}")
        L.append(f"- audio: {lp['audio']}, {lp['start_s']} to {lp['end_s']} s "
                 f"(T = {lp['end_s'] - lp['start_s']:.3f} s)")
        if lp.get("notes"):
            L.append(f"- notes: {lp['notes']}")
        L.append("")
    o = s["operator"]
    if o:
        drift = o["drift_per_pass"]
        L += ["## Operator", "",
              f"- n = {o['n']}, N = {o['N']}, rho = {o['rho']:.4f}",
              (f"- f = {o['f_hz']:.1f} Hz (Nyquist {o['nyquist_hz']:.1f} Hz), source {o['source_hz']} Hz, "
              f"T = {o['T_s']:.3f} s"),
              (f"- Gram condition {_g(o['gram_cond'], ':.3g')}, drift per pass at build "
              f"{_g(o['drift_per_pass_at_build'], ':.2e')}"),
              "- drift after 1, 2, 3 passes: " + (", ".join(f"{d:.2e}" for d in drift) if drift else "n/a"),
              f"- transpose: {o['transpose']}", ""]
    p, r, lb, b = s["plate"], s["relief"], s["labels"], s["bed"]
    L += ["## Plate", "",
          f"- pitch {p['pitch_mm']:g} mm, side {p['side_mm']:g} mm, n = {p['n']}",
          (f"- {p['cols']} x {p['rows']} strips x {p['thickness_mm']:g} mm thick, on {p['orientation']}; "
          f"strip sizes (l x h x t, mm): " + "; ".join(" x ".join(f"{v:g}" for v in sz) for sz in p["strip_sizes_mm"])),
          f"- nodes {p['nodes'][0]} x {p['nodes'][1]}, {p['faces_total'] / 1e6:.2f} M faces", "",
          "## Relief", "",
          (f"- mode {r['mode']}, rule {r['rule']}, {_g(r['max_overhang_deg'])} deg over "
          f"{_g(r['area_quantile'])} of the footprint"),
          (f"- fit: facet rule {_g(r['fit_facet_mm'], ':.2f')} mm, y rule {_g(r['fit_y_mm'], ':.2f')} mm; "
          f"requested {_g(r['requested_mm'], ':.2f')} mm"),
          (f"- chosen {r['chosen_mm']:.2f} mm (scale {_g(r['scale'], ':.3f')}), printed range "
          f"{r['printed_range_mm'][0]:.2f} to {r['printed_range_mm'][1]:.2f} mm "
          f"({r['printed_range_mm'][2]:.2f} mm), body min {r['body_min_mm']:g} mm, cap {_g(r['cap_mm'])} mm"),
          (f"- mapping {r['mapping']}, clip {r['clip_pct']:g} %, interp {r['interp']}, subdiv {r['subdiv']}, "
           f"sigma {r['sigma_cells']:g} cells"),
          "", "## Labels", "",
          (f"- {'on' if lb['on'] else 'off'}: cap {_g(lb['label_mm'])} mm, engraved {_g(lb['engrave_mm'])} mm, "
           f"margin {_g(lb['label_margin_mm'])} mm"), "",
          "## Bed", "",
          (f"- {b['size_mm']:g} mm square, Z max {b['zmax_mm']:g} mm, margin {b['margin_mm']:g} mm, "
          f"gap {b['gap_mm']:g} mm, brim {b['brim_mm']:g} mm"),
          f"- strips end at Y = {b['y_extent_mm']:.1f} mm; slenderness limit h/t = {b['k_slender']:g}", ""]
    ov, m = s["overhang"], s["mass"]
    L += ["## Overhang (relief face, print orientation)", "",
          (f"- over 45 deg: {ov['relief_face_over_45_mm2'] / 100:.1f} of {ov['relief_face_mm2'] / 100:.0f} cm^2 "
          f"({100 * ov['relief_face_over_45_frac']:.2f} %), max {ov['max_deg']:.0f} deg, worst strip {ov['worst_strip']}"),
          f"- whole bed incl. engraving: {ov['bed_over_45_mm2'] / 100:.1f} cm^2 over 45 deg", "",
          "## Mass and time", "",
          (f"- solid {m['solid_g'] / 1000:.2f} kg ({m['volume_mm3'] / 1000:.0f} cm^3); shell estimate "
          f"{_g(m['shell_estimate_kg'], ':.2f')} kg, crude {_g(m['crude_hours'], ':.0f')} h")]
    if s["slicer"]:
        for name, sl in s["slicer"].items():
            L.append(f"- slicer ({name}, preset {sl['preset']}): {_g(sl['time_h'], ':.1f')} h, "
                     f"{_g(sl['filament_g'], ':.0f')} g filament, manifold {sl['all_manifold']}, "
                     f"{sl['files']} files")
    else:
        L.append("- slicer: not sliced (no slice_*.json in the build)")
    L.append("")
    if s["audio"]:
        a = s["audio"]
        ex, pr = a["loop_reconstructed_x3"], a["loop_as_printed_x3"]
        L += ["## Audio", "",
              f"- loop_clean.wav: the excerpt, {a['source_hz']} Hz, {a['T_s']:.3f} s; loop_clean_x3.wav: three times",
              (f"- loop_reconstructed_x3.wav: exact operator, {ex['rate_hz']} Hz, {ex['seconds']:.2f} s, "
              f"step error max {ex['step_error']['max']:.2e}, drift per pass "
              + ", ".join(f"{d:.2e}" for d in ex["drift_per_pass"])),
              (f"- loop_as_printed_x3.wav: {pr['levels']} levels of {pr['step_mm']} mm, clip {pr['clip_pct']} %, "
              f"step error max {pr['step_error']['max']:.2e}, drift per pass "
              + ", ".join(f"{d:.2e}" for d in pr["drift_per_pass"])
              + (f", peak {pr['peak']:.2g} scaled by {pr['scaled_by']:.2g}" if pr["scaled_by"] != 1 else "")
              + (f"; diverged after {pr['diverged_at_window']} windows ({pr['diverged_at_s']:.2f} s), "
                 f"silence from there" if pr["diverged_at_window"] is not None else "")),
              f"  {pr['note']}", ""]
    pv = s["provenance"]
    L += ["## Files (MB)", ""] + [f"- {k}: {v}" for k, v in s["files_mb"].items()] + ["",
          "## Provenance", "",
          (f"- loopviz {pv['loopviz_version']}, commit {pv['commit'] or 'unknown'}"
          + (" (dirty)" if pv["dirty"] else "")),
          f"- built {pv['built']} in {_g(pv['build_seconds'], ':.0f')} s; exported {pv['exported']}",
          f"- reassembly check: strips deviate from the plate by {_g(pv['verify_max_dev_mm'], ':.1e')} mm", ""]
    return "\n".join(L)


# -- README ------------------------------------------------------------------------------

def readme_md(s: dict, assembly: str, has_renders: bool, has_study: bool) -> str:
    name = Path(s["build"]).name
    assembly = re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", assembly, flags=re.MULTILINE)
    o, p, lp = s["operator"], s["plate"], s["loop"]
    what = (f"This is a song as the matrix that plays it. The loop ({lp['title'] or lp['name']}, "
            f"{lp['start_s']} to {lp['end_s']} s of {Path(lp['audio']).name}) is cut into N = {o['N']} "
            f"consecutive windows of n = {o['n']} samples at f = {o['f_hz']:.0f} Hz. The n x n matrix A "
            f"with A w_k = w_(k+1) and A w_N = w_1 advances the song one window at a time and closes "
            f"the loop; iterating it plays the song exactly, forever. The plate is that matrix as a "
            f"relief: cell (i, j) is entry A[i, j], row 0 at the bottom, height proportional to the "
            f"value, mapped to [-1, 1] by '{s['relief']['mapping']}' (clip percentile {s['relief']['clip_pct']:g}), interpolated "
            f"({s['relief']['interp']}) to a smooth surface {p['side_mm']:g} mm square and cut into "
            f"{p['cols']} x {p['rows']} strips that print standing on edge in one job."
            if o and lp else
            f"A demo plate from `loopviz plate demo`: a hand-made {p['n']}-row matrix as a relief, "
            f"{p['side_mm']:g} mm wide, cut into {p['cols']} x {p['rows']} strips.")
    audio = ("""- `audio/loop_clean.wav`: the excerpt at the source rate; `loop_clean_x3.wav` plays it three times.
- `audio/loop_reconstructed_x3.wav`: the operator played. Window 1 seeds it, A is applied N times per
  pass for 3 passes, every window is scaled back by its norm, written at round(f) Hz. This is what the
  matrix on the wall encodes; the drift per pass in the spec sheet is how far it is from the clean loop.
- `audio/loop_as_printed_x3.wav`: the same with the operator the plate carries: clipped at the clip
  percentile and heights in 0.05 mm steps. Expected to be degraded: the clipped entries carry playback
  and the height quantisation removes about ten bits (see the spec sheet for its step error and drift).
""" if s["audio"] else "- no audio: this build has no loop (demo plate).\n")
    return f"""# {name}

{what}

## Files

- `README.md`: this file. `spec_sheet.md` / `spec_sheet.json`: every parameter that produced the
  build, provenance (commit, version, dates), overhang, mass, slicer numbers if sliced.
{audio}- `stl/bed_print_as_is.stl`: every strip placed and oriented on the printer bed; print this.
- `stl/strips/<label>.stl`: the same, one file per strip, same placement.
- `stl/plate_full.stl`: the uncut plate in the plate frame, reference geometry, not a print file.
- `analysis/`: `preview_plate.png` (hillshade with seams and labels), `preview_bed.png` (bed drawing
  and oblique mesh), `preview_overhang.png` (worst strip coloured by overhang), `metrics.json`,
  `layout.json` (bed, per-strip boxes and 4x4 transforms), `assembly.md`{
  ', the design study (`RESULTS.md` and figures)' if has_study else ''}{
  ', `render/` images' if has_renders else ''}.
- `renders/`: {'rendered views of the STLs' if has_renders else 'the preview images (no renders were made)'}.

## How to print and assemble

{assembly}
"""


# -- package -----------------------------------------------------------------------------

def _copy(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def export_build(build_dir: Path, out: Path, design_study: Path | None = None,
                 zip_it: bool = False, verbose: bool = True) -> dict:
    """Assemble the package; returns {"files": [...], "sheet": ..., "zip": ...}."""
    build_dir, out = Path(build_dir), Path(out)
    say = print if verbose else (lambda *a, **k: None)
    for req in ("metrics.json", "layout.json", "assembly.md", "plate.stl", "bed.stl"):
        if not (build_dir / req).exists():
            raise SystemExit(f"{build_dir} is not a finished plate build: missing {req}")
    metrics = json.loads((build_dir / "metrics.json").read_text())
    layout = json.loads((build_dir / "layout.json").read_text())
    out.mkdir(parents=True, exist_ok=True)

    spec = loop_of(metrics, build_dir)
    audio = None
    if spec is not None:
        if not spec.audio.exists():
            raise SystemExit(f"audio {spec.audio} of the loop not found; run from the project root")
        say(f"audio: {spec.name}, n = {metrics['n']}, N = {metrics['N']}")
        audio = export_audio(spec, metrics["n"], metrics["N"], metrics["relief_mm"],
                             layout["params"]["clip_pct"], out / "audio", build_dir / "loop.wav")
        say(f"  exact drift per pass {audio['loop_reconstructed_x3']['drift_per_pass']}, as printed "
            f"{audio['loop_as_printed_x3']['drift_per_pass']} ({audio['loop_as_printed_x3']['levels']} levels)")

    for f in ANALYSIS_FILES:
        _copy(build_dir / f, out / "analysis" / f)
    has_study = False
    if design_study and design_study.exists():
        for f in [design_study / "RESULTS.md", *sorted(design_study.glob("q*.png"))]:
            if _copy(f, out / "analysis" / "design_study" / f.name):
                has_study = True
    renders = sorted((build_dir / "render").glob("*.png"))
    for f in renders:
        _copy(f, out / "analysis" / "render" / f.name)
        _copy(f, out / "renders" / f.name)
    if not renders:
        for f in PREVIEWS:
            _copy(build_dir / f, out / "renders" / f)

    _copy(build_dir / "plate.stl", out / "stl" / "plate_full.stl")
    _copy(build_dir / "bed.stl", out / "stl" / "bed_print_as_is.stl")
    for f in sorted((build_dir / "strips").glob("*.stl")):
        _copy(f, out / "stl" / "strips" / f.name)
    say("STLs copied")

    sheet = spec_sheet(metrics, layout, build_dir, audio, spec)
    (out / "spec_sheet.json").write_text(json.dumps(sheet, indent=1))
    (out / "spec_sheet.md").write_text(spec_sheet_md(sheet))
    (out / "README.md").write_text(readme_md(sheet, (build_dir / "assembly.md").read_text(),
                                             bool(renders), has_study))
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    result = {"files": files, "sheet": sheet, "zip": None,
              "bytes": sum((out / f).stat().st_size for f in files)}
    if zip_it:
        z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        result["zip"] = z
        say(f"zip: {z} ({Path(z).stat().st_size / 1e6:.1f} MB; the STLs dominate)")
    say(f"{len(files)} files, {result['bytes'] / 1e6:.1f} MB in {out}/")
    return result


def cmd_export(args) -> None:
    export_build(Path(args.build), Path(args.out), design_study=Path(args.design_study),
                 zip_it=args.zip)


def add_export_parser(ss) -> None:
    q = ss.add_parser("export", help="self-contained package (audio, spec sheet, STLs, "
                                     "analysis, renders, README) from a finished build")
    q.add_argument("--build", required=True, help="runs/plate/<name>")
    q.add_argument("--out", required=True, help="export/<name>")
    q.add_argument("--design-study", default="runs/exp_smooth",
                   help="directory with RESULTS.md and q*.png to include if present")
    q.add_argument("--zip", action="store_true", help="also write <out>.zip")
    q.set_defaults(fn=cmd_export)
