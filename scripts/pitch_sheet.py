"""Pitch contact sheet: how the 660 mm smooth plate looks at each cell pitch.

For every candidate (pitch, rho) the exact operator is built on the
corrected loop, turned into the cubic surface with the SAME printed range
(max_range_mm on a thickness_mm strip), and rendered as a hillshade with
fixed Lambertian shading (light azimuth 315, elevation 35 deg, ambient
0.3) computed from the surface gradient in mm/mm. Because the shading is
not auto-normalised, a panel with steeper facets really looks steeper.

Outputs in --out:
  pitch_sheet.png        window (100 mm) and zoom (30 mm) per candidate
  pitch_sheet_full.png   the whole plate per candidate, downsampled
  <cand>_window.png, <cand>_zoom.png, <cand>_full.png
  pitch_sheet.md         the table
  pitch_sheet.json       the numbers

  uv run python scripts/pitch_sheet.py --out runs/plate/pitch_sheet
"""

from __future__ import annotations

import argparse
import json
from math import cos, floor, radians, sin
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import welch

from loopviz.songmatrix import materialize

try:
    from loopviz.print3d.relief_cli import load_loop, probe
    from loopviz.print3d.surface import slope_stats, surface
except ModuleNotFoundError:                                 # flat package layout
    from loopviz.relief_cli import load_loop, probe
    from loopviz.surface import slope_stats, surface

AUDIO = Path("data/audio_3d/song.wav")
START, END = 324.68, 336.705           # corrected loop, T = 12.025 s
DRIFT_TOL = 1e-6


def shade(Z: np.ndarray, d: float, azimuth_deg: float = 315.0,
          elevation_deg: float = 35.0, ambient: float = 0.3) -> np.ndarray:
    """Lambertian shading in [0, 1] of a height field with node spacing d
    (same unit as Z). Azimuth clockwise from +y (north), so 315 is light
    from the upper left; nothing is normalised, so the look is absolute."""
    hy, hx = np.gradient(Z, d, d)
    az, el = radians(azimuth_deg), radians(elevation_deg)
    lx, ly, lz = cos(el) * sin(az), cos(el) * cos(az), sin(el)
    ndotl = (-hx * lx - hy * ly + lz) / np.sqrt(1.0 + hx * hx + hy * hy)
    return ambient + (1.0 - ambient) * np.clip(ndotl, 0.0, 1.0)


def crop(surf, S: np.ndarray, x0: float, x1: float, y0: float, y1: float):
    """Node-aligned crop of a shaded field, with the extent it covers."""
    i0, i1 = np.searchsorted(surf.y, [y0, y1])
    j0, j1 = np.searchsorted(surf.x, [x0, x1])
    i1, j1 = min(i1 + 1, S.shape[0]), min(j1 + 1, S.shape[1])
    half = 0.5 * surf.node_mm          # pixel blocks are centred on nodes
    ext = (surf.x[j0] - half, surf.x[j1 - 1] + half, surf.y[i0] - half, surf.y[i1 - 1] + half)
    return S[i0:i1, j0:j1], ext, (x0, x1, y0, y1)


def downsample(S: np.ndarray, px: int) -> np.ndarray:
    """Box-filter a shaded field to about px pixels on its long side."""
    im = Image.fromarray((S * 255).round().astype(np.uint8))
    h, w = S.shape
    k = px / max(h, w)
    return np.asarray(im.resize((max(1, round(w * k)), max(1, round(h * k))),
                                Image.BOX)) / 255.0


def energy_below(signal: np.ndarray, sr: int, nyq: float) -> float:
    freqs, psd = welch(signal, fs=sr, nperseg=8192)
    cum = np.cumsum(psd) / psd.sum()
    return float(np.interp(nyq, freqs, cum))


def candidate(signal, T, sr, pitch, rho, a) -> dict:
    n = floor(a.side / pitch)
    r = probe(signal, T, n, rho)
    pl = r["plan"]
    A = materialize(r["op"])
    surf = surface(A, pitch, thickness_mm=a.thickness, relief_mm=1000.0,
                   max_range_mm=a.thickness - a.body_min, subdiv=3, interp="cubic")
    st = slope_stats(surf)
    return {
        "pitch_mm": pitch, "rho": rho, "n": n, "N": pl.N, "f_hz": pl.f,
        "nyquist_hz": pl.f / 2, "drift": r["drift"], "gram_cond": r["gram_cond"],
        "energy_below_nyquist": energy_below(signal, sr, pl.f / 2),
        "printed_range_mm": surf.printed_range_mm, "relief_scale": surf.scale,
        "frac_facet_over_limit": st["frac_facet_over_limit"],
        "max_facet_deg": st["max_facet_deg"],
        "facet_q99_deg": st["facet_deg"]["q99"],
        "bed_stl_mb": 200.0 * (3 * n) ** 2 / 1e6,
        "surf": surf,
    }


def title(c: dict) -> str:
    t = (f"pitch {c['pitch_mm']:g} mm, n = {c['n']}, rho {c['rho']:g}\n"
         f"f {c['f_hz'] / 1e3:.1f} kHz, Nyq {c['nyquist_hz'] / 1e3:.1f} kHz, "
         f"over 45: {100 * c['frac_facet_over_limit']:.2f} %")
    if c["drift"] > DRIFT_TOL:
        t += f"\nDRIFT {c['drift']:.0e} > {DRIFT_TOL:g}"
    return t


def tag(c: dict) -> str:
    return f"p{c['pitch_mm']:g}_rho{c['rho']:g}"


def save_panel(S: np.ndarray, ext, lim, path: Path, ttl: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 6.9))
    ax.imshow(S, cmap="gray", vmin=0, vmax=1, origin="lower", extent=ext,
              interpolation="nearest")
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def sheet(rows: list[tuple[str, list[tuple[dict, callable]]]], path: Path,
          dpi: int, panel_in: float = 3.3, title_fs: float = 8.5) -> None:
    """Rows of equally sized square panels placed by hand (rows may differ
    in length). Each panel is one candidate rendered by one renderer, drawn
    with nearest-pixel display so the facet size stays visible."""
    cols = max(len(r) for _, r in rows)
    left, gap_x, gap_y, top_pad, bot_pad = 0.75, 0.3, 1.05, 0.55, 0.3
    fig_w = left + cols * (panel_in + gap_x) + 0.1
    fig_h = bot_pad + len(rows) * (panel_in + gap_y) + top_pad
    fig = plt.figure(figsize=(fig_w, fig_h))
    for ri, (label, panels) in enumerate(rows):
        y0 = fig_h - top_pad - (ri + 1) * (panel_in + gap_y) + gap_y - 0.35
        for k, (c, fn) in enumerate(panels):
            x0 = left + k * (panel_in + gap_x)
            ax = fig.add_axes([x0 / fig_w, y0 / fig_h, panel_in / fig_w, panel_in / fig_h])
            S, ext, lim = fn(c)
            ax.imshow(S, cmap="gray", vmin=0, vmax=1, origin="lower",
                      extent=ext, interpolation="nearest")
            ax.set_xlim(lim[0], lim[1])
            ax.set_ylim(lim[2], lim[3])
            ax.set_xticks([lim[0], lim[1]])
            ax.set_yticks([lim[2], lim[3]])
            ax.tick_params(labelsize=7, length=2, pad=1)
            lo, hi = ax.xaxis.get_majorticklabels()
            lo.set_ha("left")
            hi.set_ha("right")
            ax.set_title(title(c), fontsize=title_fs, linespacing=1.15)
            if k == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def md_table(cands: list[dict], a) -> str:
    hdr = ("| pitch mm | rho | n | N | f Hz | Nyquist Hz | energy below Nyq | "
           "drift / pass | over 45 deg (footprint) | facet q99 deg | max facet deg | "
           "bed.stl MB |")
    sep = "|" + "---|" * 12
    lines = [hdr, sep]
    for c in cands:
        lines.append(
            f"| {c['pitch_mm']:g} | {c['rho']:g} | {c['n']} | {c['N']} | {c['f_hz']:.0f} | "
            f"{c['nyquist_hz']:.0f} | {c['energy_below_nyquist']:.4f} | {c['drift']:.1e} | "
            f"{100 * c['frac_facet_over_limit']:.2f} % | {c['facet_q99_deg']:.1f} | "
            f"{c['max_facet_deg']:.1f} | {c['bed_stl_mb']:.0f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="runs/plate/pitch_sheet")
    ap.add_argument("--side", type=float, default=660.0, help="plate side (mm)")
    ap.add_argument("--thickness", type=float, default=13.0, help="strip (mm)")
    ap.add_argument("--body-min", type=float, default=3.0,
                    help="thinnest point (mm): printed range = thickness - body-min")
    ap.add_argument("--window", type=float, default=100.0, help="window side (mm)")
    ap.add_argument("--pitches", default="1.2,1.35,1.5,1.7,2.0,2.5",
                    help="pitches at --rho-main")
    ap.add_argument("--rho-main", type=float, default=0.95)
    ap.add_argument("--rhos", default="0.95,0.8,0.6",
                    help="rho sweep at --rho-pitch")
    ap.add_argument("--rho-pitch", type=float, default=1.5)
    ap.add_argument("--extra", default="1.2:0.6,1.35:0.8",
                    help="pitch:rho pairs to add (default: the fine pitches at a "
                         "rho that keeps f below the 16 kHz source rate, where "
                         "rho 0.95 has no stable operator); empty for none")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--full-px", type=int, default=600)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pitches = [float(v) for v in a.pitches.split(",")]
    rhos = [float(v) for v in a.rhos.split(",")]
    extra = [tuple(float(v) for v in pr.split(":")) for pr in a.extra.split(",") if pr]

    signal, T, sr = load_loop(AUDIO, START, END)
    print(f"loop {START}..{END} s: T = {T:.3f} s, {sr} Hz, {signal.size} samples")

    cache: dict[tuple[float, float], dict] = {}

    def get(pitch, rho):
        key = (pitch, rho)
        if key not in cache:
            c = candidate(signal, T, sr, pitch, rho, a)
            cache[key] = c
            print(f"pitch {pitch:g} rho {rho:g}: n {c['n']} N {c['N']} f {c['f_hz']:.0f} "
                  f"drift {c['drift']:.1e} range {c['printed_range_mm']:.2f} mm "
                  f"over45 {100 * c['frac_facet_over_limit']:.2f} % "
                  f"max {c['max_facet_deg']:.1f} deg", flush=True)
        return cache[key]

    set_pitch = [get(p, a.rho_main) for p in pitches]
    set_rho = [get(a.rho_pitch, r) for r in rhos]
    set_cap = [get(p, r) for p, r in extra]

    # same window in every panel: centred-ish on the plate, away from edges
    wx0, wy0 = 300.0, 150.0
    wx1, wy1 = wx0 + a.window, wy0 + a.window
    zx0, zy0, zs = 320.0, 190.0, 30.0

    for c in cache.values():
        c["S"] = shade(c["surf"].Z, c["surf"].node_mm)

    def window(c):
        return crop(c["surf"], c["S"], wx0, wx1, wy0, wy1)

    def zoom(c):
        return crop(c["surf"], c["S"], zx0, zx0 + zs, zy0, zy0 + zs)

    def full(c):
        s = c["surf"]
        half = 0.5 * s.node_mm
        ext = (s.x[0] - half, s.x[-1] + half, s.y[0] - half, s.y[-1] + half)
        return downsample(c["S"], a.full_px), ext, (0, a.side, 0, a.side)

    for c in cache.values():
        t = title(c)
        S, ext, lim = window(c)
        save_panel(S, ext, lim, out / f"{tag(c)}_window.png", f"{a.window:g} mm window\n{t}", a.dpi)
        S, ext, lim = zoom(c)
        save_panel(S, ext, lim, out / f"{tag(c)}_zoom.png", f"{zs:g} mm zoom\n{t}", a.dpi)
        S, ext, lim = full(c)
        save_panel(S, ext, lim, out / f"{tag(c)}_full.png", f"full plate\n{t}", a.dpi)

    row2 = set_rho + set_cap
    lab1 = f"rho {a.rho_main:g}, pitch sweep"
    lab2 = (f"pitch {a.rho_pitch:g} rho sweep" + (f"; f <= {sr / 1e3:g} kHz at "
            + ", ".join(f"{p:g}" for p, _ in extra) if extra else ""))
    sheet([(f"{lab1}\n{a.window:g} mm window", [(c, window) for c in set_pitch]),
           (f"{lab1}\n{zs:g} mm zoom", [(c, zoom) for c in set_pitch]),
           (f"{lab2}\n{a.window:g} mm window", [(c, window) for c in row2]),
           (f"{lab2}\n{zs:g} mm zoom", [(c, zoom) for c in row2])],
          out / "pitch_sheet.png", a.dpi)
    sheet([(f"{lab1}\nfull plate", [(c, full) for c in set_pitch]),
           (f"{lab2}\nfull plate", [(c, full) for c in row2])],
          out / "pitch_sheet_full.png", a.dpi, panel_in=5.5, title_fs=11)

    cands = set_pitch + set_rho[1:] + set_cap
    table = md_table(cands, a)
    rng = a.thickness - a.body_min
    md = f"""# pitch sheet: the 660 mm plate at each candidate pitch

Loop {START} to {END} s of {AUDIO.name} (T = {T:.3f} s, {sr} Hz). Plate side
{a.side:g} mm, n = floor({a.side:g} / pitch), N = round(rho n), f = N n / T.
Every candidate is the exact operator (relief_cli.probe), materialised,
clipped at the 99.5 percentile and interpolated bicubic to 3 nodes per
cell; the relief is scaled so the printed range is exactly {rng:g} mm on a
{a.thickness:g} mm strip (thinnest point {a.body_min:g} mm). "over 45 deg" is the
footprint fraction of mesh facets whose on-edge overhang exceeds 45 deg
(surface.slope_stats, facet rule); facet q99 is the 99th percentile of
that angle. "energy below Nyq" is the share of the loop's Welch power
(8192-point segments) below f/2, as in exp_smooth Q1; bed.stl MB is
200 bytes (3 n)^2 / 1e6 (binary STL, 50 bytes per triangle, 2 triangles
per node cell, top and bottom).

{table}

Renders (`pitch_sheet.png`): {a.window:g} mm window x {wx0:g}..{wx1:g}, y {wy0:g}..{wy1:g}
and {zs:g} mm zoom x {zx0:g}..{zx0 + zs:g}, y {zy0:g}..{zy0 + zs:g} of the plate, the same
in every panel. Lambertian shading from the surface gradient, light
azimuth 315 (upper left), elevation 35 deg, ambient 0.3, fixed grey
scale (no per-panel normalisation), one display pixel block per node so
the facet size is visible. `pitch_sheet_full.png`: the whole plate box
filtered to about {a.full_px} px. Individual panels: `<p>_<rho>_window.png`,
`_zoom.png`, `_full.png`.

Notes: a drift of 1 means the operator does not exist at that (n, rho):
f exceeds the {sr / 1e3:g} kHz source rate, the resampled windows carry nothing
above {sr / 2e3:g} kHz and the Gram matrix is singular, so the surface drawn is
the inverse of numerical noise (pure corduroy). {("The extra rows " + ", ".join(f"{p:g} mm at rho {r:g}" for p, r in extra) + " keep f below the source rate and are the usable versions of those pitches.") if extra else ""}

Reproduce: `uv run python scripts/pitch_sheet.py --out {a.out}`.
"""
    (out / "pitch_sheet.md").write_text(md)
    (out / "pitch_sheet.json").write_text(json.dumps(
        [{k: v for k, v in c.items() if k not in ("surf", "S")} for c in cands], indent=2))
    print(table)
    print(f"wrote {out}/pitch_sheet.png, pitch_sheet_full.png, pitch_sheet.md")


if __name__ == "__main__":
    main()
