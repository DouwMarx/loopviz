"""Design experiments for `loopviz plate`: a smooth (interpolated) relief of
the exact song operator, printed as strips standing on edge.

Six questions (see runs/exp_smooth/RESULTS.md for the tables and readings):

1. Spectrum of the loop: how much of the music survives resampling to the
   f implied by each cell pitch on a 660 mm plate (f = rho n^2 / T).
2. Roughness of the exact operator A at those n: neighbour differences,
   lag-1 autocorrelation and the 2D spectrum, along rows and columns.
3. Interpolation of a 24 x 24 patch: nearest / bilinear / bicubic /
   fourier, overshoot and slope.
4. Relief amplitude allowed by the on-edge overhang limit, per quantile
   of surface area, with and without Gaussian pre-smoothing.
5. FDM overhang geometry table (unsupported line fraction).
6. Null-space smoothing: how much smoother can an exact operator be made
   by alternating low-pass and projection onto {A : A W = W_next}.

Run: uv run python scripts/exp_smooth_surface.py --out runs/exp_smooth
     [--only q1 q3 ...]   deterministic, about a minute.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from scipy.ndimage import gaussian_filter
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from loopviz.loopspec import load_loop
from loopviz.print3d.relief_cli import max_feasible_rho, probe
from loopviz.print3d.surface import cell_values as surface_cells
from loopviz.print3d.surface import interpolate
from loopviz.songmatrix import loop_degradation, materialize

ROOT = Path(__file__).resolve().parents[2]
SIDE_MM = 660.0
PITCHES = (3.0, 2.5, 2.0, 1.5, 1.39)
RHO = 0.95
CLIP_PCT = 99.5
DPI = 120
DRIFT_TOL = 1e-6


# -- small helpers ---------------------------------------------------------------

def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(v):
        if isinstance(v, float):
            return f"{v:.3g}" if (abs(v) < 1e-3 or abs(v) >= 1e4) and v != 0 else f"{v:.4g}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def n_for(pitch: float) -> int:
    return int(np.floor(SIDE_MM / pitch))


def f_for(n: int, T: float, rho: float = RHO) -> float:
    return rho * n * n / T


def cell_values(A: np.ndarray, clip_pct: float = CLIP_PCT,
                sigma_cells: float = 0.0) -> tuple[np.ndarray, float]:
    """loopviz.print3d.surface.cell_values plus the clip value m (blurred v is
    renormalised to [-1, 1] there, so relief stays peak to peak)."""
    return surface_cells(A, clip_pct, sigma_cells), float(np.percentile(np.abs(A), clip_pct))


INTERP = {
    "nearest": lambda v, s: interpolate(v, s, "nearest"),
    "bilinear": lambda v, s: interpolate(v, s, "linear"),
    "bicubic": lambda v, s: interpolate(v, s, "cubic"),
    "fourier": lambda v, s: interpolate(v, s, "fourier"),
}


def lag1(A: np.ndarray) -> tuple[float, float]:
    """Lag-1 autocorrelation along i (rows, y) and along j (columns, x)."""
    B = A - A.mean()
    var = (B * B).sum()
    ri = (B[1:, :] * B[:-1, :]).sum() / var
    rj = (B[:, 1:] * B[:, :-1]).sum() / var
    return float(ri), float(rj)


def spectrum(A: np.ndarray) -> dict:
    """2D power spectrum of A - mean. Frequencies in cycles per cell, so
    the grid Nyquist is 0.5. hf_frac = energy with |k| > 0.25 (outer half
    of radial frequency); hf_i / hf_j the same for the 1D marginals."""
    B = A - A.mean()
    P = np.abs(np.fft.fft2(B)) ** 2
    ky = np.fft.fftfreq(A.shape[0])[:, None]
    kx = np.fft.fftfreq(A.shape[1])[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    tot = P.sum()
    edges = np.linspace(0, 0.5, 26)
    idx = np.digitize(k.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < 25)
    cnt = np.bincount(idx[ok], minlength=25)
    pw = np.bincount(idx[ok], weights=P.ravel()[ok], minlength=25)
    radial = pw / np.maximum(cnt, 1) / (tot / P.size)     # 1 = white
    # 1D marginals: mean power at a given |k_i| (over all k_j) and vice versa
    e1 = np.linspace(0, 0.5, 26)
    mi = np.digitize(np.abs(ky.ravel()), e1) - 1
    mj = np.digitize(np.abs(kx.ravel()), e1) - 1
    marg_i = np.array([P[mi == b, :].mean() for b in range(25)]) / (tot / P.size)
    marg_j = np.array([P[:, mj == b].mean() for b in range(25)]) / (tot / P.size)
    return {"hf_frac": float(P[k > 0.25].sum() / tot),
            "k1": 0.5 * (e1[1:] + e1[:-1]), "marg_i": marg_i, "marg_j": marg_j,
            "hf_i": float(P[np.broadcast_to(np.abs(ky) > 0.25, P.shape)].sum() / tot),
            "hf_j": float(P[np.broadcast_to(np.abs(kx) > 0.25, P.shape)].sum() / tot),
            "k": 0.5 * (edges[1:] + edges[:-1]), "radial": radial}


def neighbour_diffs(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|A_ij - A_i+1,j| (along i, y) and |A_ij - A_i,j+1| (along j, x)."""
    return np.abs(np.diff(A, axis=0)), np.abs(np.diff(A, axis=1))


QS = (50, 90, 99, 99.9)


def quantile_row(d: np.ndarray) -> list[float]:
    return [float(np.percentile(d, q)) for q in QS] + [float(d.max())]


def shade(Z: np.ndarray, dx: float, vmin: float, vmax: float,
          cells: bool = False) -> np.ndarray:
    """Hillshade like reliefviz.hillshade (azimuth 315, altitude 45, soft
    blend), fixed vmin/vmax so panels share one colour scale."""
    ls = LightSource(azdeg=315, altdeg=45)
    if cells:
        Z = np.repeat(np.repeat(Z, 3, axis=0), 3, axis=1)
        dx = dx / 3
    return ls.shade(Z, cmap=plt.get_cmap("gray"), vert_exag=1.0, dx=dx, dy=dx,
                    blend_mode="soft", vmin=vmin, vmax=vmax)


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Value below which a fraction q of the total weight lies."""
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, q * cw[-1])])


# -- operators -----------------------------------------------------------------

class Ops:
    """Exact operators at (n, rho), built once, cached."""

    def __init__(self, signal, T):
        self.signal, self.T, self.cache = signal, T, {}

    def get(self, n: int, rho: float = RHO) -> dict:
        key = (n, rho)
        if key not in self.cache:
            t = time.time()
            r = probe(self.signal, self.T, n, rho)
            r["A"] = materialize(r["op"])
            r["secs"] = time.time() - t
            self.cache[key] = r
            print(f"  operator n={n} rho={rho}: N={r['plan'].N} f={r['plan'].f:.0f} Hz "
                  f"drift={r['drift']:.1e} ({r['secs']:.1f} s)", flush=True)
        return self.cache[key]


# -- Q1 spectrum of the loop -----------------------------------------------------

def q1(signal, T, sr, out: Path) -> str:
    freqs, psd = welch(signal, fs=sr, nperseg=8192)
    cum = np.cumsum(psd) / psd.sum()
    rows = []
    marks = []
    for p in PITCHES:
        n = n_for(p)
        f = f_for(n, T)
        nyq = f / 2
        frac = float(np.interp(nyq, freqs, cum))
        rows.append([p, n, round(f), round(nyq), frac])
        marks.append((p, nyq, frac))
    for name, fr in (("source Nyquist", sr / 2),):
        rows.append([name, "", sr, round(fr), float(np.interp(fr, freqs, cum))])
    for fr in (1000, 2000, 4000):
        rows.append([f"{fr} Hz", "", "", fr, float(np.interp(fr, freqs, cum))])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.semilogx(freqs[1:], 10 * np.log10(psd[1:] / psd.max()), color="#1f4e79", lw=1)
    for p, nyq, frac in marks:
        ax.axvline(nyq, color="#c0504d", lw=0.8, ls="--")
        ax.text(nyq, 6 - 7 * (p < 1.6), f"{p:g} mm", rotation=90, va="top",
                ha="right", fontsize=8, color="#c0504d")
    ax.set_xlim(20, sr / 2)
    ax.set_ylim(-80, 8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (dB re peak)")
    ax.text(25, 4, "dashed: f/2 per pitch", fontsize=8, color="#c0504d")
    ax.set_title("Welch PSD of the loop (16 kHz source)")
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.semilogx(freqs[1:], cum[1:], color="#1f4e79", lw=1.5)
    for p, nyq, frac in marks:
        ax.plot([nyq], [frac], "o", color="#c0504d")
    ax.text(0.97, 0.05, "\n".join(f"{p:g} mm: f/2 = {nyq:.0f} Hz, {100 * frac:.1f} %"
                                   for p, nyq, frac in marks),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "ec": "#c0504d"})
    ax.set_xlim(20, sr / 2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("fraction of energy below f")
    ax.set_title("Cumulative energy; dots = Nyquist f/2 of each pitch")
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("Q1: loop spectrum vs the sample rate each pitch implies (660 mm plate, rho 0.95)")
    fig.tight_layout()
    fig.savefig(out / "q1_spectrum.png", dpi=DPI)
    plt.close(fig)

    return (
        "## Q1: spectrum of the loop\n\n"
        f"Loop 324.68 to 338.18 s of song.wav, T = {T:.2f} s, source {sr} Hz. "
        "Building the operator at side n resamples the loop to f = rho n^2 / T "
        "(scipy.signal.resample, a brick-wall low-pass at f/2), so everything "
        "above f/2 is discarded before it reaches the matrix.\n\n"
        "Columns: pitch (mm), n = floor(660 / pitch), f (Hz) = 0.95 n^2 / 13.5, "
        "f/2 = the highest frequency the matrix carries, energy fraction = share "
        "of the loop's Welch power (8192-point segments) below f/2.\n\n"
        + md_table(["pitch", "n", "f Hz", "f/2 Hz", "energy fraction below f/2"], rows)
        + "\n\n![q1](q1_spectrum.png)\n\n"
        "Reading: the loop is bass-heavy (organ, choir, timpani): 71 percent of "
        "the power sits below 1 kHz and 96 percent below 2 kHz, and the PSD falls "
        "about 30 dB between 1 and 6 kHz. Coarse pitches (3 and 2.5 mm, f/2 of "
        "1.7 and 2.5 kHz) keep the body of the sound but lose the brightness; "
        "2.0 mm keeps 99.8 percent of the energy; 1.5 and 1.39 mm are the full "
        "source (f/2 near the source Nyquist of 8 kHz). Energy fraction is not "
        "intelligibility: the missing fractions of a percent at 2 mm are the "
        "consonant-like transients and the shimmer, which a listener notices "
        "first. The same curve matters for Q2: the lower f/2 is relative to "
        "where the music lives, the more of the band the music fills, and the "
        "rougher the operator is along y.\n"
    )


# -- Q2 roughness --------------------------------------------------------------

def q2(ops: Ops, T: float, out: Path) -> str:
    rows_drift, rows_i, rows_j, rows_ac = [], [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(PITCHES)))
    for c, p in zip(colors, PITCHES):
        n = n_for(p)
        r = ops.get(n)
        A, pl = r["A"], r["plan"]
        extra = ""
        if r["drift"] > DRIFT_TOL:
            mf = max_feasible_rho(ops.signal, T, n, DRIFT_TOL)
            extra = f"{mf['plan'].rho:.3f}" if mf else "none"
        rows_drift.append([p, n, pl.N, round(pl.f), f"{r['drift']:.1e}",
                           f"{r['gram_cond']:.1e}", extra or "n/a (stable)"])
        _, m = cell_values(A)
        di, dj = neighbour_diffs(A)
        rows_i.append([p, n] + [x / m for x in quantile_row(di)])
        rows_j.append([p, n] + [x / m for x in quantile_row(dj)])
        ri, rj = lag1(A)
        sp = spectrum(A)
        rows_ac.append([p, n, ri, rj, sp["hf_frac"], sp["hf_i"], sp["hf_j"],
                        float(np.mean(np.abs(A) > m))])
        lab = f"{p:g} mm, n = {n}, f = {pl.f / 1000:.1f} kHz"
        ax.semilogy(sp["k"], sp["radial"], color=c, lw=1.5, label=lab)
        axes[1].semilogy(sp["k1"], sp["marg_i"], color=c, lw=1.5, label=lab)
        axes[2].semilogy(sp["k1"], sp["marg_j"], color=c, lw=1.5, label=lab)
    for a, ttl, xl in ((ax, "radially averaged 2D spectrum", "radial frequency |k|"),
                       (axes[1], "marginal spectrum along i (plate y, overhang direction)",
                        "frequency along i, cycles per cell"),
                       (axes[2], "marginal spectrum along j (plate x)",
                        "frequency along j, cycles per cell")):
        a.axhline(1.0, color="gray", lw=0.8, ls="--")
        a.axvline(0.25, color="#c0504d", lw=0.8, ls=":")
        a.set_xlim(0, 0.5)
        a.set_ylim(3e-3, 30)
        a.text(0.01, 1.15, "white noise = 1", color="gray", fontsize=8)
        a.text(0.255, 4e-3, "outer half", color="#c0504d", fontsize=8)
        a.set_xlabel(xl + " (grid Nyquist = 0.5)")
        a.set_ylabel("power / mean power")
        a.set_title(ttl, fontsize=10)
        a.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Q2: spectrum of the exact operator A at rho 0.95 (A - mean, normalised so white noise = 1)")
    fig.tight_layout()
    fig.savefig(out / "q2_spectrum.png", dpi=DPI)
    plt.close(fig)

    qh = ["pitch", "n", "q50", "q90", "q99", "q99.9", "max"]
    return (
        "## Q2: roughness of the exact operator\n\n"
        "A = W_next G^-1 W^T with W the n x N window matrix (rho = N/n = 0.95). "
        "Drift = relative error of window 1 after one full pass through A "
        "(stable if <= 1e-6); cond G = condition number of the Gram matrix; "
        "the last column is the largest rho that still passes the drift test, "
        "only searched when rho 0.95 fails.\n\n"
        + md_table(["pitch", "n", "N", "f Hz", "drift / pass", "cond G",
                    "largest stable rho"], rows_drift)
        + "\n\nNeighbour difference along i (rows, plate y, the overhang "
        "direction on edge): quantiles of |A_ij - A_i+1,j| / m, where m = "
        "percentile(|A|, 99.5) is the value mapped to full relief. 1.0 means a "
        "full-relief jump between adjacent cells.\n\n"
        + md_table(qh, rows_i)
        + "\n\nNeighbour difference along j (columns, plate x): quantiles of "
        "|A_ij - A_i,j+1| / m.\n\n"
        + md_table(qh, rows_j)
        + "\n\nCorrelation and spectrum. r1_i, r1_j = lag-1 autocorrelation "
        "along i and j (1 = smooth, 0 = white noise, negative = alternating). "
        "hf_2d = fraction of the spectral energy of A - mean with radial "
        "frequency |k| > 0.25 cycles/cell (the outer half of the band up to the "
        "grid Nyquist 0.5); hf_i, hf_j = the same for the 1D frequency along i "
        "and along j alone. White noise gives hf_2d = 0.80 (area of the annulus "
        "within the square), hf_i = hf_j = 0.50. clipped = fraction of entries "
        "with |A| > m (0.005 by construction).\n\n"
        + md_table(["pitch", "n", "r1_i (y)", "r1_j (x)", "hf_2d", "hf_i (y)",
                    "hf_j (x)", "clipped"], rows_ac)
        + "\n\n![q2](q2_spectrum.png)\n\n"
        "Reading: the matrix is strongly anisotropic, and in the direction that "
        "matters. Write A = W_next G^-1 W^T. Column j of A is W_next c_j, a "
        "linear combination of the next windows, so along i (plate y) the matrix "
        "inherits the spectrum of the music resampled to f: at n = 474 "
        "(f = 15.8 kHz) the music lives far below the grid Nyquist and A is smooth "
        "along y (r1_i = 0.90, only 0.3 percent of the energy in the outer half); "
        "at n = 220 (f = 3.4 kHz) the music fills its band and y is rough too "
        "(r1_i = 0.18). Row i of A is a combination of the rows of G^-1 W^T, the "
        "dual basis of the windows (d_k . w_l = delta_kl). With W = U S V^T the "
        "dual basis is V S^-1 U^T: it weights each left singular pattern u_r by "
        "1 / s_r, so the WEAKEST components of the windows dominate. Oversampled "
        "music has almost no energy near f/2, so those weakest components are "
        "the near-Nyquist patterns, and along j (plate x) the matrix is not white "
        "but alternating: r1_j = -0.28 at n = 220 and -0.78 at n = 474, with 98 "
        "percent of the energy in the outer half. The higher f and the worse the "
        "Gram conditioning, the stronger the effect: along x the operator is a "
        "picture of the inverse of the music, dominated by its noise floor. On "
        "edge, y is the printer's Z and the y slope is the overhang, so the smooth "
        "direction is the one the printer cares about; the Nyquist chatter along "
        "x becomes a 2-cell (4 mm at 2 mm pitch) side-to-side ripple inside each "
        "layer, which FDM prints without trouble. The neighbour-difference "
        "quantiles say the same in relief units: along y the median jump is 0.17 "
        "of the full relief at n = 330 (0.09 at n = 474), along x it is 0.4 at "
        "every n, and the 99.9th percentile along x exceeds the whole clip range "
        "(2 m peak to peak): after clipping, adjacent cells sit at the lowest "
        "and the highest level.\n"
    )


# -- Q3 interpolation ---------------------------------------------------------------

def pick_patch(A: np.ndarray, size: int) -> tuple[int, int]:
    di, dj = neighbour_diffs(A)
    D = np.zeros_like(A)
    D[:-1, :] = np.maximum(D[:-1, :], di)
    D[:, :-1] = np.maximum(D[:, :-1], dj)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    i0 = int(np.clip(i - size // 2, 0, A.shape[0] - size))
    j0 = int(np.clip(j - size // 2, 0, A.shape[1] - size))
    return i0, j0


def q3(ops: Ops, out: Path, subdiv: int = 6, size: int = 24,
       relief_mm: float = 4.0) -> str:
    pitch = 2.0
    n = n_for(pitch)
    A = ops.get(n)["A"]
    v, _ = cell_values(A)
    i0, j0 = pick_patch(A, size)
    patch = v[i0:i0 + size, j0:j0 + size]
    rng = patch.max() - patch.min()
    results = {name: fn(patch, subdiv) for name, fn in INTERP.items()}
    h_cell = pitch / subdiv
    rows = []
    for name, Z in results.items():
        gy = np.diff(Z, axis=0) / h_cell * pitch       # range per pitch
        gx = np.diff(Z, axis=1) / h_cell * pitch
        rows.append([name, (Z.max() - patch.max()) / rng, (patch.min() - Z.min()) / rng,
                     np.abs(gy).max() / rng, np.abs(gx).max() / rng,
                     float(np.percentile(np.hypot(gy[:, :-1], gx[:-1, :]), 99) / rng)])
    # hillshades at a common colour scale, heights in mm at relief_mm peak to peak
    H = {k: 0.5 * relief_mm * Z for k, Z in results.items()}
    vmin = min(Z.min() for Z in H.values())
    vmax = max(Z.max() for Z in H.values())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (name, Z) in zip(axes, H.items()):
        rgb = shade(Z, h_cell, vmin, vmax)
        ax.imshow(rgb, origin="lower", extent=(0, size * pitch, 0, size * pitch),
                  interpolation="nearest")
        ax.set_title(f"{name}: peak {Z.max():.2f} mm")
        ax.set_xlabel("x mm")
        if ax is axes[0]:
            ax.set_ylabel("y mm")
        else:
            ax.set_yticks([])
    fig.suptitle(f"Q3: 24 x 24 cells of A (n = {n}, pitch {pitch} mm, rows {i0}..{i0 + size - 1}, "
                 f"cols {j0}..{j0 + size - 1}) interpolated at subdiv {subdiv}, "
                 f"relief {relief_mm} mm peak to peak, one colour scale")
    fig.tight_layout()
    fig.savefig(out / "q3_interpolation.png", dpi=DPI)
    plt.close(fig)
    return (
        "## Q3: interpolation methods on a spiky patch\n\n"
        f"Patch: the 24 x 24 cells of the n = {n} (pitch 2 mm) matrix around the "
        f"largest neighbour difference, rows {i0} to {i0 + size - 1}, columns {j0} "
        f"to {j0 + size - 1}; v = clip(A/m, -1, 1), range of v in the patch "
        f"{rng:.3f}. Fine grid: {subdiv} nodes per cell, nodes at k p / {subdiv}, "
        "cell centres at (i + 0.5) p, edges extended (mode nearest). nearest / "
        "bilinear / bicubic = scipy.ndimage.map_coordinates order 0 / 1 / 3 "
        "(order 3 is the interpolating cubic B-spline: passes through the data, "
        "overshoots next to jumps). fourier = trigonometric interpolant of the "
        "even (mirror) extension to 48 x 48: strictly band-limited, rings "
        "(Gibbs) next to spikes.\n\n"
        "Columns: overshoot = (max interp - max data) / range, undershoot = "
        "(min data - min interp) / range (how far the surface leaves the data "
        "envelope, as a fraction of the patch's height range); max |dh/dy|, "
        "max |dh/dx| = steepest node-to-node slope on the fine grid, in units of "
        "range per pitch (multiply by relief_mm / pitch_mm for the physical slope "
        "at a given relief, e.g. 2 at 4 mm relief on 2 mm pitch); q99 |grad| = "
        "99th percentile of the gradient magnitude, same unit.\n\n"
        + md_table(["method", "overshoot", "undershoot", "max abs(dh/dy)", "max abs(dh/dx)",
                    "q99 abs(grad)"], rows)
        + "\n\n![q3](q3_interpolation.png)\n\n"
        "Reading: nearest is the old stepped relief: zero overshoot and formally "
        "infinite slope (here the finite node spacing makes it range / (pitch / "
        f"{subdiv}) = {subdiv} range per pitch, a vertical wall; the y value is "
        "smaller only because the largest jump in this patch is along x). Bilinear "
        "has no overshoot and slope at most 1 range per pitch, but the surface is "
        "a tent field with creases along cell rows and columns, visible in the "
        "hillshade as a woven texture. Bicubic overshoots the data by 14 percent "
        "of the range next to the spikes (peak 2.55 mm where the data stop at "
        "2.00 mm at 4 mm relief) and is 30 percent steeper than bilinear, but it "
        "is C2 and the hillshade shows smooth ridges with no creases: this is the "
        "wave look the plate is after. Fourier rings: 28 percent overshoot, 50 "
        "percent undershoot, the steepest slopes of the smooth methods, and "
        "halos around every spike; it is the worst choice for a spiky field "
        "even though it is the only method that adds no frequency above the "
        "grid Nyquist. All four show the Q2 anisotropy directly: ridges run "
        "along y, the texture period along x is about 2 cells.\n"
    )


# -- Q4 amplitude vs overhang --------------------------------------------------------

def slopes_at_unit_relief(v: np.ndarray, pitch: float, subdiv: int):
    """Slopes of the fine quads at relief 1 mm peak to peak: h = 0.5 v mm.
    gy, gx at quad centres (mean of the two opposite edges), unit mm/mm."""
    Z = 0.5 * interpolate(v, subdiv, "cubic")
    h = pitch / subdiv
    dy = np.diff(Z, axis=0) / h
    dx = np.diff(Z, axis=1) / h
    gy = 0.5 * (dy[:, 1:] + dy[:, :-1])
    gx = 0.5 * (dx[1:, :] + dx[:-1, :])
    return gy.ravel(), gx.ravel()


def relief_for(Rc: np.ndarray, g2: np.ndarray, q: float) -> tuple[float, float]:
    """Relief R at which an area fraction q of the surface has Rc >= R.

    Rc = the relief at which each fine quad crosses the angle limit (inf if
    never). Area weights sqrt(1 + R^2 g2) depend on R, so: unweighted
    quantile first, then one re-weighting. Returns (weighted, unweighted)."""
    R0 = float(np.quantile(Rc, 1.0 - q))
    if not np.isfinite(R0):
        return np.inf, np.inf
    w = np.sqrt(1.0 + R0 * R0 * g2)
    return weighted_quantile(Rc, w, 1.0 - q), R0


def area_frac_over(Rc: np.ndarray, g2: np.ndarray, R: float) -> float:
    """Area fraction of quads past their limit at relief R."""
    w = np.sqrt(1.0 + R * R * g2)
    return float((w * (Rc < R)).sum() / w.sum())


def q4(ops: Ops, out: Path, subdiv: int = 3) -> str:
    cases = ((2.0, n_for(2.0)), (1.5, n_for(1.5)))
    sigmas = (0.0, 0.5, 1.0)
    qs = (0.90, 0.95, 0.99, 0.999)
    alphas = (45.0, 55.0)
    rows_y, rows_t, rows_x, curves = [], [], [], {}
    worst_weighting = 0.0
    for pitch, n in cases:
        A = ops.get(n)["A"]
        for s in sigmas:
            v, _ = cell_values(A, sigma_cells=s)
            v0, _ = cell_values(A)
            p2p = float(v.max() - v.min())
            rms_ratio = float(np.sqrt((v * v).mean() / (v0 * v0).mean()))
            gy, gx = slopes_at_unit_relief(v, pitch, subdiv)
            g2 = gy ** 2 + gx ** 2
            for q in qs:
                ry = [pitch, s, p2p, rms_ratio, q]
                rt = [pitch, s, q]
                rx = [pitch, s, q]
                for a in alphas:
                    t = np.tan(np.radians(a))
                    # spec rule: |dh/dy| > tan(alpha), both signs, y slope only
                    Rc_spec = t / np.maximum(np.abs(gy), 1e-300)
                    R_spec, R_unw = relief_for(Rc_spec, g2, q)
                    worst_weighting = max(worst_weighting, abs(R_spec / R_unw - 1))
                    # true facet overhang: tan(theta) = dh/dy / sqrt(1 + (dh/dx)^2),
                    # downward-facing (dh/dy > 0) only; crossing at
                    # R^2 (gy^2 - t^2 gx^2) = t^2; never if gy <= t |gx|
                    d = gy ** 2 - t * t * gx ** 2
                    ever = (gy > 0) & (d > 0)
                    Rc_true = np.where(ever, t / np.sqrt(np.maximum(d, 1e-300)), np.inf)
                    R_true, _ = relief_for(Rc_true, g2, q)
                    # area fraction that can ever exceed alpha (R -> inf, area
                    # weight -> |grad h|)
                    wg = np.sqrt(g2)
                    plateau = 100 * float((wg * ever).sum() / wg.sum())
                    ry += [R_spec, R_spec * p2p / 2]
                    rt += ["never" if np.isinf(R_true) else R_true, plateau]
                    Rc_x = t / np.maximum(np.abs(gx), 1e-300)
                    rx.append(relief_for(Rc_x, g2, q)[0])
                rows_y.append(ry)
                rows_t.append(rt)
                rows_x.append(rx)
            t45 = np.tan(np.radians(45))
            Rs = np.geomspace(1, 200, 50)
            Rc_spec = t45 / np.maximum(np.abs(gy), 1e-300)
            d = gy ** 2 - t45 * t45 * gx ** 2
            Rc_true = np.where((gy > 0) & (d > 0), t45 / np.sqrt(np.maximum(d, 1e-300)), np.inf)

            curves[(pitch, s)] = (Rs, [area_frac_over(Rc_spec, g2, R) for R in Rs],
                                  [area_frac_over(Rc_true, g2, R) for R in Rs])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    colors = {0.0: "#1f4e79", 0.5: "#2e8b57", 1.0: "#c0504d"}
    for ax, (pitch, n) in zip(axes, cases):
        for s in sigmas:
            Rs, fs, ft = curves[(pitch, s)]
            ax.loglog(Rs, np.maximum(fs, 1e-5), "-", color=colors[s], lw=1.6,
                      label=f"sigma {s:g} cells, spec rule |dh/dy| > tan 45")
            ax.loglog(Rs, np.maximum(ft, 1e-5), "--", color=colors[s], lw=1.3,
                      label=f"sigma {s:g} cells, true overhang > 45 deg")
        for qq, lab in ((0.10, "90 %"), (0.05, "95 %"), (0.01, "99 %"), (0.001, "99.9 %")):
            ax.axhline(qq, color="gray", lw=0.6, ls=":")
            ax.text(1.05, qq * 1.15, lab, fontsize=8, color="gray")
        ax.set_xlim(1, 200)
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("nominal relief R (mm peak to peak of the clip range)")
        ax.set_title(f"pitch {pitch} mm, n = {n}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("area fraction over the 45 deg limit")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Q4: surface area steeper than 45 deg in y vs relief (bicubic, subdiv {subdiv}); "
                 "solid = spec rule, dashed = true facet overhang")
    fig.tight_layout()
    fig.savefig(out / "q4_amplitude.png", dpi=DPI)
    plt.close(fig)
    hy = ["pitch", "sigma", "v p2p", "v rms ratio", "q",
          "R45 spec", "h45 printed", "R55 spec", "h55 printed"]
    ht = ["pitch", "sigma", "q", "R45 true", "ever > 45 (% area)", "R55 true", "ever > 55 (% area)"]
    hx = ["pitch", "sigma", "q", "R45 x", "R55 x"]
    return (
        "## Q4: relief amplitude allowed by the overhang limit\n\n"
        f"Surface: v = clip(A/m, -1, 1), optional Gaussian blur of sigma cells, "
        f"bicubic to {subdiv} nodes per cell (the plate default), h = R/2 * v with "
        "R the nominal relief (peak to peak of the clip range). Slopes are "
        "measured on the fine quads (mean of the two opposite edge differences), "
        "so they are the slopes of the printed facets, not of the ideal cubic. "
        "Interpolation is linear in the heights, so every slope scales with R, "
        "and each quad has a relief Rc at which it crosses the angle limit; the "
        "relief that keeps an AREA fraction q of the surface inside the limit is "
        "the (1 - q) quantile of Rc, area-weighted by sqrt(1 + |grad h|^2) at "
        "that relief (one re-weighting from the unweighted quantile; the "
        f"weighting moved R by at most {100 * worst_weighting:.1f} percent).\n\n"
        "Two angle rules. 'spec': |dh/dy| > tan(alpha), both signs, y slope "
        "only (what surface.fit_relief computes, linear in R). 'true': the "
        "facet actually overhangs in the print. On edge the face normal "
        "(-dh/dx, -dh/dy, 1) becomes (-dh/dx, -1, -dh/dy), so a facet hangs "
        "only when dh/dy > 0 (the other sign leans back like a roof), and the "
        "layer-to-layer offset perpendicular to the perimeter line is "
        "tan(theta) = (dh/dy) / sqrt(1 + (dh/dx)^2): a facet that is also "
        "steep in x has its perimeter running obliquely and overhangs less. "
        "Rc for 'true' solves R^2 ((dh/dy)^2 - tan^2(alpha) (dh/dx)^2) = "
        "tan^2(alpha) at unit relief, infinite when the x slope keeps the facet "
        "inside the limit at every R.\n\n"
        "Columns: v p2p = peak-to-peak of v after the blur (2 means the full "
        "clip range is used; the blur shrinks it, so the printed height range is "
        "R * p2p / 2, given as 'h printed'); v rms ratio = rms(v blurred) / "
        "rms(v) = how much of the matrix survives the blur; R45 / R55 = nominal "
        "relief in mm at the 45 and 55 deg limits.\n\n"
        "Spec rule along y (the overhang direction on edge):\n\n"
        + md_table(hy, rows_y)
        + "\n\nTrue facet rule along y. 'ever' = the area fraction that can "
        "exceed the limit at ANY relief: a facet with dh/dy <= tan(alpha) "
        "abs(dh/dx) stays inside the limit however large R is (as R grows it "
        "turns into a sideways-facing vertical wall), so when 100 (1 - q) is "
        "larger than 'ever' the answer is 'never'. Nominal R, multiply by "
        "p2p / 2 for the printed height as above.\n\n"
        + md_table(ht, rows_t)
        + "\n\nSpec rule along x on abs(dh/dx) (not an overhang on edge; shows the "
        "anisotropy):\n\n"
        + md_table(hx, rows_x)
        + "\n\n![q4](q4_amplitude.png)\n\n"
        "Reading: the relief budget is set by the tail of the slope "
        "distribution, and the tail is the spikes. At 2 mm pitch without blur "
        "the spec rule allows 6.3 mm of relief with 99 percent of the area "
        "under 45 degrees and 4.7 mm at 99.9 percent (8.9 and 6.7 mm at 55 "
        "degrees); at 1.5 mm pitch 7.9 and 5.7 mm, MORE than at 2 mm, because "
        "the n = 440 matrix is smoother along y (Q2: the music sits lower in "
        "its band) and that outweighs the shorter cell. The true facet rule "
        "changes the picture in two ways. In the tail it is 1.2 to 1.8x more "
        "generous (8.8 and 5.6 mm at 2 mm pitch; 14.5 and 7.5 mm at 1.5 mm): "
        "only half of the slopes hang, and the large x slopes turn the "
        "perimeter oblique. In the bulk it saturates: at 2 mm pitch only 7 "
        "percent of the area can ever exceed 45 degrees (3 percent at 1.5 mm), "
        "because for the rest the x slope dominates and the facet becomes a "
        "sideways wall, so '90 percent under 45' is satisfied at every "
        "relief. The x table shows the anisotropy: the same spec rule applied "
        "to x allows only 2.1 mm at 2 mm pitch and 1.5 mm at 1.5 mm pitch, 3 to "
        "5 times less than y; printed flat (x and y both overhang-free but "
        "steep in x) the relief would be limited by x. Blurring buys little: "
        "sigma 0.5 cells raises the printed height at 99 percent from 6.3 to "
        "8.1 mm (1.3x) but keeps only 63 percent of the matrix rms, and sigma 1 "
        "doubles the height while keeping 20 percent (the blur removes the "
        "near-Nyquist content that IS the x structure); the huge nominal R in "
        "those rows is a scaling artefact (v no longer spans the clip range) "
        "and the 'h printed' column is the real height. Recommendation: no "
        "blur, bicubic, 4 to 6 mm of relief at 2 mm pitch keeps 99 to 99.9 "
        "percent of the area inside the spec rule at 45 degrees and more "
        "inside the true rule. The surface module should report the facet "
        "rule as well, and renormalise v after any blur so R stays the printed "
        "height.\n"
    )


# -- Q5 FDM table ---------------------------------------------------------------------

def q5(out: Path) -> str:
    layers = (0.10, 0.15, 0.20)
    lw = 0.45
    alphas = list(range(30, 71, 5))
    rows = []
    for a in alphas:
        r = [a]
        for L in layers:
            u = L * np.tan(np.radians(a)) / lw
            r.append(f"{u:.2f}" + (" ok" if u <= 0.5 else (" marginal" if u <= 0.6 else " sag")))
        rows.append(r)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    aa = np.linspace(20, 75, 200)
    for L, c in zip(layers, ("#1f4e79", "#2e8b57", "#c0504d")):
        ax.plot(aa, L * np.tan(np.radians(aa)) / lw, color=c, lw=1.5, label=f"layer {L} mm")
        a_half = np.degrees(np.arctan(0.5 * lw / L))
        ax.plot([a_half], [0.5], "o", color=c)
        ax.annotate(f"{a_half:.0f} deg", (a_half, 0.5), textcoords="offset points",
                    xytext=(4, -14), fontsize=8, color=c)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.text(21, 0.52, "u = 0.5: half the line unsupported (rule of thumb)", fontsize=8, color="gray")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.text(21, 1.02, "u = 1: nothing under the line (bridge)", fontsize=8, color="gray")
    ax.set_xlim(20, 75)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("overhang angle from vertical (deg)")
    ax.set_ylabel(f"unsupported fraction u = layer tan(alpha) / {lw} mm line")
    ax.set_title("Q5: FDM overhang geometry (line width 0.45 mm)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "q5_fdm_overhang.png", dpi=DPI)
    plt.close(fig)
    return (
        "## Q5: FDM overhang geometry\n\n"
        "Each layer of a wall tilted alpha from vertical is shifted outward by "
        "layer_height * tan(alpha) relative to the layer below; the outer line "
        "of width w then has an unsupported fraction u = layer * tan(alpha) / w "
        "hanging in air (u = 1: the whole line is a bridge). Table: u for a 0.45 "
        "mm line (0.4 mm nozzle), labels ok (u <= 0.5), marginal (0.5 < u <= "
        "0.6), sag (u > 0.6).\n\n"
        + md_table(["alpha (deg)"] + [f"layer {L} mm" for L in layers], rows)
        + "\n\n![q5](q5_fdm_overhang.png)\n\n"
        "What is physics and what is rule of thumb: the formula u = layer "
        "tan(alpha) / w is plain geometry (3dx.info states it as 'each layer "
        "steps outward by layer height x tan(angle)' and tabulates 1 - u for "
        "a 0.45 mm line, matching this table: 78 % supported at 45 deg and "
        "0.10 mm, 56 % at 0.20 mm, 0 % at 60 deg and 0.30 mm). The threshold "
        "u <= 0.5 is the rule of thumb behind the 45-degree rule: at 0.2 mm "
        "layers and 0.4 mm lines, 45 deg is exactly u = 0.5, 'each new layer "
        "still sits on about half of the layer below it' (Snapmaker). Where the "
        "real limit lies depends on cooling, speed, material and the extrusion "
        "rounding into a bead: online calculators let the overlap fraction vary "
        "between 0.4 and 0.6 (convertools), and vendor guides say well-cooled "
        "PLA at low layer height prints past 55 deg while hot PETG or ABS sags "
        "below 45 (Snapmaker). So for this plate: at 0.10 mm layers the "
        "geometric u = 0.5 point is 66 deg, at 0.15 mm 56 deg, at 0.20 mm 48 "
        "deg. The spec's 45 deg limit is comfortable at any of these layer "
        "heights; 55 deg is fine at 0.10 and 0.15 mm layers and marginal at "
        "0.20 mm. Two caveats the formula does not capture: (1) the relief is "
        "a long thin wall on edge, so the overhanging band is one or two "
        "perimeter lines wide with nothing behind it to lean on, which argues "
        "for the conservative end; (2) the back-leaning faces (dh/dy < 0) are "
        "upward-facing slopes that print as stairs with a step of layer / "
        "tan(alpha) in width, so they show layer lines but do not fail.\n\n"
        "Sources:\n"
        "- https://3dx.info/beyond-basics-how-layer-height-and-line-width-impact-overhang-performance-in-3d-printing/ "
        "(formula and the support-percentage table for a 0.45 mm line)\n"
        "- https://www.snapmaker.com/blog/45-degree-rule-3d-printing/ "
        "(the half-overlap rule of thumb and its material and cooling caveats)\n"
        "- https://www.convertools.net/3d-printing/max-overhang-angle/ "
        "(calculator with the overlap fraction as a free parameter, 0.4 to 0.6)\n"
    )


# -- Q6 null space -------------------------------------------------------------------

def smooth_metrics(A: np.ndarray, m: float) -> dict:
    ri, rj = lag1(A)
    sp = spectrum(A)
    di, dj = neighbour_diffs(A)
    return {"r1_i": ri, "r1_j": rj, "hf": sp["hf_frac"], "hf_i": sp["hf_i"],
            "hf_j": sp["hf_j"], "max_di": float(di.max() / m),
            "max_dj": float(dj.max() / m), "q99_di": float(np.percentile(di, 99) / m),
            "q99_dj": float(np.percentile(dj, 99) / m)}


def diff_matrix(n: int) -> np.ndarray:
    return np.eye(n - 1, n, k=1) - np.eye(n - 1, n)


def dirichlet(A: np.ndarray) -> tuple[float, float]:
    """Sum of squared neighbour differences along i and along j."""
    return float((np.diff(A, axis=0) ** 2).sum()), float((np.diff(A, axis=1) ** 2).sum())


def alternate(A0: np.ndarray, W: np.ndarray, sigma: float, iters: int):
    """Low-pass A, then the min-norm correction back onto {A : A W = W_next}.
    The correction A + (W_next - A W) G^-1 W^T is the orthogonal (Frobenius)
    projection onto that affine set; A - A0 is then automatically of the
    form Z P_perp (its rows are orthogonal to span W)."""
    Wn = np.roll(W, -1, axis=1)
    Ginv = np.linalg.pinv(W.T @ W, rcond=1e-10)
    A = A0.copy()
    hist = []
    for _ in range(iters):
        A = gaussian_filter(A, sigma, mode="nearest")
        A = A + (Wn - A @ W) @ Ginv @ W.T
        hist.append(lag1(A) + tuple(dirichlet(A)))
    return A, hist


def x_only_optimum(A0: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Minimise only |A D^T|^2 (x-roughness): per row, closed form."""
    n, N = W.shape
    Q = np.linalg.svd(W, full_matrices=True)[0][:, N:]
    K = diff_matrix(n).T @ diff_matrix(n)
    Y = -(A0 @ K @ Q) @ np.linalg.inv(Q.T @ K @ Q)
    return A0 + Y @ Q.T


def dirichlet_optimum(A0: np.ndarray, W: np.ndarray) -> np.ndarray:
    """The exact operator with the least Dirichlet energy |D A|^2 + |A D^T|^2.

    A = A0 + Y Q^T with Q an orthonormal basis of span(W)^perp (so A W =
    W_next for every Y). Stationarity gives the Sylvester equation
    K Y + Y (Q^T K Q) = -(K A0 Q + A0 K Q), K = D^T D. Since A0 Q = 0 (rows
    of A0 lie in span W) the first term vanishes: the y-roughness |D A|^2 =
    |D A0|^2 + |D Y|^2 can only grow, A0 is already the smoothest exact
    operator along i; only the x term can be reduced."""
    from scipy.linalg import solve_sylvester

    n, N = W.shape
    U = np.linalg.svd(W, full_matrices=True)[0]
    Q = U[:, N:]
    K = diff_matrix(n).T @ diff_matrix(n)
    C = -(K @ A0 @ Q + A0 @ K @ Q)
    Y = solve_sylvester(K + 1e-9 * np.eye(n), Q.T @ K @ Q, C)
    return A0 + Y @ Q.T


def q6(ops: Ops, out: Path, n: int = 150, iters: int = 20, sigma: float = 1.0,
       patch: int = 40, pitch: float = 2.0, relief_mm: float = 4.0) -> str:
    rows, panels = [], []
    for rho in (0.5, 0.95):
        r = ops.get(n, rho)
        A0, W, pl = r["A"], r["W"], r["plan"]
        Wn = np.roll(W, -1, axis=1)
        m = float(np.percentile(np.abs(A0), CLIP_PCT))
        A1, hist = alternate(A0, W, sigma, iters)
        A2 = dirichlet_optimum(A0, W)
        A3 = x_only_optimum(A0, W)
        Ey0, Ex0 = dirichlet(A0)
        for lab, A in (("A0", A0), (f"alternation x{iters}", A1),
                       ("Dirichlet optimum", A2), ("x-only optimum", A3)):
            s = smooth_metrics(A, m)
            Ey, Ex = dirichlet(A)
            rows.append([rho, pl.N, (n - pl.N) / n, lab, Ey / Ey0, Ex / Ex0,
                         s["r1_i"], s["r1_j"], s["hf"], s["q99_di"], s["q99_dj"],
                         s["max_di"], s["max_dj"],
                         f"{loop_degradation(A, W, 1)[0]:.1e}",
                         f"{np.abs(A @ W - Wn).max() / np.abs(Wn).max():.1e}"])
        panels.append((rho, pl.N, A0, A2, hist, m, (Ey0, Ex0)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    i0 = j0 = (n - patch) // 2
    for k, (rho, N, A0, A2, hist, m, (Ey0, Ex0)) in enumerate(panels):
        Hs = [0.5 * relief_mm * np.clip(A[i0:i0 + patch, j0:j0 + patch] / m, -1, 1)
              for A in (A0, A2)]
        vmin, vmax = -0.5 * relief_mm, 0.5 * relief_mm
        for c, (lab, H) in enumerate(zip(("A0 (min-norm exact)", "Dirichlet optimum (exact)"), Hs)):
            ax = axes[k, c]
            ax.imshow(shade(H, pitch, vmin, vmax, cells=True), origin="lower",
                      extent=(0, patch * pitch, 0, patch * pitch), interpolation="nearest")
            ax.set_title(f"rho {rho} (N = {N}): {lab}")
            ax.set_xlabel("x mm")
            ax.set_ylabel("y mm")
        ax = axes[k, 2]
        h = np.array(hist)
        it = np.arange(1, iters + 1)
        ax.plot(it, h[:, 2] / Ey0, "-o", ms=3, color="#1f4e79", label="alternation: energy along i (y)")
        ax.plot(it, h[:, 3] / Ex0, "-s", ms=3, color="#c0504d", label="alternation: energy along j (x)")
        Ey2, Ex2 = dirichlet(A2)
        ax.axhline(Ey2 / Ey0, color="#1f4e79", ls="--", lw=1, label="Dirichlet optimum, i")
        ax.axhline(Ex2 / Ex0, color="#c0504d", ls="--", lw=1, label="Dirichlet optimum, j")
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.set_ylim(0.5, 1.3)
        ax.set_xlabel("iteration")
        ax.set_ylabel("sum of squared neighbour differences / that of A0")
        ax.set_title(f"rho {rho}: free fraction (n - N)/n = {(n - N) / n:.2f}")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Q6: null-space smoothing, n = {n}; {patch} x {patch} centre patch at "
                 f"{relief_mm} mm relief, one colour scale per row; right: roughness energy per iteration")
    fig.tight_layout()
    fig.savefig(out / "q6_nullspace.png", dpi=DPI)
    plt.close(fig)
    hdr = ["rho", "N", "free frac", "matrix", "E_y / E_y(A0)", "E_x / E_x(A0)", "r1_i", "r1_j",
           "hf_2d", "q99 di/m", "q99 dj/m", "max di/m", "max dj/m", "drift / pass",
           "max abs(AW - Wn) rel"]
    return (
        "## Q6: null-space smoothing\n\n"
        f"n = {n}. Every A = A0 + Z P_perp plays the loop exactly; the set of such "
        "A is an affine space of dimension n (n - N) ('free frac' = (n - N)/n of "
        "all entries). Two ways to use it. (a) Alternation as in the spec: "
        f"A <- gaussian_filter(A, sigma = {sigma:g} cell); A <- A + (W_next - A W) "
        f"G^-1 W^T (orthogonal projection back onto the exact set), {iters} "
        "times. (b) The exact answer to 'smoothest exact operator' in the "
        "Dirichlet sense: minimise E_y + E_x = sum of squared neighbour "
        "differences along i plus along j over A = A0 + Y Q^T (Q = orthonormal "
        "basis of span(W)^perp), a Sylvester equation, solved in closed form; "
        "and the same for E_x alone ('x-only optimum', per-row least squares).\n\n"
        "Columns: E_y, E_x = Dirichlet energies relative to A0; r1 = lag-1 "
        "autocorrelation along i and j; hf_2d = energy fraction at |k| > 0.25; "
        "di, dj = neighbour differences along i and j divided by m = "
        "percentile(|A0|, 99.5) of the original matrix; drift / pass = "
        "loop_degradation after one pass; last column = max |A W - W_next| / "
        "max |W_next| (exactness).\n\n"
        + md_table(hdr, rows)
        + "\n\n![q6](q6_nullspace.png)\n\n"
        "Reading: the null space cannot help the overhang direction at all, "
        "and this is a theorem, not a measurement. The rows of A0 = W_next G^-1 "
        "W^T lie in span(W), so A0 Q = 0, and for any correction Y Q^T the "
        "y-roughness splits as |D A|^2 = |D A0|^2 + |D Y|^2: the min-norm "
        "operator is already the smoothest exact operator along i, and "
        "anything added makes it rougher in y (the table shows E_y rising "
        "to 1.05 at the optimum). What the free part can do is cancel part of "
        "the x-roughness of each row with vectors orthogonal to span(W): at "
        "rho 0.5 the optimum removes 18 percent of E_x (6 percent of the "
        "total), at rho 0.95 7 percent (2 percent of the total), because the "
        "free subspace has only n - N dimensions per row out of n, and the "
        "x-roughness of A0 (the dual-basis Nyquist chatter of Q2) lives mostly "
        "inside span(W) where nothing can touch it. Minimising E_x alone "
        "removes 43 percent of E_x at rho 0.5 but makes E_y eleven times "
        "larger: the free part can only move roughness from x into y, the "
        "wrong direction for an on-edge print. The Gaussian alternation is a "
        "heuristic that converges to a fixed point, not to the optimum; it "
        "gets half of the optimum's E_x reduction in a few iterations and then "
        "stalls, and the hillshades of A0 and the optimum are indistinguishable "
        "by eye (r1_j moves from 0 to 0.29 at rho 0.5, still nowhere near a "
        "smooth field). "
        "Both solutions still play exactly (drift unchanged, residual at "
        "machine precision). Conclusion: an exact operator is as rough as its "
        "data make it; smoothing for the plate has to come from the display "
        "(interpolation, Q3; amplitude choice, Q4), or from lowering rho, "
        "which buys x-smoothness slowly and y-smoothness not at all (y depends "
        "on f relative to the music, Q2).\n"
    )


# -- main ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default="data/audio_3d/song.wav")
    ap.add_argument("--start", type=float, default=324.68)
    ap.add_argument("--end", type=float, default=338.18)
    ap.add_argument("--out", default="runs/exp_smooth")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["q1", "q2", "q3", "q4", "q5", "q6"])
    ap.add_argument("--subdiv", type=int, default=3, help="fine nodes per cell for Q4")
    ap.add_argument("--q6-iters", type=int, default=20)
    args = ap.parse_args()
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    signal, T, sr = load_loop(ROOT / args.audio, args.start, args.end)
    ops = Ops(signal, T)
    todo = args.only or ["q1", "q2", "q3", "q4", "q5", "q6"]
    sections = []
    for q in todo:
        t = time.time()
        print(f"== {q}", flush=True)
        if q == "q1":
            sections.append(q1(signal, T, sr, out))
        elif q == "q2":
            sections.append(q2(ops, T, out))
        elif q == "q3":
            sections.append(q3(ops, out))
        elif q == "q4":
            sections.append(q4(ops, out, subdiv=args.subdiv))
        elif q == "q5":
            sections.append(q5(out))
        elif q == "q6":
            sections.append(q6(ops, out, iters=args.q6_iters))
        print(f"   {time.time() - t:.1f} s", flush=True)
    head = (
        "# exp_smooth_surface: design numbers for the smooth on-edge plate\n\n"
        f"Loop {args.start} to {args.end} s of {Path(args.audio).name} (T = {T:.2f} s, "
        f"{sr} Hz). Plate side {SIDE_MM:g} mm, rho = {RHO}, clip percentile "
        f"{CLIP_PCT}. Matrix convention: A_ij, i = row = plate y (up; the printer's "
        "Z when the strip stands on edge), j = column = plate x. Reproduce with "
        f"`uv run python scripts/exp_smooth_surface.py --out {args.out}`.\n\n"
    )
    path = out / "RESULTS.md"
    if args.only and path.exists():
        # merge: replace the rerun sections, keep the others
        old = path.read_text()
        parts = old.split("\n## ")
        keep = {("## " + p).split("\n")[0]: "## " + p for p in parts[1:]}
        for s in sections:
            keep[s.split("\n")[0]] = s.rstrip("\n") + "\n"
        body = "\n".join(keep[k] for k in sorted(keep))
        path.write_text(head + body)
    else:
        path.write_text(head + "\n".join(sections))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
