"""The matrix as a smooth height field on a fine node grid.

Cell (i, j) of the n x m matrix is centred at ((j + 0.5) p, (i + 0.5) p),
row 0 at the bottom (y = 0). Entries are clipped to [-1, 1] at a robust
percentile, optionally Gaussian-smoothed, and interpolated to the nodes
k p / subdiv (k = 0 .. n subdiv) of a grid spanning the whole plate.
Height h = t - relief / 2 + (relief / 2) v puts zero entries at mid depth
and the surface in [t - relief, t], except that cubic and fourier
overshoot the data. The printed range is therefore measured after
interpolation: with `max_range_mm` the relief is scaled down so the
printed range fits (the thickness constraint applies to what is printed,
not to the nominal relief), and the surface is always shifted so its
highest point sits exactly at t.

Interpolation is the plug-in point (`interpolate`): nearest (stepped),
linear, cubic (default) and fourier (mirror-padded zero-pad FFT). All of
them are linear in the heights, so every slope scales with relief_mm and
`fit_relief` needs one evaluation.

Overhang on edge (plate y = printer Z): a facet with gradient (hx, hy)
has print-frame normal (-hx, -1, -hy), so it hangs only when hy > 0 and
its angle from vertical is tan(theta) = hy / sqrt(1 + hx^2) (the "facet"
rule, the default). The simpler "y" rule |hy| <= tan(alpha) ignores the
sign and the x slope and is kept for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

METHODS = ("nearest", "linear", "cubic", "fourier")
_ORDER = {"nearest": 0, "linear": 1, "cubic": 3}


@dataclass(frozen=True)
class Surface:
    Z: np.ndarray          # (ny, nx) node heights, mm, plate z
    x: np.ndarray          # (nx,) node x, mm
    y: np.ndarray          # (ny,) node y, mm
    pitch_mm: float
    subdiv: int
    interp: str
    relief_mm: float = float("nan")         # nominal relief after scaling
    printed_range_mm: float = float("nan")  # max Z - min Z
    scale: float = 1.0                      # relief / requested relief
    offset_mm: float = 0.0                  # shift of the mid-depth plane from t - relief / 2

    @property
    def node_mm(self) -> float:
        return self.pitch_mm / self.subdiv


MAPPINGS = ("clip", "tanh", "none")


def cell_values(A: np.ndarray, clip_pct: float = 99.5,
                sigma_cells: float = 0.0, mapping: str = "clip") -> np.ndarray:
    """v in [-1, 1] from A, then smoothed (reflect boundary) when
    sigma_cells > 0. With m = percentile(|A|, clip_pct):

    clip: v = clip(A / m, -1, 1), the paper prints' mapping (default)
    tanh: soft knee, v = tanh(A / m) / tanh(1), rescaled to span [-1, 1]
    none: v = A / max|A|, nothing clipped, the spikes take the range
    """
    A = np.asarray(A, dtype=np.float64)
    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}, got {mapping!r}")
    if mapping == "none":
        v = A / max(float(np.abs(A).max()), 1e-30)
    else:
        m = float(np.percentile(np.abs(A), clip_pct))
        if mapping == "clip":
            v = np.clip(A / max(m, 1e-30), -1.0, 1.0)
        else:
            v = np.tanh(A / max(m, 1e-30)) / np.tanh(1.0)
            v /= max(float(np.abs(v).max()), 1e-30)
    if sigma_cells > 0:
        v = gaussian_filter(v, sigma_cells, mode="reflect")
        v /= max(float(np.abs(v).max()), 1e-30)       # relief_mm stays peak to peak
    return v


def node_coords(n: int, pitch_mm: float, subdiv: int) -> np.ndarray:
    return np.arange(n * subdiv + 1) * (pitch_mm / subdiv)


def interpolate(v: np.ndarray, subdiv: int, method: str = "cubic") -> np.ndarray:
    """Cell-centre values (n, m) to node values (n subdiv + 1, m subdiv + 1).

    nearest/linear/cubic: ndimage spline of that order at the fractional
    cell index k / subdiv - 0.5, edge values extended (mode nearest).
    fourier: mirror the array to (2n, 2m) so it is even about every plate
    edge, zero-pad its spectrum by `subdiv`, shift by half a cell in the
    frequency domain so the samples land exactly on the nodes, inverse
    FFT. Band-limited and periodic, so it rings near sharp features.
    """
    v = np.asarray(v, dtype=np.float64)
    n, m = v.shape
    if method in _ORDER:
        iy = np.arange(n * subdiv + 1) / subdiv - 0.5
        ix = np.arange(m * subdiv + 1) / subdiv - 0.5
        IY, IX = np.meshgrid(iy, ix, indexing="ij")
        return map_coordinates(v, [IY, IX], order=_ORDER[method], mode="nearest")
    if method == "fourier":
        return _fourier(v, subdiv)
    raise ValueError(f"unknown interpolation {method!r}, choose from {METHODS}")


def _fourier(v: np.ndarray, s: int) -> np.ndarray:
    n, m = v.shape
    vm = np.block([[v, v[:, ::-1]], [v[::-1], v[::-1, ::-1]]])          # (2n, 2m)
    S = np.fft.fftshift(np.fft.fft2(vm))                                 # f in [-n, n)
    big = np.zeros((2 * n * s, 2 * m * s), dtype=complex)
    cy, cx = n * s, m * s                                                # index of f = 0
    big[cy - n:cy + n, cx - m:cx + m] = S
    # the Nyquist bins (-n, -m) stand for +-n, +-m: split them
    big[cy - n, :] *= 0.5
    big[cy + n, :] = big[cy - n, :]
    big[:, cx - m] *= 0.5
    big[:, cx + m] = big[:, cx - m]
    fy = np.arange(-cy, cy)[:, None]
    fx = np.arange(-cx, cx)[None, :]
    # original sample i sits at (i + 0.5) p: shift by -s/2 fine samples so
    # fine sample k sits at k p / s, i.e. on the node
    big *= np.exp(-1j * np.pi * (fy / (2 * n) + fx / (2 * m)))
    fine = np.fft.ifft2(np.fft.ifftshift(big)).real * s * s
    return fine[:n * s + 1, :m * s + 1]


def surface(A: np.ndarray, pitch_mm: float, thickness_mm: float, relief_mm: float,
            subdiv: int = 3, interp: str = "cubic", clip_pct: float = 99.5,
            sigma_cells: float = 0.0, max_range_mm: float | None = None,
            mapping: str = "clip") -> Surface:
    """Height field of A on the fine grid with its highest node at
    thickness_mm. The printed range is relief_mm times the interpolant's
    overshoot (1 for nearest and linear); with max_range_mm the relief is
    scaled down (never clipped) so the printed range fits."""
    if subdiv < 1:
        raise ValueError("subdiv must be >= 1")
    v = cell_values(A, clip_pct, sigma_cells, mapping)
    u = interpolate(v, subdiv, interp)
    rng = float(u.max() - u.min()) * relief_mm / 2
    scale = 1.0
    if max_range_mm is not None and rng > max_range_mm:
        scale = max_range_mm / rng
        relief_mm *= scale
        rng = max_range_mm
    Z = thickness_mm + (relief_mm / 2) * (u - u.max())
    offset = float(relief_mm / 2 * (1.0 - u.max()))
    n, m = v.shape
    return Surface(Z=Z, x=node_coords(m, pitch_mm, subdiv),
                   y=node_coords(n, pitch_mm, subdiv), pitch_mm=pitch_mm,
                   subdiv=subdiv, interp=interp, relief_mm=relief_mm,
                   printed_range_mm=rng, scale=scale, offset_mm=offset)


def tri_gradients(Z: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Signed (dh/dx, dh/dy) of every triangle of slab_mesh's triangulation
    (cell (i, j) split along the (i, j) to (i + 1, j + 1) diagonal), so the
    statistics describe the printed facets. Flat arrays of 2 (ny-1)(nx-1)
    equal-footprint triangles."""
    gx = np.diff(Z, axis=1) / dx            # (ny, nx-1), along rows
    gy = np.diff(Z, axis=0) / dy            # (ny-1, nx), along columns
    # triangle (a, b, c): row i edge a-b and column j+1 edge b-c
    # triangle (a, c, d): row i+1 edge d-c and column j edge a-d
    GX = np.concatenate([gx[:-1].ravel(), gx[1:].ravel()])
    GY = np.concatenate([gy[:, 1:].ravel(), gy[:, :-1].ravel()])
    return GX, GY


def facet_angle(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Overhang angle (deg) of each facet printed on edge: 0 for facets
    that do not hang (hy <= 0), else atan(hy / sqrt(1 + hx^2))."""
    return np.degrees(np.arctan2(np.maximum(gy, 0.0), np.hypot(1.0, gx)))


RULES = ("facet", "y", "x", "grad")


def critical_relief(gx1: np.ndarray, gy1: np.ndarray, max_overhang_deg: float,
                    rule: str = "facet") -> np.ndarray:
    """Relief (mm) at which each facet, with gradient (gx1, gy1) at 1 mm
    relief, reaches the overhang limit; inf if it never does. facet:
    R^2 (hy^2 - tan^2 a hx^2) = tan^2 a, inf when hy <= tan a |hx|.
    y / x / grad: tan a / |slope|."""
    t = tan(radians(max_overhang_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        if rule == "facet":
            d = gy1 ** 2 - t ** 2 * gx1 ** 2
            rc = np.where((gy1 > 1e-9) & (d > 1e-18), t / np.sqrt(np.abs(d)), np.inf)
        elif rule in ("x", "y", "grad"):
            s = {"x": np.abs(gx1), "y": np.abs(gy1), "grad": np.hypot(gx1, gy1)}[rule]
            rc = np.where(s > 1e-9, t / s, np.inf)
        else:
            raise ValueError(f"rule must be one of {RULES}")
    return rc


QUANTILES = (50, 90, 99, 99.9)


def _q(a: np.ndarray) -> dict:
    d = {f"q{str(q).replace('.', '')}": float(np.percentile(a, q)) for q in QUANTILES}
    d["max"] = float(a.max())
    return d


def slope_stats(surf: Surface, limit_deg: float = 45.0) -> dict:
    """Slope quantiles over the mesh triangles (every triangle weighs its
    footprint, so every 'frac' is a fraction of the plate's footprint), x
    and y separately (the matrix is anisotropic), the facet overhang
    angle on edge, and the fractions over `limit_deg` under the y rule
    and the facet rule."""
    d = surf.node_mm
    gx, gy = tri_gradients(surf.Z, d, d)
    ax, ay = np.abs(gx), np.abs(gy)
    g = np.hypot(ax, ay)
    fa = facet_angle(gx, gy)
    t = tan(radians(limit_deg))
    return {"abs_dx": _q(ax), "abs_dy": _q(ay), "grad": _q(g), "facet_deg": _q(fa),
            "limit_deg": limit_deg,
            "frac_dy_over_limit": float(np.mean(ay > t)),
            "frac_dx_over_limit": float(np.mean(ax > t)),
            "frac_facet_over_limit": float(np.mean(fa > limit_deg)),
            "frac_hanging": float(np.mean(gy > 0)),
            "max_angle_y_deg": float(np.degrees(np.arctan(ay.max()))),
            "max_angle_x_deg": float(np.degrees(np.arctan(ax.max()))),
            "max_facet_deg": float(fa.max())}


def fit_relief(A: np.ndarray, pitch_mm: float, max_overhang_deg: float = 45.0,
               area_quantile: float = 0.99, rule: str = "facet",
               **surface_kw) -> float:
    """Largest relief_mm (peak to peak) such that `area_quantile` of the
    footprint stays inside the overhang limit under `rule` (facet: the
    true on-edge overhang; y / x / grad: |slope| <= tan). Slopes are
    linear in relief_mm, so one surface at 1 mm gives every facet's
    critical relief and the answer is its (1 - q) quantile. inf when
    enough of the surface never hangs."""
    surf = surface(A, pitch_mm, thickness_mm=1.0, relief_mm=1.0, **surface_kw)
    d = surf.node_mm
    gx, gy = tri_gradients(surf.Z, d, d)
    rc = critical_relief(gx, gy, max_overhang_deg, rule)
    with np.errstate(invalid="ignore"):
        v = np.quantile(rc, 1.0 - area_quantile)       # nan only when all inf
    return float("inf") if np.isnan(v) else float(v)
