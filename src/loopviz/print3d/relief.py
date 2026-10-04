"""The song operator as a 3D-printed height field.

Same object as the paper prints (the exact n x n window-advance operator
A0 of songmatrix.py), different display: each matrix entry becomes a
square column of side `pitch` whose top height encodes the entry. The
matrix is never resampled - one cell per entry, every entry present.

Sizing
------
The paper print's pixel pitch (0.25 mm) is replaced by a printable cell
pitch (about 1-3 mm on FDM), so the canvas holds far fewer entries:

    n = floor(side / pitch),     f = rho n^2 / T        (songmatrix)

Pitch is therefore the sample-rate knob: at a fixed footprint and loop
length, halving the pitch quadruples f.

Height is the other budget. The top of a column lands on a layer
boundary, so the print carries at most

    levels = relief / layer + 1

distinct heights (~5-6 bits on FDM). Bigger relief buys levels but tall
thin columns fail: relief <= max_aspect * pitch keeps them printable.
Every entry is quantized to those levels *before* meshing, so the STL is
exactly what the printer can make, and `quantized_matrix` returns the
matrix as the object actually carries it, so playback of the physical
object can be measured (loop_degradation) rather than assumed.

Mesh
----
`heightfield_mesh` builds a watertight stepped solid with numpy: per-cell
top quads, a base grid, vertical walls only where neighbours differ, and
the outer skirt; duplicate vertices are welded so equal-height neighbours
share edges. Volume is base*side^2 + pitch^2 * sum(relief heights), which
`check_mesh` verifies analytically. STL is written directly (binary), no
CAD kernel needed; slicers take it as is.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, replace
from math import log2
from pathlib import Path

import numpy as np

from .songmatrix import Plan, plan

# -- printers ------------------------------------------------------------------

@dataclass(frozen=True)
class Printer:
    """What the machine can make. Numbers are conservative defaults, see
    doc/relief.md for the sources; override any of them from the CLI or
    the environment (LOOPVIZ_BED_MM, LOOPVIZ_NOZZLE_MM, LOOPVIZ_LAYER_MM,
    LOOPVIZ_STEP_LAYERS, LOOPVIZ_MIN_PITCH_MM, LOOPVIZ_MAX_ASPECT,
    LOOPVIZ_MAX_RELIEF_MM, LOOPVIZ_MARGIN_MM)."""

    name: str
    bed_mm: float = 305.0        # square build area side
    nozzle_mm: float = 0.4
    layer_mm: float = 0.10       # top surfaces land on multiples of this
    step_layers: int = 2         # layers per level: 1-layer steps hide in
                                 # top-surface noise (+-0.05 mm)
    min_pitch_mm: float = 2.0    # smallest cell that still reads as a crisp
                                 # square (>= 4 line widths, corner radius
                                 # ~ half a line width); 2.5 mm is safe
    max_aspect: float = 5.0      # max protrusion above the tallest
                                 # neighbour, in pitches (nozzle-drag wobble)
    max_relief_mm: float = 6.0   # height range cap (print time, looks)
    margin_mm: float = 2.5       # per side, kept clear of the bed edge

    @property
    def side_mm(self) -> float:
        return self.bed_mm - 2 * self.margin_mm

    @property
    def step_mm(self) -> float:
        """Height difference between adjacent levels."""
        return self.step_layers * self.layer_mm


PRINTERS = {
    # 0.4 mm nozzle, 0.45 mm lines: 2.0 mm cells marginal, 2.5 mm crisp
    "fdm04": Printer("fdm04", nozzle_mm=0.4, layer_mm=0.10, min_pitch_mm=2.0),
    # 0.25 mm nozzle, 0.28 mm lines: 1.5 mm cells crisp
    "fdm025": Printer("fdm025", nozzle_mm=0.25, layer_mm=0.05, min_pitch_mm=1.5),
    # 0.2 mm nozzle (Bambu class): 1.2 mm cells, 0.04-0.14 mm layers
    "fdm02": Printer("fdm02", nozzle_mm=0.2, layer_mm=0.06, min_pitch_mm=1.2),
    # MSLA resin: ~40 um pixels, 0.05 mm layers; the largest consumer
    # plates are 16:9 (330 x 185 mm), so 305 mm needs two tiles
    "msla": Printer("msla", bed_mm=185.0, nozzle_mm=0.043, layer_mm=0.05,
                    step_layers=1, min_pitch_mm=0.5, max_aspect=8.0, margin_mm=2.0),
}

_ENV = {"bed_mm": "LOOPVIZ_BED_MM", "nozzle_mm": "LOOPVIZ_NOZZLE_MM",
        "layer_mm": "LOOPVIZ_LAYER_MM", "min_pitch_mm": "LOOPVIZ_MIN_PITCH_MM",
        "max_aspect": "LOOPVIZ_MAX_ASPECT", "max_relief_mm": "LOOPVIZ_MAX_RELIEF_MM",
        "margin_mm": "LOOPVIZ_MARGIN_MM"}


def printer(name: str = "fdm04", **overrides) -> Printer:
    """A preset, then environment variables, then explicit overrides."""
    p = PRINTERS[name]
    env = {k: float(os.environ[v]) for k, v in _ENV.items() if v in os.environ}
    if "LOOPVIZ_STEP_LAYERS" in os.environ:
        env["step_layers"] = int(os.environ["LOOPVIZ_STEP_LAYERS"])
    return replace(p, **{**env, **{k: v for k, v in overrides.items() if v is not None}})


# -- sizing ------------------------------------------------------------------

@dataclass(frozen=True)
class ReliefPlan:
    """A printable (n, N, f, T) plus the geometry that displays it."""

    plan: Plan
    pitch_mm: float
    relief_mm: float      # height range spanned by the entries
    base_mm: float        # solid slab under the relief
    step_mm: float        # height between adjacent levels (whole layers)

    @property
    def n(self) -> int:
        return self.plan.n

    @property
    def side_mm(self) -> float:
        return self.n * self.pitch_mm

    @property
    def levels(self) -> int:
        """Distinct top heights the printer can realise."""
        return int(round(self.relief_mm / self.step_mm)) + 1

    @property
    def bits_per_cell(self) -> float:
        return log2(self.levels)

    @property
    def capacity_bits(self) -> float:
        """Information the object can hold: cells x bits per cell."""
        return self.n ** 2 * self.bits_per_cell

    @property
    def aspect(self) -> float:
        return self.relief_mm / self.pitch_mm

    @property
    def height_mm(self) -> float:
        return self.base_mm + self.relief_mm


def relief_plan(T: float, pr: Printer, pitch_mm: float, rho: float = 0.95,
                relief_mm: float | None = None, base_mm: float = 2.0,
                side_mm: float | None = None, tiles: int = 1) -> ReliefPlan:
    """Largest matrix of this pitch that fits the printer, at rank fraction rho.

    With tiles = k the piece is k x k prints: n = k * floor(side / pitch)
    so every tile fits the bed exactly. relief defaults to
    min(max_relief, max_aspect * pitch), rounded down to whole level steps.
    """
    side = pr.side_mm if side_mm is None else side_mm
    if tiles < 1:
        raise ValueError("tiles must be >= 1")
    n = tiles * int(np.floor(side / pitch_mm + 1e-9))
    if n < 2:
        raise ValueError(f"pitch {pitch_mm} mm leaves n={n} on {side} mm")
    if base_mm <= 0:
        raise ValueError("base_mm must be > 0 (the plinth carries the relief)")
    if relief_mm is None:
        relief_mm = min(pr.max_relief_mm, pr.max_aspect * pitch_mm)
    relief_mm = int(round(relief_mm / pr.step_mm, 6)) * pr.step_mm
    if relief_mm < pr.step_mm:
        raise ValueError(f"relief {relief_mm} mm is below one level step "
                         f"({pr.step_mm} mm): nothing to display")
    return ReliefPlan(plan=plan(T, n=n, rho=rho), pitch_mm=pitch_mm,
                      relief_mm=relief_mm, base_mm=base_mm, step_mm=pr.step_mm)


def sweep(T: float, pr: Printer, pitches: list[float], rho: float = 0.95,
          **kw) -> list[ReliefPlan]:
    return [relief_plan(T, pr, p, rho=rho, **kw) for p in pitches]


# -- heights ---------------------------------------------------------------------

def signed_levels(A: np.ndarray, levels: int, clip_pct: float = 99.5) -> np.ndarray:
    """Quantize a signed matrix to integer levels 0..levels-1, zero at centre.

    Symmetric robust clip at the clip_pct percentile of |A| (as the paper
    prints do), then uniform steps. With an odd number of levels zero maps
    exactly to the middle level.
    """
    m = np.percentile(np.abs(A), clip_pct)
    v = np.clip(A / max(m, 1e-30), -1.0, 1.0)              # [-1, 1]
    return np.rint((v + 1.0) / 2.0 * (levels - 1)).astype(np.int64)


def quantized_matrix(A: np.ndarray, levels: int, clip_pct: float = 99.5) -> np.ndarray:
    """The matrix as the printed object carries it: clipped and stepped,
    on the original scale, so it can be applied to windows."""
    m = np.percentile(np.abs(A), clip_pct)
    q = signed_levels(A, levels, clip_pct)
    return (q / (levels - 1) * 2.0 - 1.0) * m


def heights(A: np.ndarray, rp: ReliefPlan, clip_pct: float = 99.5) -> np.ndarray:
    """Top height (mm, from the bed) of every cell, on layer boundaries."""
    q = signed_levels(A, rp.levels, clip_pct)
    return rp.base_mm + q * rp.step_mm


def max_protrusion(H: np.ndarray) -> float:
    """Largest height of any cell above its tallest edge-neighbour: the
    only part of a cell that is a free-standing column (the grid is one
    solid). This, not the relief range, is what the aspect limit bounds."""
    pad = np.pad(H, 1, constant_values=-np.inf)
    nb = np.max([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]], axis=0)
    return float(np.max(H - nb))


# -- mesh ----------------------------------------------------------------------

def _ladder(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Triangulate the strip between two vertical polylines A, B of shape
    (k, s, 3), both sorted bottom-to-top and sharing first/last heights.
    Fixed topology; points that were clamped onto a neighbour become
    degenerate triangles and are removed by weld(). Orientation follows
    (A, B): for a wall at constant x with A at the smaller y the normal
    is +x; at constant y with A at the smaller x it is -y."""
    s = A.shape[1]
    tris = []
    for t in range(s - 1):
        tris.append(np.stack([A[:, t], B[:, t], B[:, t + 1]], axis=1))
        tris.append(np.stack([A[:, t], B[:, t + 1], A[:, t + 1]], axis=1))
    return np.concatenate(tris, axis=0)


def _side(x: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
          extra: list[np.ndarray]) -> np.ndarray:
    """Vertical polyline at (x, y): lo, the extra heights clamped into
    [lo, hi] and sorted, hi. Shape (k, 2 + len(extra), 3)."""
    z = np.stack([lo, *[np.clip(e, lo, hi) for e in extra], hi], axis=1)
    z = np.sort(z, axis=1)
    k, s = z.shape
    xy = np.broadcast_to(np.stack([x, y], axis=1)[:, None, :], (k, s, 2))
    return np.concatenate([xy, z[:, :, None]], axis=2)


def heightfield_mesh(H: np.ndarray, pitch_mm: float,
                     weld_tol_mm: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Watertight stepped solid from an (n, m) height map (mm from z=0).

    Cell (i, j) occupies x in [j p, (j+1) p], y in [i p, (i+1) p], z in
    [0, H[i, j]]. Walls are split wherever a neighbouring cell's top meets
    them, so there are no T-junctions: every edge is shared by exactly two
    faces (four where two cells touch only along a vertical edge, the
    unavoidable voxel case). Returns (V, F): float64 (nv, 3) vertices and
    int64 (nf, 3) faces, outward-oriented.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2 or (H <= 0).any():
        raise ValueError("H must be 2D with strictly positive heights")
    n, m = H.shape
    p = float(pitch_mm)
    x0 = np.arange(m) * p
    y0 = np.arange(n) * p
    X0, Y0 = np.meshgrid(x0, y0)      # (n, m): X varies along j (columns)
    X1, Y1 = X0 + p, Y0 + p
    # heights of the row above / below, column left / right (NaN = none;
    # np.clip keeps the bound when the value is NaN via nan_to_num below)
    pad = np.full((n + 2, m + 2), np.nan)
    pad[1:-1, 1:-1] = H
    tris = []

    def flat(*a):
        return [np.asarray(v, dtype=np.float64).ravel() for v in a]

    def pts(x, y, z):
        return np.stack([x, y, z], axis=1)

    # tops (normal +z) and bottom (normal -z), two triangles each
    xa, ya, xb, yb, h = flat(X0, Y0, X1, Y1, H)
    z0 = np.zeros_like(h)
    tris.append(np.stack([pts(xa, ya, h), pts(xb, ya, h), pts(xb, yb, h)], 1))
    tris.append(np.stack([pts(xa, ya, h), pts(xb, yb, h), pts(xa, yb, h)], 1))
    tris.append(np.stack([pts(xa, ya, z0), pts(xa, yb, z0), pts(xb, yb, z0)], 1))
    tris.append(np.stack([pts(xa, ya, z0), pts(xb, yb, z0), pts(xb, ya, z0)], 1))

    def clean(e, lo):
        return np.where(np.isnan(e), lo, e)

    # walls between column neighbours (i, j) | (i, j+1) at x = (j+1) p
    if m > 1:
        hl, hr = flat(H[:, :-1], H[:, 1:])
        lo, hi = np.minimum(hl, hr), np.maximum(hl, hr)
        xw, ya, yb = flat(X1[:, :-1], Y0[:, :-1], Y1[:, :-1])
        # corners at y = ya touch row i-1, at y = yb touch row i+1
        below = [clean(e, lo) for e in flat(pad[:-2, 1:-2], pad[:-2, 2:-1])]
        above = [clean(e, lo) for e in flat(pad[2:, 1:-2], pad[2:, 2:-1])]
        A = _side(xw, ya, lo, hi, below)
        B = _side(xw, yb, lo, hi, above)
        left_taller = hl > hr                       # outward normal +x
        tris.append(_ladder(A[left_taller], B[left_taller]))
        tris.append(_ladder(B[~left_taller], A[~left_taller]))
    # walls between row neighbours (i, j) | (i+1, j) at y = (i+1) p
    if n > 1:
        hb, ht = flat(H[:-1, :], H[1:, :])
        lo, hi = np.minimum(hb, ht), np.maximum(hb, ht)
        yw, xa, xb = flat(Y1[:-1, :], X0[:-1, :], X1[:-1, :])
        left = [clean(e, lo) for e in flat(pad[1:-2, :-2], pad[2:-1, :-2])]
        right = [clean(e, lo) for e in flat(pad[1:-2, 2:], pad[2:-1, 2:])]
        A = _side(xa, yw, lo, hi, left)
        B = _side(xb, yw, lo, hi, right)
        top_taller = ht > hb                        # outward normal -y
        tris.append(_ladder(A[top_taller], B[top_taller]))
        tris.append(_ladder(B[~top_taller], A[~top_taller]))

    # outer skirt, z = 0 up to each edge cell; corners touch one neighbour
    zero = np.zeros(n)
    h = H[:, 0]
    A = _side(np.zeros(n), y0, zero, h, [clean(pad[:-2, 1], zero)])
    B = _side(np.zeros(n), y0 + p, zero, h, [clean(pad[2:, 1], zero)])
    tris.append(_ladder(B, A))                      # x = 0, normal -x
    h = H[:, -1]
    xe = np.full(n, m * p)
    A = _side(xe, y0, zero, h, [clean(pad[:-2, -2], zero)])
    B = _side(xe, y0 + p, zero, h, [clean(pad[2:, -2], zero)])
    tris.append(_ladder(A, B))                      # x = m p, normal +x
    zero = np.zeros(m)
    h = H[0, :]
    A = _side(x0, np.zeros(m), zero, h, [clean(pad[1, :-2], zero)])
    B = _side(x0 + p, np.zeros(m), zero, h, [clean(pad[1, 2:], zero)])
    tris.append(_ladder(A, B))                      # y = 0, normal -y
    h = H[-1, :]
    ye = np.full(m, n * p)
    A = _side(x0, ye, zero, h, [clean(pad[-2, :-2], zero)])
    B = _side(x0 + p, ye, zero, h, [clean(pad[-2, 2:], zero)])
    tris.append(_ladder(B, A))                      # y = n p, normal +y

    T = np.concatenate([t for t in tris if t.size], axis=0)   # (k, 3, 3)
    return weld(T.reshape(-1, 3), weld_tol_mm)


def weld(P: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices of a triangle soup (3k, 3) and drop
    degenerate triangles (zero-height walls between equal neighbours)."""
    key = np.round(P / tol).astype(np.int64)
    V, inv = np.unique(key, axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    return V.astype(np.float64) * tol, F[ok]


def mesh_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Signed volume by the divergence theorem (positive iff outward)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def expected_volume(H: np.ndarray, pitch_mm: float) -> float:
    return float(np.sum(H) * pitch_mm ** 2)


def check_mesh(V: np.ndarray, F: np.ndarray, H: np.ndarray | None = None,
               pitch_mm: float | None = None) -> dict:
    """Manifold checks with numpy alone.

    closed   : every undirected edge is used an even number of times (2,
               or 4 on a voxel edge where two columns touch only along a
               vertical edge - the solid is still closed there)
    oriented : every directed edge is used as often as its reverse, so
               neighbouring faces agree on outside
    volume   : divergence theorem, compared with the analytic
               base*area + pitch^2 * sum(H) when H is given
    """
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    und = np.sort(e, axis=1)
    _, cnt = np.unique(und, axis=0, return_counts=True)
    # oriented: the multiset of directed edges equals that of their reverses
    de, dcnt = np.unique(e, axis=0, return_counts=True)
    re_, rcnt = np.unique(e[:, ::-1], axis=0, return_counts=True)
    balanced = de.shape == re_.shape and bool((de == re_).all() and (dcnt == rcnt).all())
    out = {
        "vertices": int(V.shape[0]),
        "faces": int(F.shape[0]),
        "closed": bool((cnt % 2 == 0).all()),
        "oriented": balanced,
        "nonmanifold_edges": int((cnt > 2).sum()),
        "volume": mesh_volume(V, F),
    }
    if H is not None and pitch_mm is not None:
        out["expected_volume"] = expected_volume(H, pitch_mm)
        out["volume_ok"] = bool(abs(out["volume"] - out["expected_volume"])
                                <= 1e-6 * max(1.0, out["expected_volume"]))
    out["watertight"] = out["closed"] and out["oriented"]
    return out


def write_stl(V: np.ndarray, F: np.ndarray, path: Path,
              name: str = "loopviz relief") -> Path:
    """Binary STL, one 50-byte record per triangle, normals from winding."""
    path = Path(path)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(F.shape[0], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                      ("attr", "<u2")])
    rec["n"] = nrm
    rec["v"] = np.stack([a, b, c], axis=1)
    header = name.encode()[:80].ljust(80, b"\0")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", F.shape[0]))
        fh.write(rec.tobytes())
    return path


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary STL back to a welded (V, F); used by the tests."""
    with open(path, "rb") as fh:
        fh.read(80)
        (k,) = struct.unpack("<I", fh.read(4))
        rec = np.frombuffer(fh.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                              ("attr", "<u2")], count=k)
    return weld(rec["v"].reshape(-1, 3).astype(np.float64), 1e-4)
