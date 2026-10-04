"""Closed slab meshes from a node-grid height field, plus the small
geometry kit around them: engraved labels, rigid transforms, print
metrics (overhang), concatenation and a fast STL vertex reader.

slab_mesh builds a 2-manifold: the top grid (two triangles per fine
cell, diagonal (i, j) to (i + 1, j + 1)), a bottom (the full grid when
the back carries engraving, else a fan from one centre vertex to the
boundary nodes, which keeps the uncut plate's file half the size) and
four vertical sides as quads between consecutive boundary nodes. Every
edge is shared by exactly two faces and the winding is outward. The
volume is `slab_volume`, the exact integral of the triangulated surface
(trapezoidal plus a corner term of order dx dy / 12 times four corner
heights).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

DENSITY_G_CM3 = 1.24                       # PLA
OVERHANG_EDGES = (0.0, 30.0, 45.0, 50.0, 60.0, 90.0)

# rotation about x by +90 deg: (x, y, z) -> (x, -z, y); det +1
ON_EDGE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


# -- grid topology ---------------------------------------------------------------

def grid_faces(ny: int, nx: int) -> np.ndarray:
    """Two upward (+z) triangles per cell of a row-major (ny, nx) node grid."""
    i, j = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
    a = (i * nx + j).ravel()
    b, c, d = a + 1, a + nx + 1, a + nx
    return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int64)


def boundary_loop(ny: int, nx: int) -> np.ndarray:
    """Boundary node indices, counter-clockwise seen from +z, from (0, 0)."""
    bottom = np.arange(nx)                                  # y = 0, x increasing
    right = nx - 1 + nx * np.arange(1, ny)                  # x = max, y increasing
    top = (ny - 1) * nx + np.arange(nx - 2, -1, -1)         # y = max, x decreasing
    left = nx * np.arange(ny - 2, 0, -1)                    # x = 0, y decreasing
    return np.concatenate([bottom, right, top, left]).astype(np.int64)


# -- the slab --------------------------------------------------------------------

def slab_mesh_tagged(top: np.ndarray, bottom: np.ndarray | float,
                     x: np.ndarray, y: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(V, F, tag) with tag 0 = top, 1 = bottom, 2 = side per face."""
    top = np.asarray(top, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ny, nx = top.shape
    if x.shape != (nx,) or y.shape != (ny,) or ny < 2 or nx < 2:
        raise ValueError("top must be (len(y), len(x)) with at least 2 nodes per axis")
    X, Y = np.meshgrid(x, y)
    Vt = np.stack([X.ravel(), Y.ravel(), top.ravel()], 1)
    nt = Vt.shape[0]
    loop = boundary_loop(ny, nx)
    nb = loop.size
    Ft = grid_faces(ny, nx)
    if np.ndim(bottom) == 0:
        z0 = float(bottom)
        if (top <= z0).any():
            raise ValueError("top must lie above the bottom everywhere")
        Vb = np.concatenate([Vt[loop], [[x.mean(), y.mean(), 0.0]]])
        Vb[:, 2] = z0
        centre = nt + nb
        k = np.arange(nb)
        k1 = (k + 1) % nb
        Fb = np.stack([np.full(nb, centre), nt + k1, nt + k], 1)
        B = nt + k                                        # bottom index of loop node k
    else:
        bottom = np.asarray(bottom, dtype=np.float64)
        if bottom.shape != top.shape or (top <= bottom).any():
            raise ValueError("bottom must match top's shape and lie below it")
        Vb = np.stack([X.ravel(), Y.ravel(), bottom.ravel()], 1)
        Fb = (Ft + nt)[:, ::-1]
        B = nt + loop
    k = np.arange(nb)
    T, T1 = loop[k], loop[(k + 1) % nb]
    B1 = B[(k + 1) % nb]
    Fs = np.concatenate([np.stack([T, B, B1], 1), np.stack([T, B1, T1], 1)])
    V = np.concatenate([Vt, Vb])
    F = np.concatenate([Ft, Fb, Fs]).astype(np.int64)
    tag = np.concatenate([np.zeros(len(Ft), np.int8), np.ones(len(Fb), np.int8),
                          np.full(len(Fs), 2, np.int8)])
    return V, F, tag


def slab_mesh(top: np.ndarray, bottom: np.ndarray | float,
              x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed outward-oriented slab between the height fields top and
    bottom (an array, or a constant for a flat back) over the node grid
    (x, y). Vertices float64 (nv, 3), faces int64 (nf, 3)."""
    V, F, _ = slab_mesh_tagged(top, bottom, x, y)
    return V, F


def _tri_integral(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(x)[None, :]
    dy = np.diff(y)[:, None]
    a, b, c, d = h[:-1, :-1], h[:-1, 1:], h[1:, 1:], h[1:, :-1]
    return float(np.sum(dx * dy * (2 * a + b + 2 * c + d) / 6.0))


def slab_volume(top: np.ndarray, bottom: np.ndarray | float,
                x: np.ndarray, y: np.ndarray) -> float:
    """Exact volume of slab_mesh: the integral of the triangulated top
    minus that of the bottom."""
    top = np.asarray(top, dtype=np.float64)
    vt = _tri_integral(top, x, y)
    if np.ndim(bottom) == 0:
        area = (x[-1] - x[0]) * (y[-1] - y[0])
        return vt - float(bottom) * area
    return vt - _tri_integral(np.asarray(bottom, dtype=np.float64), x, y)


def trapezoid_volume(top: np.ndarray, bottom: np.ndarray | float,
                     x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal integral of top - bottom (differs from slab_volume by a
    corner term of order dx dy / 12 times four corner heights)."""
    h = np.asarray(top, dtype=np.float64) - bottom
    return float(np.trapezoid(np.trapezoid(h, x, axis=1), y))


# -- labels ------------------------------------------------------------------------

def text_mask(text: str, cap_px: float, glyph_up: bool = True) -> np.ndarray:
    """Boolean raster (rows top-down) of `text` with cap height cap_px
    pixels in Pillow's bundled font, followed by a filled triangle
    pointing up (apex at the top row) when glyph_up."""
    from PIL import Image, ImageDraw, ImageFont

    probe = ImageFont.load_default(size=100)
    l, t, r, b = probe.getbbox("H")
    font = ImageFont.load_default(size=max(1.0, 100.0 * cap_px / (b - t)))
    l, t, r, b = font.getbbox(text)
    tw, th = r - l, b - t
    cap = max(round(cap_px), 1)
    tri_w = max(round(0.8 * cap), 2) if glyph_up else 0
    gap = max(round(0.5 * cap), 1) if glyph_up else 0
    W, H = tw + gap + tri_w, max(th, cap)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    d.text((-l, H - th - t), text, fill=255, font=font)
    if glyph_up:
        x0 = tw + gap
        d.polygon([(x0 + tri_w / 2, H - cap), (x0, H - 1), (x0 + tri_w - 1, H - 1)], fill=255)
    return np.asarray(img) > 127


def engrave_text(bottom: np.ndarray | float, x: np.ndarray, y: np.ndarray, text: str,
                 height_mm: float, depth_mm: float, anchor_xy: tuple[float, float],
                 mirror_x: bool = True, glyph_up: bool = True) -> np.ndarray:
    """Recess `text` (cap height height_mm) by depth_mm into the back face
    z = bottom, the label box's top-left corner at anchor_xy (plate frame).
    mirror_x flips the raster so it reads correctly when the back is seen
    from behind (looking along +z). Raises if the ink would touch the
    boundary nodes, which must stay on the back plane."""
    ny, nx = y.size, x.size
    out = np.full((ny, nx), float(bottom)) if np.ndim(bottom) == 0 \
        else np.array(bottom, dtype=np.float64)
    d = float(y[1] - y[0])
    m = text_mask(text, height_mm / d, glyph_up)
    if mirror_x:
        m = m[:, ::-1]
    m = m[::-1]                                        # row 0 is now the lowest y
    h, w = m.shape
    j0 = int(np.argmin(np.abs(x - anchor_xy[0])))
    i1 = int(np.argmin(np.abs(y - anchor_xy[1])))      # top row of the box
    i0 = i1 - h + 1
    if not (1 <= i0 and i1 <= ny - 2 and 1 <= j0 and j0 + w - 1 <= nx - 2):
        raise ValueError(f"label {text!r} ({w}x{h} nodes) does not fit inside the "
                         f"{nx}x{ny} node grid at anchor {anchor_xy}")
    sub = out[i0:i1 + 1, j0:j0 + w]
    sub[m] = sub[m] + depth_mm
    return out


def manifold_check(V: np.ndarray, F: np.ndarray) -> dict:
    """Closed 2-manifold test with integer edge keys (fast on millions of
    faces): every undirected edge in exactly two faces, every directed
    edge used once (consistent winding), positive volume."""
    from .relief import mesh_volume

    nv = V.shape[0]
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    directed = e[:, 0] * nv + e[:, 1]
    und = np.minimum(e[:, 0], e[:, 1]) * nv + np.maximum(e[:, 0], e[:, 1])
    _, cnt = np.unique(und, return_counts=True)
    _, dcnt = np.unique(directed, return_counts=True)
    vol = mesh_volume(V, F)
    out = {"vertices": nv, "faces": int(F.shape[0]), "closed": bool((cnt == 2).all()),
           "oriented": bool((dcnt == 1).all()) and cnt.size * 2 == directed.size,
           "volume": vol}
    out["watertight"] = out["closed"] and out["oriented"] and vol > 0
    return out


# -- rigid motion ------------------------------------------------------------------

def homogeneous(R: np.ndarray, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def transform(V: np.ndarray, R: np.ndarray, t=None) -> np.ndarray:
    """Apply a 4x4 matrix, or (R 3x3, t)."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape == (4, 4):
        M = R
    else:
        M = homogeneous(R, (0.0, 0.0, 0.0) if t is None else t)
    return V @ M[:3, :3].T + M[:3, 3]


# -- metrics -------------------------------------------------------------------------

def face_normals(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals and areas per face."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    twice = np.linalg.norm(n, axis=1)
    return n / np.maximum(twice, 1e-30)[:, None], 0.5 * twice


def overhang_angles(V: np.ndarray, F: np.ndarray, up=(0.0, 0.0, 1.0),
                    skip_floor: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per face: overhang angle (degrees, 0 = vertical wall, 90 = ceiling,
    NaN for up-facing or floor faces), area, and floor mask. The floor is
    every face lying flat on the lowest plane along `up` (bed contact)."""
    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    n, area = face_normals(V, F)
    d = n @ up
    hmin = (V @ up).min()
    floor = np.zeros(len(F), bool)
    if skip_floor:
        hv = (V @ up)[F]
        floor = (d < -0.999) & (np.abs(hv - hmin) < 1e-6).all(axis=1)
    alpha = np.full(len(F), np.nan)
    down = (d < 0) & ~floor
    alpha[down] = np.degrees(np.arcsin(np.clip(-d[down], 0.0, 1.0)))
    return alpha, area, floor


def mesh_metrics(V: np.ndarray, F: np.ndarray, up=(0.0, 0.0, 1.0),
                 skip_floor: bool = True) -> dict:
    """Volume, mass (PLA), area, bbox and area-weighted overhang histogram
    for faces whose normal points against `up` (floor faces excluded),
    plus the same histogram for up-facing slopes (0 = vertical, 90 =
    flat top)."""
    from .relief import mesh_volume

    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    alpha, area, floor = overhang_angles(V, F, up, skip_floor)
    n, _ = face_normals(V, F)
    d = n @ up
    total = float(area.sum())
    down = ~np.isnan(alpha)
    hist, _ = np.histogram(alpha[down], bins=OVERHANG_EDGES, weights=area[down])
    over = float(area[down & (alpha > 45.0)].sum())
    upf = d > 0
    beta = np.degrees(np.arcsin(np.clip(d[upf], 0.0, 1.0)))
    uhist, _ = np.histogram(beta, bins=OVERHANG_EDGES, weights=area[upf])
    vol = mesh_volume(V, F)
    return {
        "volume_mm3": vol, "mass_g": vol * DENSITY_G_CM3 / 1000.0,
        "area_mm2": total, "floor_area_mm2": float(area[floor].sum()),
        "bbox_min": V.min(axis=0).tolist(), "bbox_max": V.max(axis=0).tolist(),
        "overhang": {"edges_deg": list(OVERHANG_EDGES), "area_mm2": hist.tolist(),
                     "area_over_45_mm2": over, "frac_over_45": over / max(total, 1e-30),
                     "max_deg": float(alpha[down].max()) if down.any() else 0.0},
        "upslope": {"edges_deg": list(OVERHANG_EDGES), "area_mm2": uhist.tolist()},
    }


def concat(meshes) -> tuple[np.ndarray, np.ndarray]:
    """Several (V, F) into one vertex and one face array (separate shells)."""
    Vs, Fs, off = [], [], 0
    for V, F in meshes:
        Vs.append(V)
        Fs.append(F + off)
        off += V.shape[0]
    return np.concatenate(Vs), np.concatenate(Fs)


def stl_vertices(path: Path, tol: float = 1e-3) -> np.ndarray:
    """Distinct vertices of a binary STL, rounded to tol (fast: one
    integer key per vertex instead of np.unique over rows)."""
    with open(path, "rb") as fh:
        fh.read(80)
        (k,) = struct.unpack("<I", fh.read(4))
        rec = np.frombuffer(fh.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                              ("attr", "<u2")], count=k)
    P = rec["v"].reshape(-1, 3).astype(np.float64)
    q = np.round(P / tol).astype(np.int64)
    lo = q.min(axis=0)
    q -= lo
    span = q.max(axis=0) + 1
    key = (q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2]
    _, idx = np.unique(key, return_index=True)
    return P[idx]
