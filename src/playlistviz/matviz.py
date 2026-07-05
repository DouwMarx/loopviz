"""Display modes for an exactly-materialized matrix (1 entry = 1 pixel).

No pooling, no tone curve, and NO CROPS: every mode shows every entry of
the matrix, because the artwork's claim is that the displayed object
reproduces the song - a crop does not. The only freedom is how a signed
entry becomes ink.

Image modes (gray, diverging) map entries to pixels 1:1. Glyph modes
(hinton, bubble) rasterize one cell of `cell x cell` pixels per entry
with pure numpy - a 1000x1000 matrix becomes an 8000x8000 image, every
entry present. A 3D bar mode existed briefly and was removed: a million
bars cannot actually be rendered, and showing a subset is against the
spirit of the piece.

All figure functions save to a path and return it; array functions return
float arrays in [0, 1] (gray) or [0, 1]^3 (rgb).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def signed_unit(A: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Symmetric robust map of a signed matrix to [0, 1]; 0 -> mid-gray."""
    m = np.percentile(np.abs(A), pct)
    return np.clip(A / max(2 * m, 1e-30) + 0.5, 0.0, 1.0)


def gray(A: np.ndarray, pct: float = 99.5) -> np.ndarray:
    """Black-and-white: sign and magnitude as one luminance axis."""
    return signed_unit(A, pct)


def diverging(A: np.ndarray, cmap: str = "RdBu_r",
              pct: float = 99.5) -> np.ndarray:
    """Diverging palette around zero -> (n, n, 3) RGB in [0, 1]."""
    return plt.get_cmap(cmap)(signed_unit(A, pct))[..., :3]


def save_rgb(img: np.ndarray, path: Path) -> Path:
    from PIL import Image

    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _signed_norm(A: np.ndarray, pct: float = 99.5) -> np.ndarray:
    m = np.percentile(np.abs(A), pct)
    return np.clip(A / max(m, 1e-30), -1.0, 1.0)


def _cell_grid(n: int, cell: int, radii: np.ndarray,
               dist: np.ndarray) -> np.ndarray:
    """Boolean (n*cell, n*cell) mask: pixel on iff inside its cell's glyph.

    dist is the (cell, cell) distance field of one cell; a pixel at offset
    (dy, dx) in cell (i, j) is on iff dist[dy, dx] <= radii[i, j].
    """
    R = np.repeat(np.repeat(radii.astype(np.float32), cell, 0), cell, 1)
    D = np.tile(dist.astype(np.float32), (n, n))
    return D <= R


def hinton(A: np.ndarray, path: Path, cell: int = 8) -> Path:
    """Full-matrix Hinton diagram: square area ~ |entry|, white = +,
    black = -, on mid-gray. Rasterized with numpy, one cell per entry."""
    from PIL import Image

    V = _signed_norm(A)
    n = V.shape[0]
    c = (cell - 1) / 2.0
    off = np.abs(np.arange(cell) - c)
    dist = np.maximum(off[:, None], off[None, :])       # Chebyshev: squares
    radii = c * np.sqrt(np.abs(V))
    mask = _cell_grid(n, cell, radii, dist)
    sign = np.repeat(np.repeat(V > 0, cell, 0), cell, 1)
    img = np.full((n * cell, n * cell), 128, dtype=np.uint8)
    img[mask & sign] = 255
    img[mask & ~sign] = 0
    Image.fromarray(img, mode="L").save(path)
    return path


def bubble(A: np.ndarray, path: Path, cell: int = 8,
           cmap: str = "RdBu_r") -> Path:
    """Full-matrix bubble chart: circle area ~ |entry|, diverging color.
    Rasterized with numpy, one cell per entry."""
    from PIL import Image

    V = _signed_norm(A)
    n = V.shape[0]
    c = (cell - 1) / 2.0
    off = np.arange(cell) - c
    dist = np.sqrt(off[:, None] ** 2 + off[None, :] ** 2)  # circles
    radii = c * np.sqrt(np.abs(V))
    mask = _cell_grid(n, cell, radii, dist)
    colors = (plt.get_cmap(cmap)((V + 1) / 2)[..., :3] * 255).astype(np.uint8)
    img = np.full((n * cell, n * cell, 3), (245, 242, 236), dtype=np.uint8)
    big = np.repeat(np.repeat(colors, cell, 0), cell, 1)
    img[mask] = big[mask]
    Image.fromarray(img, mode="RGB").save(path)
    return path


def wireframe(A: np.ndarray, path: Path, dpi: int = 200) -> Path:
    """3D wireframe of the FULL matrix as a height field - every row and
    column drawn (stride 1). Dense for large n, but complete."""
    V = _signed_norm(A)
    n = V.shape[0]
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(x, y, V, rstride=1, cstride=1,
                      linewidth=0.08, color="#202020")
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.35))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
