"""Display modes for an exactly-materialized matrix (1 entry = 1 pixel).

No pooling, no tone curve: the only freedom left is how a signed matrix
entry becomes ink. Image modes (gray, diverging) map every entry; glyph
modes (hinton, bubble) and 3D modes (wireframe, bars) only stay readable
for sides up to a few hundred, so they are meant for crops / detail views.

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


def best_crop(A: np.ndarray, size: int) -> tuple[int, int]:
    """Top-left corner of the size x size crop with the most energy."""
    n = A.shape[0]
    if size >= n:
        return 0, 0
    k = n // size
    blocks = np.abs(A[:k * size, :k * size]
                    ).reshape(k, size, k, size).sum(axis=(1, 3))
    i, j = np.unravel_index(np.argmax(blocks), blocks.shape)
    return i * size, j * size


def _signed_norm(A: np.ndarray, pct: float = 99.5) -> np.ndarray:
    m = np.percentile(np.abs(A), pct)
    return np.clip(A / max(m, 1e-30), -1.0, 1.0)


def hinton(A: np.ndarray, path: Path, dpi: int = 200) -> Path:
    """Hinton diagram: square area ~ |entry|, white = +, black = -."""
    V = _signed_norm(A)
    n = V.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#808080")
    half = np.sqrt(np.abs(V)) / 2
    ii, jj = np.nonzero(half > 0.02)
    for i, j in zip(ii, jj):
        h = half[i, j]
        c = "white" if V[i, j] > 0 else "black"
        ax.add_patch(plt.Rectangle((j - h, i - h), 2 * h, 2 * h,
                                   facecolor=c, edgecolor="none"))
    ax.set_xlim(-1, n)
    ax.set_ylim(n, -1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=ax.get_facecolor())
    plt.close(fig)
    return path


def bubble(A: np.ndarray, path: Path, cmap: str = "RdBu_r",
           dpi: int = 200) -> Path:
    """Bubble chart: circle area ~ |entry|, diverging color by value."""
    V = _signed_norm(A)
    n = V.shape[0]
    ii, jj = np.nonzero(np.abs(V) > 0.02)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#f5f2ec")
    # max circle diameter ~= grid spacing (720 pt axes width / n cells)
    s = np.abs(V[ii, jj]) * (720 / n) ** 2 * 0.9
    ax.scatter(jj, ii, s=s, c=V[ii, jj], cmap=cmap, vmin=-1, vmax=1,
               linewidths=0, alpha=0.9)
    ax.set_xlim(-1, n)
    ax.set_ylim(n, -1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=ax.get_facecolor())
    plt.close(fig)
    return path


def wireframe(A: np.ndarray, path: Path, max_lines: int = 128,
              dpi: int = 200) -> Path:
    """3D wireframe of the matrix as a height field."""
    V = _signed_norm(A)
    n = V.shape[0]
    stride = max(1, n // max_lines)
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(x, y, V, rstride=stride, cstride=stride,
                      linewidth=0.4, color="#202020")
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.35))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def bars3d(A: np.ndarray, path: Path, cmap: str = "RdBu_r",
           dpi: int = 200) -> Path:
    """3D bar chart (only readable for small crops, side <~ 48)."""
    V = _signed_norm(A)
    n = V.shape[0]
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    x, y, z = x.ravel(), y.ravel(), V.ravel()
    colors = plt.get_cmap(cmap)((z + 1) / 2)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection="3d")
    ax.bar3d(x, y, np.minimum(z, 0), 0.8, 0.8, np.abs(z),
             color=colors, shade=True, linewidth=0)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.4))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
