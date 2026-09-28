"""Pictures of a relief before it is printed.

mesh_views  : the actual triangles of the STL, drawn with matplotlib from
              four viewpoints (top, three obliques) with flat shading.
              Exact - what you see is what the file contains - but O(faces)
              polygons through matplotlib, so for n <~ 60.
hillshade   : lit height map for any n; fast, but shows the height field,
              not the mesh. Use it to judge the large prints.
heightmap   : 16-bit PNG of the cell heights (lowest cell = 0, highest =
              65535; the base slab is not encoded). Some
              slicers import height maps directly, but they interpolate
              between pixels; the STL is the authoritative stepped object.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _shade(V: np.ndarray, F: np.ndarray, light=(0.4, -0.6, 0.7)) -> np.ndarray:
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0.0, 1.0)
    return 0.35 + 0.6 * lam


def mesh_views(V: np.ndarray, F: np.ndarray, path: Path, title: str = "",
               dpi: int = 130) -> Path:
    """Four shaded views of the exact mesh."""
    lo, hi = V.min(axis=0), V.max(axis=0)
    span = hi - lo
    shade = _shade(V, F)
    views = [("top", 90, -90), ("front-left", 35, -60),
             ("front-right", 35, -120), ("low oblique", 18, -40)]
    fig = plt.figure(figsize=(12, 10))
    for k, (name, elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, k, projection="3d")
        tris = V[F]
        pc = Poly3DCollection(tris, linewidths=0.15, edgecolors="#333333")
        pc.set_facecolor(np.repeat(shade[:, None], 3, axis=1))
        ax.add_collection3d(pc)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(0, hi[2])
        ax.set_box_aspect((span[0], span[1], max(span[2], 1e-9)))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x mm")
        ax.set_ylabel("y mm")
        ax.set_zlabel("z mm")
        ax.set_title(name)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def hillshade(H: np.ndarray, pitch_mm: float, path: Path, title: str = "",
              azimuth_deg: float = 315.0, altitude_deg: float = 45.0,
              dpi: int = 100) -> Path:
    """Lambertian lighting of the cell height map; each cell drawn flat,
    so steps between cells appear as one-pixel light/dark seams."""
    from matplotlib.colors import LightSource

    ls = LightSource(azdeg=azimuth_deg, altdeg=altitude_deg)
    # upsample cells to 3x3 pixels so seams between cells are visible
    Hu = np.repeat(np.repeat(H, 3, axis=0), 3, axis=1)
    rgb = ls.shade(Hu, cmap=plt.get_cmap("gray"), vert_exag=1.0,
                   dx=pitch_mm / 3, dy=pitch_mm / 3, blend_mode="soft")
    n, m = H.shape
    fig, ax = plt.subplots(figsize=(8, 8 * n / m))
    ax.imshow(rgb, origin="lower", extent=(0, m * pitch_mm, 0, n * pitch_mm),
              interpolation="nearest")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def heightmap_png(H: np.ndarray, path: Path) -> Path:
    from .render import _write_png16_gray

    z = (H - H.min()) / max(H.max() - H.min(), 1e-30)
    _write_png16_gray((z * 65535).astype("<u2"), path)
    return path


def sheet(images: list[tuple[str, Path]], path: Path, cols: int = 2) -> Path:
    """Contact sheet of already-rendered PNGs (for the demo folder)."""
    from PIL import Image, ImageDraw

    ims = [(lbl, Image.open(p).convert("RGB")) for lbl, p in images]
    w = max(im.width for _, im in ims)
    h = max(im.height for _, im in ims)
    rows = (len(ims) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * (h + 24)), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for k, (lbl, im) in enumerate(ims):
        x, y = (k % cols) * w, (k // cols) * (h + 24)
        out.paste(im, (x, y + 24))
        d.text((x + 4, y + 6), lbl, fill=(0, 0, 0))
    out.save(path)
    return path
