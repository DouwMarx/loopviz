"""Render images of the playlist operator A without materializing it.

A = L R^T (rank N + q). Two linear-in-the-factors pixel statistics:

  block mean  : M_pq = mean_{i in p, j in q} A_ij = (pool L)_p . (pool R)_q
  block energy: E_pq = mean_{i in p, j in q} A_ij^2 = tr(G_p H_q) / (k_p k_q)

with per-block Grams G_p = sum_{i in p} L_i L_i^T (the N x N cross-Gram trick
from the spec). Cost scales with output pixels and rank^2, never with D^2.

Tone mapping and cosine palette turn the two fields into a print-ready RGB
image; all of that is controlled by the same theta as the Z generator, so the
optimizer shapes both operator content and its presentation.
"""

from __future__ import annotations

import numpy as np


def _trim_to_multiple(F: np.ndarray, P: int) -> tuple[np.ndarray, int]:
    D = F.shape[0]
    if P > D:
        raise ValueError(f"resolution {P} exceeds dimension {D}")
    k = D // P
    return F[: P * k], k


def pool_rows(F: np.ndarray, P: int) -> np.ndarray:
    """Block-mean the D rows of (D, r) down to P rows."""
    Ft, k = _trim_to_multiple(F, P)
    return Ft.reshape(P, k, -1).mean(axis=1)


def block_grams(F: np.ndarray, P: int) -> np.ndarray:
    """Per-block Grams: (P, r, r) with G[p] = sum_{i in block p} F_i F_i^T / k."""
    Ft, k = _trim_to_multiple(F, P)
    Fb = Ft.reshape(P, k, -1)
    return np.einsum("pkr,pks->prs", Fb, Fb) / k


def mean_image(L: np.ndarray, R: np.ndarray, P: int) -> np.ndarray:
    """Signed block-mean image of A = L R^T at resolution P x P."""
    return pool_rows(L, P) @ pool_rows(R, P).T


def energy_image(L: np.ndarray, R: np.ndarray, P: int) -> np.ndarray:
    """Block mean-square image of A = L R^T at resolution P x P."""
    G = block_grams(L, P).reshape(P, -1)
    H = block_grams(R, P).reshape(P, -1)
    return G @ H.T  # tr(G_p H_q) since both are symmetric


def _robust_unit(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi - lo < 1e-30:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def scalar_field(L: np.ndarray, R: np.ndarray, P: int, params: dict[str, float]) -> np.ndarray:
    """Combine energy and signed-mean statistics into one field in [0, 1]."""
    compress = 10.0 ** params["log_compress"]
    E = energy_image(L, R, P)
    E = np.log1p(compress * E / (np.abs(E).mean() + 1e-30)) / np.log1p(compress)
    E = _robust_unit(E)
    mix = params["energy_mix"]
    if mix < 1.0:
        M = _robust_unit(mean_image(L, R, P))
        t = mix * E + (1.0 - mix) * M
    else:
        t = E
    t = np.clip(t, 0.0, 1.0) ** params["gamma"]

    v = params["vignette"]
    if v > 0:
        yy, xx = np.mgrid[0:P, 0:P]
        r2 = ((yy - (P - 1) / 2) ** 2 + (xx - (P - 1) / 2) ** 2) / (2 * ((P - 1) / 2) ** 2)
        t = t * (1.0 - v * r2)
    return t


def cosine_palette(t: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """rgb = a + b cos(2 pi (c t + d)) per channel, clipped to [0, 1]."""
    out = np.empty(t.shape + (3,))
    for i, ch in enumerate("rgb"):
        a, b = params[f"pal_a_{ch}"], params[f"pal_b_{ch}"]
        c, d = params[f"pal_c_{ch}"], params[f"pal_d_{ch}"]
        out[..., i] = a + b * np.cos(2 * np.pi * (c * t + d))
    return np.clip(out, 0.0, 1.0)


def render(L: np.ndarray, R: np.ndarray, P: int, params: dict[str, float]) -> np.ndarray:
    """Full render: factored A -> (P, P, 3) float RGB in [0, 1]."""
    return cosine_palette(scalar_field(L, R, P, params), params)


def save_png(img: np.ndarray, path, bit_depth: int = 8) -> None:
    """Write an RGB float image to PNG (8- or 16-bit for print)."""
    from PIL import Image

    img = np.clip(img, 0.0, 1.0)
    if bit_depth == 16:
        arr = (img * 65535).astype("<u2")
        # Pillow lacks native 16-bit RGB; write via raw pypng-style encoder
        _write_png16(arr, path)
    else:
        Image.fromarray((img * 255).astype(np.uint8), mode="RGB").save(path)


def _write_png16(arr: np.ndarray, path) -> None:
    """Minimal 16-bit RGB PNG writer (big-endian samples per PNG spec)."""
    import struct
    import zlib

    h, w, _ = arr.shape
    be = arr.astype(">u2")
    raw = b"".join(b"\x00" + be[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
