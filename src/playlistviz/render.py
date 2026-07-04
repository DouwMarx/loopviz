"""Render grayscale images of the playlist operator A without materializing it.

A = L R^T (rank N + q). Two linear-in-the-factors pixel statistics:

  block mean  : M_pq = mean_{i in p, j in q} A_ij = (pool L)_p . (pool R)_q
  block energy: E_pq = mean_{i in p, j in q} A_ij^2 = tr(G_p H_q) / (k_p k_q)

with per-block Grams G_p = sum_{i in p} L_i L_i^T (the cross-Gram trick from
the spec). Cost scales with output pixels and rank^2, never with D^2.

`stride` subsamples rows within each block (every stride-th audio sample)
for cheap approximate renders inside the ES loop; final renders use stride 1.

Tone mapping (log compression, gamma, a free nonmonotonic tone curve,
vignette) turns the scalar field into the printed grayscale image. Output is
a 2D float array in [0, 1].
"""

from __future__ import annotations

import numpy as np


def _blocks(F: np.ndarray, P: int, stride: int = 1) -> np.ndarray:
    """View (D, r) as (P, k', r) blocks, optionally strided within blocks."""
    D = F.shape[0]
    if P > D:
        raise ValueError(f"resolution {P} exceeds dimension {D}")
    k = D // P
    Fb = F[: P * k].reshape(P, k, -1)
    if stride > 1 and stride < k:
        Fb = Fb[:, ::stride]
    return Fb


def pool_rows(F: np.ndarray, P: int, stride: int = 1) -> np.ndarray:
    """Block-mean the D rows of (D, r) down to P rows."""
    return _blocks(F, P, stride).mean(axis=1)


def block_grams(F: np.ndarray, P: int, stride: int = 1) -> np.ndarray:
    """Per-block Grams: (P, r, r) with G[p] = mean_{i in block p} F_i F_i^T."""
    Fb = _blocks(F, P, stride)
    return np.einsum("pkr,pks->prs", Fb, Fb) / Fb.shape[1]


def mean_image(L: np.ndarray, R: np.ndarray, P: int, stride: int = 1) -> np.ndarray:
    """Signed block-mean image of A = L R^T at resolution P x P."""
    return pool_rows(L, P, stride) @ pool_rows(R, P, stride).T


def energy_image(L: np.ndarray, R: np.ndarray, P: int, stride: int = 1) -> np.ndarray:
    """Block mean-square image of A = L R^T at resolution P x P."""
    G = block_grams(L, P, stride).reshape(P, -1)
    H = block_grams(R, P, stride).reshape(P, -1)
    return G @ H.T  # tr(G_p H_q) since both are symmetric


def _robust_unit(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi - lo < 1e-30:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def scalar_field(L: np.ndarray, R: np.ndarray, P: int, params: dict[str, float],
                 stride: int = 1) -> np.ndarray:
    """Combine energy and signed-mean statistics into one field in [0, 1]."""
    compress = 10.0 ** params["log_compress"]
    E = energy_image(L, R, P, stride)
    E = np.log1p(compress * E / (np.abs(E).mean() + 1e-30)) / np.log1p(compress)
    E = _robust_unit(E)
    mix = params["energy_mix"]
    if mix < 1.0:
        M = _robust_unit(mean_image(L, R, P, stride))
        t = mix * E + (1.0 - mix) * M
    else:
        t = E
    return np.clip(t, 0.0, 1.0)


def tone_map(t: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Gamma + free nonmonotonic tone curve + vignette, output in [0, 1]."""
    y = t ** params["gamma"]
    for i in (1, 2):
        b, c, d = params[f"tone_b{i}"], params[f"tone_c{i}"], params[f"tone_d{i}"]
        y = y + b * np.cos(2 * np.pi * (c * t + d))
    y = _robust_unit(y, 0.0, 100.0)

    v = params["vignette"]
    if v > 0:
        P = y.shape[0]
        yy, xx = np.mgrid[0:P, 0:P]
        r2 = ((yy - (P - 1) / 2) ** 2 + (xx - (P - 1) / 2) ** 2) / (2 * ((P - 1) / 2) ** 2)
        y = y * (1.0 - v * r2)
    return y


def render(L: np.ndarray, R: np.ndarray, P: int, params: dict[str, float],
           stride: int = 1) -> np.ndarray:
    """Full render: factored A -> (P, P) grayscale float in [0, 1]."""
    return tone_map(scalar_field(L, R, P, params, stride), params)


def save_png(img: np.ndarray, path, bit_depth: int = 8) -> None:
    """Write a grayscale float image to PNG (8- or 16-bit for print)."""
    from PIL import Image

    img = np.clip(img, 0.0, 1.0)
    if img.ndim != 2:
        raise ValueError("expected 2D grayscale image")
    if bit_depth == 16:
        _write_png16_gray((img * 65535).astype("<u2"), path)
    else:
        Image.fromarray((img * 255).astype(np.uint8), mode="L").save(path)


def _write_png16_gray(arr: np.ndarray, path) -> None:
    """Minimal 16-bit grayscale PNG writer (big-endian per PNG spec)."""
    import struct
    import zlib

    h, w = arr.shape
    be = arr.astype(">u2")
    raw = b"".join(b"\x00" + be[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)
