"""Aesthetic feature map phi: grayscale image -> R^11.

Seven structural metrics from the aesthetic-optimization spec plus four
literature-backed additions (edge-orientation entropy, compression
complexity, luminance skewness, center-of-mass balance). Color metrics were
removed with color rendering. Each metric has a population target t and a
normalization scale s; per-metric loss is ((phi - t)/s)^2.

Measurement is two-scale (image and its 2x block-downsample averaged) to
penalize scale-fragile solutions, as in the spec.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    name: str
    target: float
    scale: float
    source: str


METRICS: list[MetricSpec] = [
    MetricSpec("beta_slope", 2.0, 0.4, "natural-scene 1/f^2 statistics"),
    MetricSpec("fractal_dim", 1.4, 0.15, "Taylor/Spehar preferred D ~ 1.3-1.5"),
    MetricSpec("entropy", 5.0, 1.2, "Berlyne inverted-U mid complexity"),
    MetricSpec("edge_density", 0.08, 0.05, "visual clutter penalty"),
    MetricSpec("gradient_gini", 0.75, 0.12, "sparse coding / processing fluency"),
    MetricSpec("symmetry", 0.30, 0.30, "mild mirror symmetry preferred"),
    MetricSpec("rms_contrast", 0.20, 0.07, "anti-washout"),
    MetricSpec("edge_orient_entropy", 0.95, 0.10, "Redies 2017: art near max EOE"),
    MetricSpec("compress_complexity", 0.50, 0.20, "Forsythe 2011 inverted-U midpoint"),
    MetricSpec("lum_skewness", 0.0, 0.6, "Graham/Redies: art has ~0 luminance skew"),
    MetricSpec("balance_dcm", 0.05, 0.08, "Hubner/Fillinger DCM ~ -0.84 with liking"),
]

METRIC_NAMES = [m.name for m in METRICS]
TARGETS = np.array([m.target for m in METRICS])
SCALES = np.array([m.scale for m in METRICS])
N_METRICS = len(METRICS)


# -- helpers ------------------------------------------------------------------

def _gradients(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gy, gx = np.gradient(L)
    return gx, gy


def _grad_mag(L: np.ndarray) -> np.ndarray:
    gx, gy = _gradients(L)
    return np.hypot(gx, gy)


def downsample2(L: np.ndarray) -> np.ndarray:
    """2x block-mean downsample of an (H, W) image."""
    h, w = L.shape[0] // 2 * 2, L.shape[1] // 2 * 2
    return L[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


# -- individual metrics ---------------------------------------------------------

def beta_slope(L: np.ndarray) -> float:
    """Slope beta of radial log power vs log f (power ~ 1/f^beta), mid-band."""
    n = min(L.shape)
    Lc = L[:n, :n] - L[:n, :n].mean()
    win = np.hanning(n)
    F = np.fft.fftshift(np.fft.fft2(Lc * np.outer(win, win)))
    power = np.abs(F) ** 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - n / 2, xx - n / 2).astype(int)
    radial = np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    fmax = n // 2
    lo, hi = max(2, int(0.01 * n)), max(4, int(0.35 * fmax))
    f = np.arange(len(radial))[lo:hi]
    p = radial[lo:hi]
    good = p > 0
    if good.sum() < 4:
        return 0.0
    slope = np.polyfit(np.log(f[good]), np.log(p[good]), 1)[0]
    return float(-slope)


def fractal_dim(L: np.ndarray) -> float:
    """Box-counting dimension of the thresholded gradient-edge map."""
    gm = _grad_mag(L)
    edges = gm > max(np.percentile(gm, 75), 1e-8)
    n = min(edges.shape)
    edges = edges[:n, :n]
    if not edges.any():
        return 0.0
    sizes, counts = [], []
    size = n // 2
    while size >= 2:
        k = n // size
        view = edges[: k * size, : k * size].reshape(k, size, k, size)
        counts.append(view.any(axis=(1, 3)).sum())
        sizes.append(size)
        size //= 2
    if len(sizes) < 2:
        return 0.0
    slope = np.polyfit(np.log(sizes), np.log(counts), 1)[0]
    return float(-slope)


def entropy(L: np.ndarray) -> float:
    """Shannon entropy (bits) of the 256-bin luminance histogram."""
    hist, _ = np.histogram(np.clip(L, 0, 1), bins=256, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(L: np.ndarray, threshold: float = 0.04) -> float:
    return float((_grad_mag(L) > threshold).mean())


def gradient_gini(L: np.ndarray) -> float:
    g = np.sort(_grad_mag(L).ravel())
    n = g.size
    total = g.sum()
    if total <= 0:
        return 0.0
    cum = np.cumsum(g)
    return float(1.0 - 2.0 * cum.sum() / (n * total) + 1.0 / n)


def symmetry(L: np.ndarray) -> float:
    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 1e-12 else 0.0

    return max(corr(L, L[:, ::-1]), corr(L, L[::-1, :]))


def rms_contrast(L: np.ndarray) -> float:
    return float(L.std())


def edge_orient_entropy(L: np.ndarray, bins: int = 24, top_n: int = 10000) -> float:
    """First-order edge-orientation entropy, normalized to [0, 1] (Redies 2017)."""
    gx, gy = _gradients(L)
    mag = np.hypot(gx, gy).ravel()
    ori = np.mod(np.arctan2(gy, gx).ravel(), np.pi)
    if mag.max() <= 1e-12:
        return 0.0
    if mag.size > top_n:
        idx = np.argpartition(mag, -top_n)[-top_n:]
        ori = ori[idx]
    hist, _ = np.histogram(ori, bins=bins, range=(0, np.pi))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))


def compress_complexity(L: np.ndarray) -> float:
    """zlib compression ratio of quantized luminance (Forsythe 2011 proxy)."""
    raw = (np.clip(L, 0, 1) * 255).astype(np.uint8).tobytes()
    return float(len(zlib.compress(raw, 9)) / max(len(raw), 1))


def lum_skewness(L: np.ndarray) -> float:
    x = L.ravel()
    sd = x.std()
    if sd < 1e-12:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / sd**3)


def balance_dcm(L: np.ndarray) -> float:
    """Deviation of the edge-energy center of mass from center, / half-diagonal."""
    w = _grad_mag(L)
    total = w.sum()
    h, wd = L.shape
    if total <= 1e-12:
        return 0.0
    yy, xx = np.mgrid[0:h, 0:wd]
    cy = (w * yy).sum() / total
    cx = (w * xx).sum() / total
    dist = np.hypot(cy - (h - 1) / 2, cx - (wd - 1) / 2)
    return float(dist / (0.5 * np.hypot(h, wd)))


# -- feature map ---------------------------------------------------------------

def features_single(L: np.ndarray) -> np.ndarray:
    """phi on one 2D grayscale image, order matches METRICS."""
    return np.array([
        beta_slope(L),
        fractal_dim(L),
        entropy(L),
        edge_density(L),
        gradient_gini(L),
        symmetry(L),
        rms_contrast(L),
        edge_orient_entropy(L),
        compress_complexity(L),
        lum_skewness(L),
        balance_dcm(L),
    ])


def features(L: np.ndarray, two_scale: bool = True) -> np.ndarray:
    """Two-scale phi: average over the image and its 2x downsample."""
    phi = features_single(L)
    if two_scale and min(L.shape) >= 128:
        phi = 0.5 * (phi + features_single(downsample2(L)))
    return phi
