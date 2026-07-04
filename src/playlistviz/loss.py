"""Scalarized aesthetic loss, barriers, and weight sampling.

L_w(theta) = sum_i w_i * ((phi_i - t_i)/s_i)^2 + barrier(phi)

Barriers are weight-independent guards against reward hacking (near-flat
images gaming single metrics), per the spec's empirical findings.
"""

from __future__ import annotations

import numpy as np

from .metrics import METRIC_NAMES, N_METRICS, SCALES, TARGETS

_IDX = {name: i for i, name in enumerate(METRIC_NAMES)}


def loss_vector(phi: np.ndarray) -> np.ndarray:
    """Per-metric squared normalized residuals, shape (N_METRICS,)."""
    return ((phi - TARGETS) / SCALES) ** 2


def barrier(phi: np.ndarray) -> float:
    """Quadratic penalties outside hard perceptual floors/ceilings."""
    pen = 0.0
    contrast = phi[_IDX["rms_contrast"]]
    if contrast < 0.10:
        pen += (100.0 * (0.10 - contrast)) ** 2
    ent = phi[_IDX["entropy"]]
    if ent < 3.0:
        pen += (10.0 * (3.0 - ent)) ** 2
    cf = phi[_IDX["colorfulness"]]
    if cf > 95.0:
        pen += (0.5 * (cf - 95.0)) ** 2
    return float(pen)


def scalar_loss(phi: np.ndarray, w: np.ndarray) -> float:
    return float(w @ loss_vector(phi) + barrier(phi))


# metrics that are meaningful on a grayscale image (color metrics excluded:
# colorfulness, hue_dispersion, mean_saturation)
GRAY_METRICS = [n for n in METRIC_NAMES
                if n not in ("colorfulness", "hue_dispersion", "mean_saturation")]
GRAY_IDX = np.array([_IDX[n] for n in GRAY_METRICS])


def metric_mask(gray: bool) -> np.ndarray:
    """Boolean mask of active metrics for the given mode."""
    mask = np.ones(N_METRICS, dtype=bool)
    if gray:
        mask[:] = False
        mask[GRAY_IDX] = True
    return mask


def equal_weights(mask: np.ndarray | None = None) -> np.ndarray:
    """Uniform weights over active metrics (zeros elsewhere), summing to 1."""
    if mask is None:
        return np.full(N_METRICS, 1.0 / N_METRICS)
    w = np.zeros(N_METRICS)
    w[mask] = 1.0 / mask.sum()
    return w


def sample_weights(alpha: float, rng: np.random.Generator,
                   mask: np.ndarray | None = None) -> np.ndarray:
    """Dirichlet(alpha * 1) draw on the (masked) simplex; alpha=inf -> equal."""
    if not np.isfinite(alpha):
        return equal_weights(mask)
    if mask is None:
        return rng.dirichlet(np.full(N_METRICS, alpha))
    w = np.zeros(N_METRICS)
    w[mask] = rng.dirichlet(np.full(int(mask.sum()), alpha))
    return w


# The Dirichlet annealing schedule from the spec: (alpha, number of runs)
DEFAULT_SCHEDULE: list[tuple[float, int]] = [
    (float("inf"), 1),  # equal-weight baseline
    (8.0, 2),           # mild variance
    (2.0, 3),           # moderate
    (0.5, 3),           # near-corner
]
