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


def equal_weights() -> np.ndarray:
    return np.full(N_METRICS, 1.0 / N_METRICS)


def sample_weights(alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Dirichlet(alpha * 1) draw on the simplex; alpha=inf -> equal weights."""
    if not np.isfinite(alpha):
        return equal_weights()
    return rng.dirichlet(np.full(N_METRICS, alpha))


# The Dirichlet annealing schedule from the spec: (alpha, number of runs)
DEFAULT_SCHEDULE: list[tuple[float, int]] = [
    (float("inf"), 1),  # equal-weight baseline
    (8.0, 2),           # mild variance
    (2.0, 3),           # moderate
    (0.5, 3),           # near-corner
]
