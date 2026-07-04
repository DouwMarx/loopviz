"""Scalarized aesthetic loss, barriers, and weight sampling.

L_w(theta) = sum_i w_i * ((phi_i - t_i)/s_i)^2 + barrier(phi)          (sum)
L_w(theta) = max_i w_i * l_i + 0.05 * sum_i l_i + barrier(phi)   (chebyshev)

The augmented weighted-Chebyshev form reaches non-convex parts of the Pareto
front that weighted sums cannot (standard multi-criteria result).

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


def loss_vector_from_phi_dict(phi: dict[str, float]) -> np.ndarray:
    """Loss vector from a stored phi dict (by name), tolerant of candidates
    saved under older metric sets (extra names ignored, all current names
    required)."""
    values = np.array([phi[name] for name in METRIC_NAMES])
    return loss_vector(values)


def barrier(phi: np.ndarray) -> float:
    """Quadratic penalties outside hard perceptual floors."""
    pen = 0.0
    contrast = phi[_IDX["rms_contrast"]]
    if contrast < 0.10:
        pen += (100.0 * (0.10 - contrast)) ** 2
    ent = phi[_IDX["entropy"]]
    if ent < 3.0:
        pen += (10.0 * (3.0 - ent)) ** 2
    return float(pen)


def scalar_loss(phi: np.ndarray, w: np.ndarray,
                scalarization: str = "sum") -> float:
    lv = loss_vector(phi)
    if scalarization == "sum":
        core = float(w @ lv)
    elif scalarization == "chebyshev":
        core = float(np.max(w * lv) + 0.05 * lv.sum())
    else:
        raise ValueError(f"unknown scalarization {scalarization!r}")
    return core + barrier(phi)


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
