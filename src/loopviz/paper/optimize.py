"""(1 + lambda) evolution strategy over theta, optionally in a random
low-rank subspace (theta = A_proj z), following the aesthetic-optimization
spec: elitist, sigma annealed multiplicatively per generation.

The objective evaluates: theta -> Z factors -> rendered image of A -> phi
-> scalarized loss under a fixed weight vector w. A0 is untouched, so every
candidate plays the playlist exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import OptConfig, RenderConfig, ZConfig
from ..operator import PlaylistOperator
from .loss import scalar_loss
from .metrics import features
from .render import render
from .zspace import N_PARAMS, generate_Z, theta_to_params


@dataclass
class EvalResult:
    theta: np.ndarray
    loss: float
    phi: np.ndarray
    image: np.ndarray | None = None


@dataclass
class ESHistory:
    best_loss: list[float] = field(default_factory=list)
    evaluations: int = 0


def make_objective(op: PlaylistOperator, w: np.ndarray, zcfg: ZConfig,
                   resolution: int, stride: int = 1,
                   scalarization: str = "sum",
                   ) -> Callable[[np.ndarray], EvalResult]:
    """Bind the full theta -> loss pipeline for a fixed operator and weights."""

    def objective(theta: np.ndarray) -> EvalResult:
        params = theta_to_params(theta)
        zf = generate_Z(params, op.D, zcfg, project_perp=op.project_perp)
        L, R = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)
        img = render(L, R, resolution, params, stride=stride)
        phi = features(img)
        return EvalResult(theta=theta.copy(),
                          loss=scalar_loss(phi, w, scalarization),
                          phi=phi, image=img)

    return objective


def run_es(objective: Callable[[np.ndarray], EvalResult],
           cfg: OptConfig,
           n_params: int = N_PARAMS,
           progress: Callable[[int, float], None] | None = None,
           ) -> tuple[EvalResult, ESHistory]:
    """Elitist (1+lambda)-ES. Returns the best evaluation and history."""
    rng = np.random.default_rng(cfg.seed)

    k = cfg.subspace_rank
    if k and k < n_params:
        A_proj = rng.standard_normal((n_params, k)) / np.sqrt(k)
        expand = lambda z: A_proj @ z
        dim = k
    else:
        expand = lambda z: z
        dim = n_params

    z_best = rng.standard_normal(dim) * 0.5
    best = objective(expand(z_best))
    hist = ESHistory(best_loss=[best.loss], evaluations=1)

    sigma = cfg.sigma0
    for gen in range(cfg.generations):
        for _ in range(cfg.population):
            z_cand = z_best + sigma * rng.standard_normal(dim)
            cand = objective(expand(z_cand))
            hist.evaluations += 1
            if cand.loss < best.loss:
                best, z_best = cand, z_cand
        hist.best_loss.append(best.loss)
        if progress is not None:
            progress(gen, best.loss)
        sigma *= cfg.sigma_decay

    return best, hist
