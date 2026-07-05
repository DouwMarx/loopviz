"""Bradley-Terry personalization of the metric weights.

Utility of candidate i is u_i = -w . loss_vec_i, so
P(i beats j) = sigmoid(w . (loss_vec_j - loss_vec_i)).

Fitting is logistic regression on loss-vector differences, L2-regularized
toward the uniform weight vector. Components fitted negative mean the
observer prefers *higher* loss on that metric (the population target or its
direction is wrong for them) - reported, not clipped away silently.

Active pair selection maximizes the expected information gain
p(1-p) * d^T H^{-1} d (D-optimal under the current fit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .metrics import N_METRICS


@dataclass
class Comparison:
    winner: str   # candidate id
    loser: str

    def to_json(self) -> str:
        return json.dumps({"winner": self.winner, "loser": self.loser})


def load_comparisons(path: Path) -> list[Comparison]:
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(Comparison(winner=d["winner"], loser=d["loser"]))
    return out


def append_comparison(path: Path, comp: Comparison) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(comp.to_json() + "\n")


@dataclass
class BTFit:
    w_raw: np.ndarray        # unconstrained fitted weights (can be negative)
    w_simplex: np.ndarray    # positive part normalized to the simplex
    log_likelihood: float
    hessian: np.ndarray      # of the negative log posterior, at the optimum
    n_comparisons: int

    def std_errors(self) -> np.ndarray:
        cov = np.linalg.pinv(self.hessian)
        return np.sqrt(np.maximum(np.diag(cov), 0.0))


def _design_matrix(comparisons: list[Comparison],
                   loss_vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Row per comparison: loss_vec(loser) - loss_vec(winner)."""
    rows = [loss_vectors[c.loser] - loss_vectors[c.winner] for c in comparisons]
    return np.asarray(rows)


def fit_bt(comparisons: list[Comparison], loss_vectors: dict[str, np.ndarray],
           l2: float = 1.0, n_metrics: int = N_METRICS) -> BTFit:
    """MAP fit of w. Prior: w ~ Normal(uniform, 1/(2*l2) I)."""
    if not comparisons:
        raise ValueError("no comparisons to fit")
    X = _design_matrix(comparisons, loss_vectors)  # (C, M); w.X > 0 = correct order
    w0 = np.full(n_metrics, 1.0 / n_metrics)

    def nlp(w):
        z = X @ w
        # log sigmoid, stable
        ll = -np.logaddexp(0.0, -z).sum()
        return -(ll) + l2 * ((w - w0) ** 2).sum()

    def grad(w):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-z))
        return -X.T @ (1.0 - p) + 2 * l2 * (w - w0)

    res = minimize(nlp, w0, jac=grad, method="L-BFGS-B")
    w = res.x
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))
    H = (X.T * (p * (1 - p))) @ X + 2 * l2 * np.eye(n_metrics)
    wp = np.maximum(w, 0.0)
    w_simplex = wp / wp.sum() if wp.sum() > 0 else np.full(n_metrics, 1.0 / n_metrics)
    ll = float(-np.logaddexp(0.0, -z).sum())
    return BTFit(w_raw=w, w_simplex=w_simplex, log_likelihood=ll,
                 hessian=H, n_comparisons=len(comparisons))


def predict_prob(fit: BTFit, loss_i: np.ndarray, loss_j: np.ndarray) -> float:
    """P(i beats j) under the fitted model."""
    z = fit.w_raw @ (loss_j - loss_i)
    return float(1.0 / (1.0 + np.exp(-z)))


def select_pairs(loss_vectors: dict[str, np.ndarray],
                 fit: BTFit | None,
                 n_pairs: int,
                 exclude: set[frozenset] | None = None,
                 rng: np.random.Generator | None = None,
                 blocks: dict[str, str] | None = None) -> list[tuple[str, str]]:
    """Choose informative pairs.

    With no fit yet: spread pairs by maximizing loss-vector distance (diverse
    candidates first). With a fit: D-optimal score p(1-p) * d^T H^{-1} d.

    blocks: optional candidate-id -> block label (e.g. which song the
    candidate encodes). Pairs are then drawn only WITHIN a block, so every
    choice compares parameter settings on identical content - a blocked
    design that keeps content preference out of the parameter weights.
    """
    rng = rng or np.random.default_rng(0)
    ids = sorted(loss_vectors)
    exclude = exclude or set()
    cands = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if blocks is not None and \
                    blocks.get(ids[a]) != blocks.get(ids[b]):
                continue
            key = frozenset((ids[a], ids[b]))
            if key in exclude:
                continue
            d = loss_vectors[ids[b]] - loss_vectors[ids[a]]
            if fit is None:
                score = float(np.linalg.norm(d))
            else:
                Hinv_d = np.linalg.solve(fit.hessian, d)
                z = fit.w_raw @ d
                p = 1.0 / (1.0 + np.exp(-z))
                score = float(p * (1 - p) * (d @ Hinv_d))
            cands.append((score, ids[a], ids[b]))
    cands.sort(reverse=True)
    pairs = [(a, b) for _, a, b in cands[:n_pairs]]
    # randomize left/right presentation
    return [(a, b) if rng.random() < 0.5 else (b, a) for a, b in pairs]
