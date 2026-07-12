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
           l2: float = 1.0, n_metrics: int | None = None,
           prior_mean: np.ndarray | None = None) -> BTFit:
    """MAP fit of w. Prior: w ~ Normal(prior_mean, 1/(2*l2) I).

    Feature dimension is inferred from the loss vectors (they may carry
    extra non-metric features, e.g. generation knobs, appended after the
    metric losses). Default prior mean: uniform 1/M over all features.
    """
    if not comparisons:
        raise ValueError("no comparisons to fit")
    X = _design_matrix(comparisons, loss_vectors)  # (C, M); w.X > 0 = correct order
    if n_metrics is None:
        n_metrics = X.shape[1]
    w0 = (np.asarray(prior_mean, dtype=float) if prior_mean is not None
          else np.full(n_metrics, 1.0 / n_metrics))

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


# -- quadratic-interaction utility model ---------------------------------------
#
# The linear fit above scores fixed-target losses ((phi-t)/s)^2, so the
# observer's optimum is pinned to the population target per metric. The
# quadratic model frees it: utility(phi) = theta . [z, z^2, z_i z_j] with
# z the pool-standardized metrics, i.e. per-metric curvature with fitted
# optima plus all pairwise interactions. Held out, it predicts choices
# clearly better (0.65 vs 0.59 accuracy at 2k comparisons); the linear fit
# is kept because its Hessian drives D-optimal pair selection.

def _quad_features(z: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(z), 1)
    return np.concatenate([z, z * z, np.outer(z, z)[iu]])


def quad_fit(comparisons: list[Comparison], phi: dict[str, np.ndarray],
             l2: float = 3.0) -> dict:
    """MAP fit of the quadratic-interaction utility. Returns a
    self-contained JSON-ready dict (standardization baked in)."""
    if not comparisons:
        raise ValueError("no comparisons to fit")
    pool = np.asarray(list(phi.values()))
    mu, sd = pool.mean(axis=0), pool.std(axis=0)
    sd[sd < 1e-9] = 1.0
    feat = {cid: _quad_features((v - mu) / sd) for cid, v in phi.items()}
    # winner - loser: theta . feat is a utility (higher = preferred)
    X = np.asarray([feat[c.winner] - feat[c.loser] for c in comparisons])
    theta = np.zeros(X.shape[1])
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(X @ theta)))
        g = X.T @ (1.0 - p) - 2 * l2 * theta
        H = -(X.T * (p * (1 - p))) @ X - 2 * l2 * np.eye(len(theta))
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-10:
            break
    ll = float(-np.logaddexp(0.0, -(X @ theta)).sum())
    stderr = np.sqrt(np.diag(np.linalg.inv(-H)))
    return {
        "mu": [float(v) for v in mu],
        "sd": [float(v) for v in sd],
        "theta": [float(v) for v in theta],
        "stderr": [float(v) for v in stderr],
        "l2": l2,
        "log_likelihood": ll,
        "n_comparisons": len(comparisons),
    }


def quad_utility(phi_vec: np.ndarray, model: dict) -> float:
    """Utility (higher = preferred) of a metric vector under a quad_fit."""
    z = (np.asarray(phi_vec) - np.asarray(model["mu"])) / np.asarray(model["sd"])
    return float(np.asarray(model["theta"]) @ _quad_features(z))


def quad_term_names(metric_names: list[str]) -> list[str]:
    iu = np.triu_indices(len(metric_names), 1)
    return (list(metric_names)
            + [f"{n}^2" for n in metric_names]
            + [f"{metric_names[i]} x {metric_names[j]}"
               for i, j in zip(*iu)])


def scorer_from_weights(weights: dict):
    """phi-dict -> score (lower = better) from a fitted_weights.json dict.

    Prefers the quadratic-interaction model when present; falls back to
    w_raw on the fixed-target loss vector for older weight files.
    """
    from .loss import loss_vector_from_phi_dict
    from .metrics import METRIC_NAMES

    if "quad" in weights:
        model = weights["quad"]
        def score(phi_dict: dict[str, float]) -> float:
            v = np.array([phi_dict[n] for n in METRIC_NAMES])
            return -quad_utility(v, model)
    else:
        w = np.asarray(weights["w_raw"])
        def score(phi_dict: dict[str, float]) -> float:
            return float(w @ loss_vector_from_phi_dict(phi_dict))
    return score


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
