"""Baseline: pure aesthetic optimization with NO music constraint.

The image is rendered from free factors L = U, R = V (no A0, no P_perp),
same generator, same renderer, same metrics, same ES budget as a real run.
This shows what the aesthetic stack can reach when the operator does not
have to reproduce the playlist - the reference point for judging how much
the music constraint costs visually.

Run: .venv/bin/python scripts/exp_free_aesthetics.py [--color]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.config import OptConfig, ZConfig
from playlistviz.loss import equal_weights, loss_vector, metric_mask, scalar_loss
from playlistviz.metrics import METRIC_NAMES, features
from playlistviz.optimize import EvalResult, run_es
from playlistviz.render import render, save_png
from playlistviz.zspace import generate_Z, theta_to_params

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_free_baseline"
D = 96000  # same ambient dimension as the real operator


def main() -> None:
    gray = "--color" not in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    zcfg = ZConfig(rank=32)
    w = equal_weights(metric_mask(gray))

    def objective(theta: np.ndarray) -> EvalResult:
        params = theta_to_params(theta)
        zf = generate_Z(params, D, zcfg)  # unprojected: no music constraint
        img = render(zf.U, zf.Vp, 384, params, gray=gray)
        phi = features(img)
        return EvalResult(theta=theta.copy(), loss=scalar_loss(phi, w), phi=phi)

    t0 = time.time()
    best, hist = run_es(objective, OptConfig(generations=14, population=12, seed=7),
                        progress=lambda g, l: print(f"  gen {g:2d}  loss {l:.3f}"))
    print(f"ES finished in {time.time() - t0:.0f}s, "
          f"loss {hist.best_loss[0]:.3f} -> {best.loss:.3f}")

    params = theta_to_params(best.theta)
    zf = generate_Z(params, D, zcfg)
    img = render(zf.U, zf.Vp, 1536, params, gray=gray)
    phi = features(img)
    loss_eq = scalar_loss(phi, w)
    save_png(img, OUT / "presentation.png")
    (OUT / "candidate.json").write_text(json.dumps({
        "id": "free_baseline",
        "mode": "gray" if gray else "color",
        "z_rank": zcfg.rank,
        "es_seed": 7,
        "alpha": None,
        "weights": list(map(float, w)),
        "theta": list(map(float, best.theta)),
        "phi": {n: float(v) for n, v in zip(METRIC_NAMES, phi)},
        "loss_vector": list(map(float, loss_vector(phi))),
        "loss_eq": float(loss_eq),
    }, indent=2))
    print(f"unconstrained L_eq = {loss_eq:.3f}  ->  {OUT}/presentation.png")
    print("(compare against the constrained candidates in `playlistviz report`)")


if __name__ == "__main__":
    main()
