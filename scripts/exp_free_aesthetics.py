"""Baseline: pure aesthetic optimization with NO music constraint.

The image is rendered from free factors L = U, R = V (no A0, no P_perp),
same generator, same renderer, same metrics, same ES budget as a real run.
This shows what the aesthetic stack can reach when the operator does not
have to reproduce the playlist - the reference point for judging how much
the music constraint costs visually.

Flags:
  --negate      sign-flip sanity check: MAXIMIZE the loss. The result should
                look conspicuously bad; if it doesn't, the metrics are vacuous.
  --rank Q      Z rank (default 48)
  --seed S      ES seed (default 7)
  --tag NAME    output subdirectory suffix

Run: .venv/bin/python scripts/exp_free_aesthetics.py [flags]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.config import OptConfig, ZConfig
from playlistviz.loss import equal_weights, loss_vector, scalar_loss
from playlistviz.metrics import METRIC_NAMES, features
from playlistviz.optimize import EvalResult, run_es
from playlistviz.render import render, save_png
from playlistviz.zspace import generate_Z, theta_to_params

ROOT = Path(__file__).parent.parent
D = 96000  # ambient dimension; visual character is D-insensitive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negate", action="store_true")
    ap.add_argument("--rank", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    name = "exp_free_negated" if args.negate else "exp_free_baseline"
    if args.tag:
        name += f"_{args.tag}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    zcfg = ZConfig(rank=args.rank)
    w = equal_weights()
    sign = -1.0 if args.negate else 1.0

    def objective(theta: np.ndarray) -> EvalResult:
        params = theta_to_params(theta)
        zf = generate_Z(params, D, zcfg)  # unprojected: no music constraint
        img = render(zf.U, zf.Vp, 384, params, stride=4)
        phi = features(img)
        return EvalResult(theta=theta.copy(),
                          loss=sign * scalar_loss(phi, w), phi=phi)

    t0 = time.time()
    best, hist = run_es(
        objective, OptConfig(generations=14, population=12, seed=args.seed),
        progress=lambda g, l: print(f"  gen {g:2d}  loss {l:.3f}"))
    print(f"ES finished in {time.time() - t0:.0f}s, "
          f"loss {hist.best_loss[0]:.3f} -> {best.loss:.3f}")

    params = theta_to_params(best.theta)
    zf = generate_Z(params, D, zcfg)
    img = render(zf.U, zf.Vp, 1536, params)
    phi = features(img)
    loss_eq = scalar_loss(phi, w)
    save_png(img, out / "presentation.png")
    (out / "candidate.json").write_text(json.dumps({
        "id": name,
        "z_rank": zcfg.rank,
        "es_seed": args.seed,
        "negated": args.negate,
        "alpha": None,
        "weights": list(map(float, w)),
        "theta": list(map(float, best.theta)),
        "phi": {n: float(v) for n, v in zip(METRIC_NAMES, phi)},
        "loss_vector": list(map(float, loss_vector(phi))),
        "loss_eq": float(loss_eq),
    }, indent=2))
    kind = "negated (should look BAD)" if args.negate else "unconstrained"
    (out / "README.md").write_text(f"""\
# {name}

What is investigated: the aesthetic stack with NO music constraint
(free factors, no A0, no P_perp) - the reference for what the playlist
constraint costs visually.{'''
This run MAXIMIZES the loss (sign flip): if the metrics mean anything,
the result should look conspicuously bad (it comes out as formless static).''' if args.negate else ''}

Result: L_eq = {loss_eq:.3f} (rank {zcfg.rank}, ES seed {args.seed}).
""")
    print(f"{kind} L_eq = {loss_eq:.3f}  ->  {out}/presentation.png")


if __name__ == "__main__":
    main()
