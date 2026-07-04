"""Embed an arbitrary image into the playlist operator.

1. ES-optimize the 2D procedural generator (targets.py) directly against
   the fitted BT weights - no operator in the loop, so evaluations are
   ~100x cheaper than Z-space ES.
2. Embed the winning target image into A's free part (embed.py).
3. Verify: playback error unchanged, achieved picture matches the target.

Output sheet: target | achieved picture of A | absolute error (amplified).

Run: .venv/bin/python scripts/exp_embed.py [--rank 160] [--res 384]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.config import OptConfig
from playlistviz.embed import display, embed_image
from playlistviz.ingest import load_matrix
from playlistviz.loss import scalar_loss
from playlistviz.metrics import features
from playlistviz.operator import PlaylistOperator
from playlistviz.optimize import EvalResult, run_es
from playlistviz.render import save_png
from playlistviz.targets import N_PARAMS_2D, generate_target, theta2d_to_params

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_embed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=160)
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    weights_file = ROOT / "runs" / "fitted_weights.json"
    w = np.asarray(json.loads(weights_file.read_text())["w_simplex"])

    # -- step 1: optimize the 2D generator directly (fast) -------------------
    def objective(theta: np.ndarray) -> EvalResult:
        img = generate_target(theta2d_to_params(theta), args.res)
        phi = features(img)
        return EvalResult(theta=theta.copy(), loss=scalar_loss(phi, w), phi=phi)

    t0 = time.time()
    best, hist = run_es(objective,
                        OptConfig(generations=14, population=12,
                                  seed=args.seed, subspace_rank=0),
                        n_params=N_PARAMS_2D)
    t_es = time.time() - t0
    print(f"2D ES: {hist.evaluations} evals in {t_es:.0f}s "
          f"({1000 * t_es / hist.evaluations:.0f} ms/eval), "
          f"loss {hist.best_loss[0]:.3f} -> {best.loss:.3f}")

    T = generate_target(theta2d_to_params(best.theta), args.res)
    save_png(T, OUT / "target.png")

    # -- step 2: embed into the operator -------------------------------------
    X, _ = load_matrix(ROOT / "data" / "songs.npz")
    op = PlaylistOperator.from_songs(X)
    t0 = time.time()
    res = embed_image(op, T, rank=args.rank)
    t_embed = time.time() - t0

    achieved_img = display(res.achieved)
    save_png(achieved_img, OUT / "achieved.png")

    # -- step 3: verify -------------------------------------------------------
    err_play = op.playback_error(U=res.U, Vp=res.Vp, scale=res.scale)
    phi = features(achieved_img)
    print(f"embed: rank {res.rank}, {t_embed:.0f}s, "
          f"image rel error {res.rel_error:.2%}")
    print(f"playback error with embedded image: {err_play:.2e}")
    print(f"personal loss: target {scalar_loss(features(T), w):.3f} "
          f"-> achieved picture {scalar_loss(phi, w):.3f}")

    err_map = np.abs(res.achieved - res.target_amp)
    err_map = err_map / max(err_map.max(), 1e-30)
    P = args.res
    sheet = Image.new("L", (3 * (P + 10), P + 10), 30)
    for k, img in enumerate([T, achieved_img, err_map]):
        sheet.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "L"),
                    (k * (P + 10) + 5, 5))
    sheet.save(OUT / "side_by_side.png")
    print(f"wrote {OUT}/side_by_side.png (target | achieved | error map)")


if __name__ == "__main__":
    main()
