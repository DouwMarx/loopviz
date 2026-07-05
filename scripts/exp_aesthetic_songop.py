"""Evolutionary aesthetic search over the pixel-exact song operator.

The parameter space the artwork actually has (at fixed print side n and
fixed song):

- rho = N/n (grid): rank fraction; sets the implied sample rate
  f = rho n^2 / T and the visual texture wavelength
- fractional attenuation of SVD components, priced by the closed-form
  playback error and projected into the tolerance budget
- the value -> ink map (symmetric clip percentile)

Constraints, checked per candidate: max per-window playback error <= tol
(guaranteed by projection) AND faithful looping - one full pass through
the song must return to window 1 (drift <= LOOP_TOL), because truncation
can be statically fine yet dynamically explosive (see exp_error_budget).

Loss: the 11 aesthetic metrics, equal weights, on the 1:1 matrix image.

Run: .venv/bin/python scripts/exp_aesthetic_songop.py [--song 2] [--n 1000]
         [--tol 1e-4] [--replicates 2] [--generations 12] [--population 10]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.config import OptConfig
from playlistviz.loss import equal_weights, scalar_loss
from playlistviz.matviz import gray
from playlistviz.metrics import METRICS, features
from playlistviz.optimize import EvalResult, run_es
from playlistviz.render import save_png
from playlistviz.sheet import make_sheet
from playlistviz.songmatrix import (build, decompose, load_audio,
                                    loop_degradation, plan, project_feasible,
                                    reconstruct)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_aesthetic_songop"
RHO_GRID = (0.5, 0.7, 0.85, 0.95)
LOOP_TOL = 1e-2
N_THETA = 7   # rho pick, tail (amax, center, width), head (amax, center), clip


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def theta_to_knobs(theta):
    u = sigmoid(np.asarray(theta, dtype=float))
    return {
        "rho_idx": min(int(u[0] * len(RHO_GRID)), len(RHO_GRID) - 1),
        # dead-zone: the lower third of the range maps to exactly zero, so
        # the ES can express "no attenuation at all" (any nonzero pattern
        # turns out to hurt loop stability far more than its static price)
        "tail_amax": max(0.0, (u[1] - 0.35) / 0.65),  # attenuate small-s end
        "tail_center": u[2],                  # ... from this index fraction
        "tail_width": 0.02 + 0.5 * u[3],
        "head_amax": max(0.0, (u[4] - 0.35) / 0.65),  # large-s end
        "head_center": 0.3 * u[5],
        "clip_pct": 97.0 + 2.95 * u[6],
    }


def attenuation_curve(k, N):
    x = np.arange(N) / max(N - 1, 1)          # 0 = largest s, 1 = smallest
    a = (k["tail_amax"] * sigmoid((x - k["tail_center"]) / k["tail_width"])
         + k["head_amax"] * sigmoid((k["head_center"] - x) / 0.05))
    return np.clip(a, 0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=int, default=2)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wav = ROOT / "data" / "audio" / f"{args.song:03d}.wav"
    signal, T = load_audio([wav])
    w_eq = equal_weights()

    print("caching operator decompositions per rho ...")
    cache = {}
    for rho in RHO_GRID:
        pl = plan(T, n=args.n, rho=rho)
        op, W = build(signal, pl)
        cache[rho] = (pl, decompose(op, W))
        print(f"  rho={rho:.2f}: N={pl.N} f={pl.f:.0f} Hz")

    def objective(theta):
        k = theta_to_knobs(theta)
        rho = RHO_GRID[k["rho_idx"]]
        pl, svd = cache[rho]
        a, err = project_feasible(svd, attenuation_curve(k, pl.N), args.tol)
        A = reconstruct(svd, a)
        loop = loop_degradation(A, svd.W, loops=1)[0]
        img = gray(A, pct=k["clip_pct"])
        phi = features(img)
        loss = scalar_loss(phi, w_eq)
        if not np.isfinite(loop):
            loss += 200.0   # diverges: hard reject
        elif loop > LOOP_TOL:
            # graded penalty so the ES can descend toward stability
            loss += 10.0 * (1.0 + np.log10(loop / LOOP_TOL))
        return EvalResult(theta=np.asarray(theta, float).copy(),
                          loss=float(loss), phi=phi, image=img)

    # exact-operator baselines per rho, default ink map
    baselines = {}
    for rho in RHO_GRID:
        pl, svd = cache[rho]
        img = gray(reconstruct(svd, np.zeros(pl.N)))
        baselines[rho] = float(scalar_loss(features(img), w_eq))
    print("exact baselines (equal weights): "
          + "  ".join(f"rho={r:.2f}: {l:.3f}" for r, l in baselines.items()))

    tiles, results = [], []
    for rep in range(args.replicates):
        cfg = OptConfig(generations=args.generations,
                        population=args.population,
                        subspace_rank=0, seed=100 + 37 * rep)
        best, hist = run_es(objective, cfg, n_params=N_THETA,
                            progress=lambda g, l, r=rep: print(
                                f"  rep {r} gen {g}: best {l:.3f}"))
        k = theta_to_knobs(best.theta)
        rho = RHO_GRID[k["rho_idx"]]
        pl, svd = cache[rho]
        a, err = project_feasible(svd, attenuation_curve(k, pl.N), args.tol)
        A = reconstruct(svd, a)
        loops = loop_degradation(A, svd.W, loops=3)
        img = gray(A, pct=k["clip_pct"])
        save_png(img, OUT / f"best_rep{rep}.png")
        save_png(img, OUT / f"best_rep{rep}_16bit.png", bit_depth=16)
        c = (args.n - 500) // 2
        tiles.append((f"rep {rep}: L={best.loss:.3f} rho={rho:.2f} "
                      f"clip={k['clip_pct']:.1f} err={err:.0e} "
                      f"loop3={loops[-1]:.0e} (1:1 crop)",
                      img[c:c + 500, c:c + 500]))
        results.append({
            "rep": rep, "loss": best.loss, "rho": rho, "N": pl.N,
            "f_hz": pl.f, "knobs": {kk: float(v) for kk, v in k.items()},
            "attenuation_mean": float(a.mean()),
            "attenuation_max": float(a.max()),
            "static_err": float(err), "loop_errs": loops,
            "evaluations": hist.evaluations,
            "phi": {m.name: float(v)
                    for m, v in zip(METRICS, best.phi)},
        })
        print(f"rep {rep}: loss {best.loss:.3f} (exact baseline "
              f"{baselines[rho]:.3f}) rho={rho:.2f} static err {err:.1e} "
              f"loop errs " + " ".join(f"{e:.1e}" for e in loops))

    lines = "\n".join(
        f"- rep {r['rep']}: loss {r['loss']:.3f}, rho={r['rho']:.2f} "
        f"(N={r['N']}, f={r['f_hz']:.0f} Hz), clip "
        f"{r['knobs']['clip_pct']:.1f}%, mean attenuation "
        f"{r['attenuation_mean']:.2e} (max {r['attenuation_max']:.2e}), "
        f"static err {r['static_err']:.1e}, loop errs "
        + ", ".join(f"{e:.1e}" for e in r["loop_errs"]) for r in results)
    base = "\n".join(f"- rho={r:.2f}: {l:.3f}" for r, l in baselines.items())
    make_sheet(tiles, OUT / "aesthetic_songop.png", tile_size=500,
               cols=min(args.replicates, 3), readme=f"""\
# exp_aesthetic_songop: ES over the pixel-exact operator's real freedoms

Song {wav.name} ({T:.1f} s), print side n={args.n}, playback tolerance
{args.tol:.0e}, loop constraint: one full pass must return to window 1
within {LOOP_TOL:.0e} (hard penalty - see exp_error_budget for why static
error alone is not enough).

Searched (7 thetas): rho in {RHO_GRID} | fractional SVD attenuation
(two sigmoid shoulders over the spectrum, projected into the tolerance
budget) | symmetric clip percentile of the ink map.

Results ({args.replicates} replicates, equal metric weights):
{lines}

Exact-operator baselines (a = 0, clip 99.5):
{base}

Interpretation guide: if winners sit at attenuation ~0 with only rho and
clip moving, then within a faithful-playback budget of {args.tol:.0e} the
picture is essentially DETERMINED by (song, rho, ink map) - the honest
aesthetic parameters are which song, the rank fraction, and the display
mapping, not hidden component surgery.
""")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}/aesthetic_songop.png")


if __name__ == "__main__":
    main()
