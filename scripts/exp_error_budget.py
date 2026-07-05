"""What does an error budget buy? Trading playback exactness for freedom.

The exact operator A0 = sum_i s_i u_i v_i^T has N components. Discarding
component i costs each window k exactly s_i |v_i . w_k| of playback error
- so a tolerance between 1e-14 and 1e-1 is a BUDGET that can be spent on
deleting components. This script spends it two ways (smallest-s first,
largest-s first) at several tolerances and shows: how many components the
budget buys, what the picture looks like, and how the loop degrades over
repeated passes through the song.

Run: .venv/bin/python scripts/exp_error_budget.py [--song 2] [--n 1000]
                                                  [--rho 0.95]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.matviz import gray
from playlistviz.render import save_png
from playlistviz.sheet import make_sheet
from playlistviz.songmatrix import (attenuation_errors, build, decompose,
                                    load_audio, loop_degradation, plan,
                                    reconstruct)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_error_budget"
TOLS = (1e-8, 1e-4, 1e-2, 1e-1)


def max_discardable(svd, order: np.ndarray, tol: float) -> np.ndarray:
    """Discard components in the given order while max window err <= tol.

    Errors add in quadrature, so cumulative cost is monotone in the count
    - binary search the largest feasible prefix.
    """
    lo, hi = 0, len(order)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        a = np.zeros(len(svd.s))
        a[order[:mid]] = 1.0
        if attenuation_errors(svd, a).max() <= tol:
            lo = mid
        else:
            hi = mid - 1
        if lo == hi:
            break
    a = np.zeros(len(svd.s))
    a[order[:lo]] = 1.0
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=int, default=2)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--rho", type=float, default=0.95)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wav = ROOT / "data" / "audio" / f"{args.song:03d}.wav"
    signal, T = load_audio([wav])
    pl = plan(T, n=args.n, rho=args.rho)
    op, W = build(signal, pl)
    svd = decompose(op, W)
    N = len(svd.s)
    print(f"operator n={pl.n} N={N} f={pl.f:.0f} Hz  "
          f"s range [{svd.s[-1]:.2e}, {svd.s[0]:.2e}]")

    c = (args.n - 500) // 2
    sl = slice(c, c + 500)
    tiles = [(f"exact (tol 0)  rank {N}",
              gray(reconstruct(svd, np.zeros(N)))[sl, sl])]
    rows = []
    orders = {"smallest-first": np.arange(N)[::-1],
              "largest-first": np.arange(N)}
    for name, order in orders.items():
        for tol in TOLS:
            a = max_discardable(svd, order, tol)
            k = int(a.sum())
            A = reconstruct(svd, a)
            err = attenuation_errors(svd, a).max()
            loops = loop_degradation(A, W, loops=3)
            img = gray(A)
            save_png(img, OUT / f"A_{name}_tol{tol:.0e}.png")
            tiles.append((f"{name} tol={tol:.0e}: dropped {k}/{N}  "
                          f"loop3 err {loops[-1]:.1e}", img[sl, sl]))
            rows.append({"order": name, "tol": tol, "dropped": k,
                         "max_window_err": float(err),
                         "loop_err": loops})
            print(f"  {name:15s} tol={tol:.0e}: dropped {k:4d}/{N}  "
                  f"window err {err:.1e}  loop errs "
                  + " ".join(f"{e:.1e}" for e in loops))

    stat = "\n".join(
        f"- {r['order']}, tol {r['tol']:.0e}: dropped {r['dropped']}/{N}, "
        f"max window err {r['max_window_err']:.1e}, loop errs "
        + ", ".join(f"{e:.1e}" for e in r["loop_err"]) for r in rows)
    make_sheet(tiles, OUT / "error_budget.png", tile_size=400, cols=3, readme=f"""\
# exp_error_budget: trading playback exactness for visual freedom

Operator: song {wav.name}, n={pl.n}, N={N}, rho={pl.rho:.2f},
f={pl.f:.0f} Hz. A0 = sum s_i u_i v_i^T; discarding component i costs
window k exactly s_i |v_i . w_k| of playback error (errors add in
quadrature). A tolerance is therefore a budget; this experiment spends it
greedily from the small-s end and from the large-s end at tol in {TOLS}.

{stat}

Readings:
- how many components each budget buys, from each end, and what the
  picture looks like (sheet shows 500px 1:1 crops; full images saved).
- loop errs = drift from window 1 after 1, 2, 3 full passes through the
  song: the exact operator acts as a cyclic shift on window space
  (eigenvalues on the unit circle - loops forever); approximation makes
  each pass drift by ~ the per-step error, so tol 1e-4 stays inaudible
  for many loops while tol 1e-1 audibly degrades within a pass.
""")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT}/error_budget.png")


if __name__ == "__main__":
    main()
