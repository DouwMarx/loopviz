"""How many height levels does the printed object need to still play?

The relief carries each entry of A0 as one of L heights (L ~ 20-80 on
FDM), i.e. the object is Q(A0), the entry-wise quantization of the exact
operator. Two questions:

1. Plain quantization: step error |Q(A0) w_k - w_{k+1}| and loop drift
   as a function of L, for several rank fractions rho (lower rho = better
   conditioned = less amplification of quantization noise) and clip.
2. Quantization-aware operator: A0 is only the min-norm exact operator;
   every A = A0 + Z P_perp is exact too, and at rho < 1 the free part has
   n(n - N) dimensions. Alternating projections between the lattice
   (quantize) and the affine solution set (A <- A + (W_next - A W) G^-1 W^T)
   look for an operator that is BOTH on the printer's height grid AND
   plays the loop. Reported: the residual of the quantized iterate.

Run: .venv/bin/python scripts/exp_relief_quantization.py
       [--audio data/audio_3d/song.wav --start 324.68 --end 338.18]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loopviz.relief import quantized_matrix
from loopviz.relief_cli import load_loop
from loopviz.songmatrix import Plan, build, loop_degradation, materialize

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "relief" / "exp_quantization"
LEVELS = (16, 32, 64, 256, 1024, 4096, 65536)
CLIPS = (99.5, 100.0)


def step_err(A, W):
    Xn = np.roll(W, -1, axis=1)
    e = np.linalg.norm(A @ W - Xn, axis=0) / np.linalg.norm(Xn, axis=0)
    return float(e.max()), float(e.mean())


def quantize_fixed(A, m, levels):
    """Uniform lattice on [-m, m] with `levels` points, values clipped."""
    v = np.clip(A / m, -1.0, 1.0)
    q = np.rint((v + 1.0) / 2.0 * (levels - 1))
    return (q / (levels - 1) * 2.0 - 1.0) * m


def alternating(A0, W, m, levels, iters, seed=0):
    """Q(A) on the lattice, P(A) the exact solution nearest to A."""
    Xn = np.roll(W, -1, axis=1)
    G = W.T @ W
    Ginv = np.linalg.pinv(G, rcond=1e-10)
    A = A0.copy()
    hist = []
    best = (np.inf, None)
    for k in range(iters):
        Aq = quantize_fixed(A, m, levels)
        emax, _ = step_err(Aq, W)
        hist.append(emax)
        if emax < best[0]:
            best = (emax, Aq)
        # project the quantized iterate back onto {A : A W = Xn}
        A = Aq + (Xn - Aq @ W) @ Ginv @ W.T
    return best[1], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/audio_3d/song.wav")
    ap.add_argument("--start", type=float, default=324.68)
    ap.add_argument("--end", type=float, default=338.18)
    ap.add_argument("--n", type=int, nargs="+", default=[150, 250])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.5, 0.7, 0.95])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--levels", type=int, nargs="+", default=list(LEVELS))
    ap.add_argument("--clips", type=float, nargs="+", default=list(CLIPS))
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    args = ap.parse_args()
    global OUT
    OUT = OUT.with_name(OUT.name + args.tag)
    OUT.mkdir(parents=True, exist_ok=True)
    signal, T, _ = load_loop(ROOT / args.audio, args.start, args.end)

    rows, lines = [], []
    for n in args.n:
        for rho in args.rho:
            N = round(rho * n)
            pl = Plan(n=n, N=N, f=N * n / T, T=T)
            op, W = build(signal, pl)
            A = materialize(op)
            exact = step_err(A, W)[0]
            lines.append(f"\n## n={n} rho={rho} (N={N}, f={pl.f:.0f} Hz): exact "
                         f"step err {exact:.1e}, Gram cond {op.gram_condition():.1e}, "
                         f"|A| 99.5% = {np.percentile(np.abs(A), 99.5):.3g}, "
                         f"max = {np.abs(A).max():.3g}")
            lines.append("| levels | clip | step err max | step err mean | drift/pass "
                         "| alt-proj best step err | alt-proj iters |")
            lines.append("|---|---|---|---|---|---|---|")
            for clip in args.clips:
                m = np.percentile(np.abs(A), clip)
                for L in args.levels:
                    Aq = quantized_matrix(A, L, clip)
                    emax, emean = step_err(Aq, W)
                    drift = loop_degradation(Aq, W, loops=1)[0]
                    alt = ""
                    row = {"n": n, "rho": rho, "clip": clip, "levels": L,
                           "step_err_max": emax, "step_err_mean": emean,
                           "drift": drift}
                    if L <= 256 and rho < 0.95:
                        t = time.time()
                        _, hist = alternating(A, W, m, L, args.iters)
                        alt = f"{min(hist):.2e} (start {hist[0]:.2e})"
                        row.update({"alt_best": float(min(hist)),
                                    "alt_iter": int(np.argmin(hist)),
                                    "alt_secs": time.time() - t})
                        alt_it = f"{int(np.argmin(hist))}/{len(hist)}"
                    else:
                        alt_it = ""
                    rows.append(row)
                    lines.append(f"| {L} | {clip:g} | {emax:.2e} | {emean:.2e} | "
                                 f"{drift:.1e} | {alt} | {alt_it} |")
                    print(f"n={n} rho={rho} clip={clip} L={L}: step {emax:.2e} "
                          f"drift {drift:.1e} {alt}", flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    (OUT / "README.md").write_text(
        "# exp_relief_quantization: does the printed object still play?\n\n"
        f"Loop {args.start}-{args.end} s (T = {T:.2f} s). Step err = max over "
        "windows of |Q(A) w_k - w_(k+1)| / |w_(k+1)| (1 = as wrong as silence). "
        "drift/pass = error after one full loop through the object; inf = the "
        "iteration blew up.\n" + "\n".join(lines) + "\n")
    print(f"wrote {OUT}/README.md")


if __name__ == "__main__":
    main()
