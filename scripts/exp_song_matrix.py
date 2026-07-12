"""One song as its own matrix - every entry a pixel, no pooling.

Idea: instead of N whole songs in R^D, chop ONE song into N windows of n
samples each (L = N*n samples total) and build the cyclic window-advance
operator

    A0 w_k = w_{k+1},  A0 w_N = w_1,     A0 = W S G^{-1} W^T,  size n x n.

A0 is n x n regardless of N, so with n = image side it is DISPLAYED 1:1,
one matrix entry per pixel - the picture IS the matrix, no averaging.
Seeded with window 1, iterating A0 plays the song and repeats.

Sizing/conditioning are studied systematically in exp_operator_sizing.py
and display modes in exp_matrix_viz.py; this script keeps the original
A0-vs-Gram comparison. Construction lives in loopviz.songmatrix.

Run: .venv/bin/python scripts/exp_song_matrix.py [--song 0] [--n 1024]
                                                 [--windows 512]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loopviz.matviz import gray
from loopviz.render import save_png
from loopviz.sheet import make_sheet
from loopviz.songmatrix import Plan, build, load_audio, materialize

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_song_matrix"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=int, default=0)
    ap.add_argument("--n", type=int, default=1024,
                    help="window length = matrix side = image side")
    ap.add_argument("--windows", type=int, default=512,
                    help="number of windows N (N < n gives the Gram rank "
                         "margin real audio needs; N = n sits exactly on "
                         "the existence boundary and is ill-conditioned)")
    args = ap.parse_args()
    n, N = args.n, args.windows
    OUT.mkdir(parents=True, exist_ok=True)

    wav = ROOT / "data" / "audio" / f"{args.song:03d}.wav"
    signal, T_sec = load_audio([wav])
    pl = Plan(n=n, N=N, f=N * n / T_sec, T=T_sec)
    print(f"song {args.song}: {T_sec:.0f} s, resampled to "
          f"{pl.f:.0f} Hz so that L = {N} x {n} = {pl.samples} samples")

    op, W = build(signal, pl)
    cond = op.gram_condition()
    err = op.playback_error()
    print(f"W: {n} x {N}, Gram condition {cond:.2e}, "
          f"playback (window-advance) error {err:.2e}")

    A0 = materialize(op)
    img = gray(A0)
    save_png(img, OUT / "A0_exact.png")
    save_png(img, OUT / "A0_exact_16bit.png", bit_depth=16)

    # the raw Gram (song self-similarity across windows) for comparison
    G_img = gray(W.T @ W - np.eye(N))

    make_sheet([(f"A0 = W S G^-1 W^T, exact {n}x{n} (1 entry = 1 pixel)", img),
                ("Gram W^T W (window self-similarity), for reference", G_img)],
               OUT / "song_matrix.png", tile_size=min(n, 640), cols=2, readme=f"""\
# exp_song_matrix: one song as its own square matrix

What is investigated: dropping the playlist and the free part entirely.
One song ({wav.name}), resampled to {pl.f:.0f} Hz so it has exactly
{N} x {n} samples, is cut into {N} windows of {n} samples (columns of W).
The cyclic window-advance operator A0 = W S G^-1 W^T is then {n} x {n}
and is displayed EXACTLY - one matrix entry per pixel, no pooling, no
approximation of any kind. Seeded with window 1, iterating A0 plays the
song and repeats. (-70 dB dither is added so silent stretches cannot
break the operator; see songmatrix.build.)

Why N < n: at N = n (the perfectly square split) the windows of real audio
are nearly linearly dependent (adjacent windows correlate; quiet passages
repeat) - measured Gram condition ~1e10 and playback breaks. N = n/2 keeps
a comfortable rank margin; exp_operator_sizing.py maps the whole boundary
(practical limit rho = N/n ~ 0.95).

- Gram condition number: {cond:.2e}
- playback (window-advance) error: {err:.2e}
- A0 has rank {N} in an {n} x {n} canvas. Dimension math: an n x n image
  holds N x n samples, i.e. T = N n / f seconds of audio at rate f.

Files: A0_exact.png (8-bit), A0_exact_16bit.png (print).
""")
    print(f"wrote {OUT}/song_matrix.png")


if __name__ == "__main__":
    main()
