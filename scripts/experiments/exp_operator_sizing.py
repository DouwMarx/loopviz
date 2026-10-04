"""How big must a pixel-exact song operator be, and how close to full rank
can you push it before it stops being a playlist?

One equation ties everything: samples L = f T = N n = rho n^2, so with the
song fixed (T) and the canvas fixed (n), the rank fraction rho = N/n and
the implied sample rate f = rho n^2 / T move together. This script sweeps
rho at fixed n and measures Gram condition + playback error, and renders
A0 at each point so the visual effect of rank is visible too.

Run: .venv/bin/python scripts/exp_operator_sizing.py [--song 2] [--n 1000]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loopviz.matviz import gray
from loopviz.render import save_png
from loopviz.sheet import make_sheet
from loopviz.songmatrix import (DISCERNIBLE_PITCH_MM, build, full_rank_side,
                                    load_audio, materialize, plan)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_operator_sizing"
RHOS = (0.5, 0.7, 0.85, 0.95, 0.98, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=int, default=2)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wav = ROOT / "data" / "audio" / f"{args.song:03d}.wav"
    signal, T = load_audio([wav])
    print(f"song {args.song}: {T:.1f} s -> full-rank side at 8 kHz: "
          f"{full_rank_side(T, 8000)} px")

    tiles, rows = [], []
    for rho in RHOS:
        pl = plan(T, n=args.n, rho=rho)
        op, _ = build(signal, pl)
        cond = op.gram_condition()
        err = op.playback_error()
        A0 = materialize(op)
        img = gray(A0)
        save_png(img, OUT / f"A0_rho{rho:.2f}.png")
        tiles.append((f"rho={pl.rho:.2f}  N={pl.N}  f={pl.f:.0f} Hz  "
                      f"cond={cond:.1e}  err={err:.0e} (full matrix)",
                      img))
        rows.append({"rho": pl.rho, "n": pl.n, "N": pl.N, "f_hz": pl.f,
                     "gram_cond": cond, "playback_err": err})
        print(f"  rho={pl.rho:.2f}  N={pl.N:5d}  f={pl.f:7.0f} Hz  "
              f"cond={cond:.2e}  playback err={err:.2e}")

    # capacity table: full-rank side needed for various audio at 8 kHz
    durations = {"song 2 (2:05)": 125.5, "song 0 (5:14)": 313.9,
                 "10 min": 600, "full playlist (27:36)": 1656,
                 "60 min": 3600}
    cap = "\n".join(
        f"- {name}: n = {full_rank_side(t, 8000)} px  ->  "
        f"{full_rank_side(t, 8000) * DISCERNIBLE_PITCH_MM / 10:.0f} cm at "
        f"{DISCERNIBLE_PITCH_MM} mm/px" for name, t in durations.items())

    stat = "\n".join(
        f"- rho={r['rho']:.2f}: N={r['N']}, f={r['f_hz']:.0f} Hz, "
        f"Gram cond {r['gram_cond']:.2e}, playback err {r['playback_err']:.2e}"
        for r in rows)
    make_sheet(tiles, OUT / "sizing_sweep.png", tile_size=500, cols=3, readme=f"""\
# exp_operator_sizing: how full-rank can the pixel-exact operator be?

The song ({wav.name}, {T:.1f} s) is cut into N windows of n = {args.n}
samples; A0 = W S G^-1 W^T is {args.n} x {args.n}, shown 1 entry = 1 pixel.
Everything is tied by  f T = N n = rho n^2 :  at fixed n, raising the rank
fraction rho = N/n raises the implied sample rate f = rho n^2 / T. So
"more rank" and "more audio fidelity" are the SAME knob here.

What is varied: rho in {RHOS}. Measured per point:

{stat}

Reading: playback is exact while the Gram stays well-conditioned; at
rho = 1 (every pixel worth exactly one sample) real audio windows go
nearly linearly dependent and the operator falls off the existence
boundary. The largest rho with playback err < 1e-6 is the practical
full-rank limit for this song.

Note on silence: this song contains digitally silent stretches, and a
linear operator cannot map the zero vector to the music that follows it -
without intervention the Gram condition is ~1e18 at EVERY rho. build()
therefore adds -70 dB deterministic dither (inaudible, standard audio
practice) before windowing; that alone restores playback to ~1e-8 for
rho <= 0.95. The rho = 1 failure that remains is the genuine boundary
(adjacent windows of real audio correlate), not a silence artifact.

## Capacity: canvas side needed for full rank at 8 kHz (n = sqrt(f T))

{cap}

A single song fits at wall-print scale with discernible pixels; a full
playlist needs meters of paper or a coarser pixel pitch - it is really a
one-song-per-print construction.

## Visual finding

The sheet tiles show the full matrices (files: A0_rho*.png). Downscaling
a signed A0 averages neighbors and cancels toward gray - so the sheet
tiles look flat and the files must be judged at native resolution (100%
zoom); that is a property of the object, not a rendering choice. The
texture wavelength
follows the implied sample rate: music energy lives at ~100-2000 Hz, so
at f ~ 7.5 kHz each audio cycle spans several pixels and A0 reads as
coherent fine striations; at f ~ 4 kHz the music sits near Nyquist and
the same construction degrades to salt-and-pepper. Higher f = smoother,
more legible texture - and oversampling beyond the source rate is
legitimate band-limited interpolation, not fabrication.
""")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT}/sizing_sweep.png")


if __name__ == "__main__":
    main()
