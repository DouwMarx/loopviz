"""Exact-only candidate pool for the pixel-print song operator.

The honest aesthetic parameter space, after exp_error_budget and the
attenuation ES showed that any playback compromise buys nothing visually:

- rho = N/n, the rank fraction (implies f = rho n^2 / T)
- the ink map's symmetric clip percentile

Every candidate is the EXACT operator - zero attenuation, eigenvalues on
the unit circle by construction, so the loop claim is structural. The
measured drift per pass (float64 + pinv truncation, linear in passes) is
reported per rho as a "loop horizon": passes until 1% drift.

The space is 2D, so instead of an evolutionary search this enumerates the
full grid, ranks it on the equal-weight metric loss, AND writes every
gridpoint as a comparison candidate (runs/songop_*/candidate.json) so
pairwise Bradley-Terry preferences can be collected on the real objects
- before trusting any metric weighting (`playlistviz compare`).

Run: .venv/bin/python scripts/exp_songop_pool.py [--song 2] [--n 1000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.loss import equal_weights, loss_vector, scalar_loss
from playlistviz.matviz import gray
from playlistviz.metrics import METRICS, features
from playlistviz.render import save_png
from playlistviz.sheet import make_sheet
from playlistviz.songmatrix import (build, load_audio, loop_degradation,
                                    materialize, plan)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_songop_pool"
RHOS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CLIPS = (98.0, 99.0, 99.5, 99.9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=int, default=2)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wav = ROOT / "data" / "audio" / f"{args.song:03d}.wav"
    signal, T = load_audio([wav])
    w_eq = equal_weights()

    tiles, rows, horizons = [], [], []
    for rho in RHOS:
        pl = plan(T, n=args.n, rho=rho)
        op, W = build(signal, pl)
        err = op.playback_error()
        A0 = materialize(op)
        drift = loop_degradation(A0, W, loops=1)[0]
        passes_to_1pct = 0.01 / max(drift, 1e-300)
        horizons.append({"rho": pl.rho, "N": pl.N, "f_hz": pl.f,
                         "playback_err": err, "drift_per_pass": drift,
                         "passes_to_1pct": passes_to_1pct,
                         "hours_to_1pct": passes_to_1pct * T / 3600})
        print(f"rho={pl.rho:.2f}: N={pl.N} f={pl.f:.0f} Hz "
              f"playback {err:.1e} drift/pass {drift:.1e} "
              f"-> 1% after {passes_to_1pct:,.0f} passes")

        for clip in CLIPS:
            img = gray(A0, pct=clip)
            phi = features(img)
            loss_eq = float(scalar_loss(phi, w_eq))
            cid = f"songop_r{int(round(rho * 100)):02d}_c{clip:.1f}"
            cdir = ROOT / "runs" / cid
            cdir.mkdir(parents=True, exist_ok=True)
            save_png(img, cdir / "presentation.png")
            (cdir / "candidate.json").write_text(json.dumps({
                "id": cid, "kind": "songop", "song": wav.name,
                "n": pl.n, "N": pl.N, "rho": pl.rho, "f_hz": pl.f,
                "clip_pct": clip, "playback_err": err,
                "loop_drift_per_pass": drift,
                "phi": {m.name: float(v) for m, v in zip(METRICS, phi)},
                "loss_vector": [float(v) for v in loss_vector(phi)],
                "loss_eq": loss_eq,
            }, indent=2))
            tiles.append((f"rho={pl.rho:.2f} clip={clip}  L={loss_eq:.2f}",
                          img))
            rows.append({"id": cid, "rho": pl.rho, "clip": clip,
                         "loss_eq": loss_eq})

    rows.sort(key=lambda r: r["loss_eq"])
    rank = "\n".join(f"- {r['id']}: L_eq = {r['loss_eq']:.3f}" for r in rows)
    hor = "\n".join(
        f"- rho={h['rho']:.2f} (f={h['f_hz']:.0f} Hz): drift "
        f"{h['drift_per_pass']:.1e}/pass -> 1% after "
        f"{h['passes_to_1pct']:,.0f} passes ({h['hours_to_1pct']:,.0f} h "
        f"of continuous looping)" for h in horizons)
    make_sheet(tiles, OUT / "songop_pool.png", tile_size=330,
               cols=len(CLIPS), readme=f"""\
# exp_songop_pool: the exact-only candidate grid

Song {wav.name} ({T:.1f} s), print side n = {args.n}. Every candidate is
the EXACT operator (zero attenuation): the playlist equation holds to
~1e-8 per window and the eigenvalues sit on the unit circle by
construction, so looping is structural, not budgeted.

What is varied: rho in {RHOS} (rows) x ink clip percentile in {CLIPS}
(columns). That is the entire honest parameter space at fixed song and
paper; f = rho n^2 / T follows from rho.

Equal-weight metric ranking (a prior, not the verdict - do pairwise
comparisons with `playlistviz compare`; candidates are in runs/songop_*):
{rank}

## Loop horizon (honesty numbers for "loop until you die")

In exact arithmetic the operator loops forever. In float64 the drift per
pass is measured at:
{hor}

Drift grows linearly (not exponentially) with passes. Lower rho = better
conditioned Gram = longer horizon; that is the honest trade against
information density.

Viewing note: sheet tiles are the full matrices, downscaled - signed
texture cancels toward gray at reduced scale, so judge candidates in
`playlistviz compare` or the runs/songop_*/presentation.png files at
native size.
""")
    (OUT / "results.json").write_text(json.dumps(
        {"ranking": rows, "loop_horizons": horizons}, indent=2))
    print(f"wrote {OUT}/songop_pool.png and {len(rows)} candidates")


if __name__ == "__main__":
    main()
