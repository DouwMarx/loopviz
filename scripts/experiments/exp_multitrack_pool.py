"""Blocked multi-track candidate pool: every song rendered under every
feasible parameter SETTING, for same-song pairwise comparisons.

A setting is track-independent: (sample rate f, ink clip percentile),
with the matrix side chosen per track as the SMALLEST n that meets the
loop-horizon requirement (most compact honest print), capped at n_max.
Cells where no n satisfies both bounds are skipped and reported.

The compare server pairs candidates only WITHIN one song (blocked
design), so every click is a pure parameter judgment - song identity is
identical on both sides - and the Bradley-Terry fit pools those
judgments across all tracks.

Run: .venv/bin/python scripts/exp_multitrack_pool.py
       [--freqs 2000,3000,4000,6000,8000] [--clips 99.5,99.9]
       [--horizon-years 100] [--n-max 2000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loopviz.loss import equal_weights, loss_vector, scalar_loss
from loopviz.matviz import gray
from loopviz.metrics import METRICS, features
from loopviz.render import save_png
from loopviz.sheet import make_sheet
from loopviz.songmatrix import (Plan, build, load_audio, loop_degradation,
                                    materialize)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_multitrack_pool"
SECONDS_PER_YEAR = 3600 * 24 * 365.25


def probe(signal, T, f, n):
    N = max(2, round(f * T / n))
    pl = Plan(n=n, N=N, f=N * n / T, T=T)
    op, W = build(signal, pl)
    drift = loop_degradation(op.factors(), W, loops=1)[0]
    return {
        "n": n, "N": N, "rho": pl.rho, "f_hz": pl.f,
        "gram_cond": op.gram_condition(),
        "playback_err": op.playback_error(),
        "drift_per_pass": drift,
        "horizon_years": 0.01 / max(drift, 1e-300) * T / SECONDS_PER_YEAR,
    }, pl, op


def smallest_feasible_n(signal, T, f, n_max, H):
    """Bisect the smallest n with loop horizon >= H years (None if none)."""
    r_top, _, _ = probe(signal, T, f, n_max)
    if r_top["horizon_years"] < H:
        return None, r_top
    lo = int(np.ceil(np.sqrt(f * T / 0.97)))   # existence boundary side
    hi = n_max
    best = r_top
    while hi - lo > 4:
        mid = (lo + hi) // 2
        r, _, _ = probe(signal, T, f, mid)
        if r["horizon_years"] >= H:
            hi, best = mid, r
        else:
            lo = mid
    return hi, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", default="2000,3000,4000,6000,8000")
    ap.add_argument("--clips", default="99.5,99.9")
    ap.add_argument("--horizon-years", type=float, default=100.0)
    ap.add_argument("--n-max", type=int, default=2000)
    args = ap.parse_args()
    freqs = [float(x) for x in args.freqs.split(",")]
    clips = [float(x) for x in args.clips.split(",")]
    H = args.horizon_years
    OUT.mkdir(parents=True, exist_ok=True)

    wavs = sorted((ROOT / "data" / "audio").glob("[0-9][0-9][0-9].wav"))
    w_eq = equal_weights()
    tiles, rows, skipped = [], [], []

    for wav in wavs:
        signal, T = load_audio([wav])
        for f in freqs:
            if f * T / args.n_max ** 2 > 0.97:
                skipped.append(f"{wav.name} @ {f:.0f} Hz: needs n > "
                               f"{args.n_max} even at full rank")
                print(f"{wav.name} f={f:.0f}: SKIP (rho > 0.97 at n_max)")
                continue
            n, r = smallest_feasible_n(signal, T, f, args.n_max, H)
            if n is None:
                skipped.append(f"{wav.name} @ {f:.0f} Hz: horizon at "
                               f"n={args.n_max} only "
                               f"{r['horizon_years']:.1f} yr")
                print(f"{wav.name} f={f:.0f}: SKIP (horizon "
                      f"{r['horizon_years']:.1f} yr at n_max)")
                continue
            r, pl, op = probe(signal, T, f, n)
            A0 = materialize(op)
            print(f"{wav.name} f={f:.0f}: n={n} rho={pl.rho:.3f} "
                  f"horizon {r['horizon_years']:.0f} yr")
            for clip in clips:
                img = gray(A0, pct=clip)
                phi = features(img)
                loss_eq = float(scalar_loss(phi, w_eq))
                cid = (f"songop_t{wav.stem}_f{f:.0f}_n{n}_c{clip:.1f}")
                cdir = ROOT / "runs" / cid
                cdir.mkdir(parents=True, exist_ok=True)
                save_png(img, cdir / "presentation.png")
                (cdir / "candidate.json").write_text(json.dumps({
                    "id": cid, "kind": "songop", "song": wav.name,
                    "n": pl.n, "N": pl.N, "rho": pl.rho, "f_hz": pl.f,
                    "clip_pct": clip, "playback_err": r["playback_err"],
                    "gram_cond": r["gram_cond"],
                    "loop_drift_per_pass": r["drift_per_pass"],
                    "loop_horizon_years": r["horizon_years"],
                    "phi": {m.name: float(v)
                            for m, v in zip(METRICS, phi)},
                    "loss_vector": [float(v) for v in loss_vector(phi)],
                    "loss_eq": loss_eq,
                }, indent=2))
                tiles.append((f"{wav.stem} f={f:.0f} n={n} clip={clip} "
                              f"L={loss_eq:.2f}", img))
                rows.append({"id": cid, "song": wav.name, "f_hz": f,
                             "n": n, "rho": pl.rho, "clip": clip,
                             "loss_eq": loss_eq,
                             "horizon_years": r["horizon_years"]})

    rows.sort(key=lambda r: r["loss_eq"])
    rank = "\n".join(f"- {r['id']}: L_eq={r['loss_eq']:.3f}" for r in rows)
    skip_txt = "\n".join(f"- {s}" for s in skipped) or "- none"
    make_sheet(tiles, OUT / "multitrack_pool.png", tile_size=280,
               cols=len(clips) * len(freqs), readme=f"""\
# exp_multitrack_pool: every song under every feasible setting

A setting is (f, clip); the matrix side is the SMALLEST n whose loop
horizon is >= {H:.0f} years (capped at {args.n_max}), found per track by
bisection - the most compact honest print of each song at each fidelity.

Settings: f in {freqs} Hz x clip in {clips}. {len(rows)} candidates
written to runs/songop_t*; comparisons are BLOCKED - `loopviz
compare` only pairs candidates of the same song, so each click is a pure
parameter judgment and the BT fit pools them across tracks.

## Skipped (track, f) cells
{skip_txt}

## Equal-weight metric ranking (a prior, not the verdict)
{rank}

Viewing note: sheet tiles are full matrices, downscaled - judge in the
compare UI (nearest-neighbor) or presentation.png at native size.
""")
    (OUT / "results.json").write_text(json.dumps(
        {"pool": rows, "skipped": skipped}, indent=2))
    print(f"wrote {OUT}/multitrack_pool.png, {len(rows)} candidates, "
          f"{len(skipped)} cells skipped")


if __name__ == "__main__":
    main()
