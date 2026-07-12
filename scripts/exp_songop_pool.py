"""Nested fidelity-constrained search for the pixel-exact song operator.

The design constraints are FIDELITY bounds, not aesthetic knobs:
- sample rate f in [f_lo, f_hi]           (sounds like the song)
- loop horizon >= H years to 1% drift     (loops until you die)
- matrix side n <= n_max                  (printable with discernible px)

Nested scheme, per sample rate f:
1. n <= n_max forces rho >= f T / n_max^2 (rho = f T / n^2).
2. Check the horizon at n = n_max (the best-conditioned allowed point);
   if even that fails, this f is infeasible under the caps.
3. Bisect the largest feasible n... rho and n are one knob here
   (rho = fT/n^2), so the band is found by bisecting n downward until the
   horizon breaks, then FINELY sweeping every few samples of n across the
   band - fine because window length can resonate with musical structure
   (n ~ integer beats -> aligned windows -> conditioning spikes), which
   coarse grids would miss. Conditioning, playback error and drift are
   recorded per n.
4. The BT candidate pool is rebuilt from points spread across the
   feasible set (x ink clip percentiles), each stamped with its horizon.

Run: .venv/bin/python scripts/exp_songop_pool.py [--song 2]
       [--f-lo 4000] [--f-hi 8000] [--n-f 9] [--horizon-years 100]
       [--n-max 1000] [--sweep-points 40] [--pool-size 12]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUT = ROOT / "runs" / "exp_songop_pool"
CLIPS = (99.5, 99.9)
SECONDS_PER_YEAR = 3600 * 24 * 365.25


def probe(signal, T, f, n):
    """Build the operator for (f, n) and measure everything cheap."""
    N = max(2, round(f * T / n))
    pl = Plan(n=n, N=N, f=N * n / T, T=T)
    op, W = build(signal, pl)
    drift = loop_degradation(op.factors(), W, loops=1)[0]
    passes = 0.01 / max(drift, 1e-300)
    return {
        "n": n, "N": N, "rho": pl.rho, "f_hz": pl.f,
        "gram_cond": op.gram_condition(),
        "playback_err": op.playback_error(),
        "drift_per_pass": drift,
        "horizon_years": passes * T / SECONDS_PER_YEAR,
    }, pl, op


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=int, default=2)
    ap.add_argument("--f-lo", type=float, default=4000)
    ap.add_argument("--f-hi", type=float, default=8000)
    ap.add_argument("--n-f", type=int, default=9)
    ap.add_argument("--horizon-years", type=float, default=100.0)
    ap.add_argument("--n-max", type=int, default=1000)
    ap.add_argument("--sweep-points", type=int, default=40)
    ap.add_argument("--pool-size", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    wav = ROOT / "data" / "audio" / f"{args.song:03d}.wav"
    signal, T = load_audio([wav])
    H = args.horizon_years

    # audio previews: judge by EAR how low f can go and still be the song
    import soundfile as sf
    from scipy.signal import resample
    prev = OUT / "audio_preview"
    prev.mkdir(parents=True, exist_ok=True)
    for fp in sorted({args.f_lo, 2500.0, 3000.0, 3500.0, 4000.0}):
        if args.f_lo <= fp <= args.f_hi:
            y = resample(signal, int(fp * T))
            sf.write(str(prev / f"f{fp:.0f}.wav"), y / max(
                1e-9, abs(y).max()) * 0.9, int(fp))
    print(f"audio previews for the f floor: {prev}/f*.wav")
    print(f"song {args.song}: {T:.1f} s | f in [{args.f_lo:.0f}, "
          f"{args.f_hi:.0f}] Hz | horizon >= {H:.0f} yr | n <= {args.n_max}")

    all_points, bands = [], []
    for f in np.linspace(args.f_lo, args.f_hi, args.n_f):
        n_hi = args.n_max
        rho_min = f * T / n_hi ** 2
        if rho_min > 0.97:
            bands.append({"f_hz": f, "feasible": False,
                          "reason": f"n<={n_hi} forces rho={rho_min:.2f} "
                                    "past the existence boundary"})
            print(f"f={f:6.0f} Hz: infeasible (rho_min={rho_min:.2f} > 0.97)")
            continue

        top, _, _ = probe(signal, T, f, n_hi)
        top["f_nominal"] = f
        if top["horizon_years"] < H:
            bands.append({"f_hz": f, "feasible": False,
                          "reason": f"horizon at n={n_hi} (rho="
                                    f"{top['rho']:.2f}) is only "
                                    f"{top['horizon_years']:.2f} yr"})
            print(f"f={f:6.0f} Hz: infeasible (horizon at n_max = "
                  f"{top['horizon_years']:.2f} yr < {H:.0f})")
            all_points.append({**top, "feasible": False})
            continue

        # bisect the smallest n (largest rho) that still meets the horizon
        lo = int(np.ceil(np.sqrt(f * T / 0.97)))      # existence side
        hi = n_hi                                     # known feasible
        while hi - lo > 2:
            mid = (lo + hi) // 2
            r, _, _ = probe(signal, T, f, mid)
            r["f_nominal"] = f
            all_points.append({**r, "feasible": r["horizon_years"] >= H})
            if r["horizon_years"] >= H:
                hi = mid
            else:
                lo = mid
        n_lo = hi
        bands.append({"f_hz": f, "feasible": True, "n_lo": n_lo,
                      "n_hi": n_hi, "rho_lo": f * T / n_hi ** 2,
                      "rho_hi": f * T / n_lo ** 2})
        print(f"f={f:6.0f} Hz: feasible n in [{n_lo}, {n_hi}]  "
              f"(rho in [{f * T / n_hi**2:.3f}, {f * T / n_lo**2:.3f}])")

        # fine sweep across the band (resonance hunting: small steps in n)
        for n in np.unique(np.linspace(n_lo, n_hi,
                                       args.sweep_points).astype(int)):
            r, _, _ = probe(signal, T, f, int(n))
            r["f_nominal"] = f
            r["feasible"] = r["horizon_years"] >= H
            all_points.append(r)
            if not r["feasible"]:  # resonance can break mid-band - record
                print(f"    n={n}: horizon dips to "
                      f"{r['horizon_years']:.1f} yr (resonance?)")

    # ---- diagnostic plot: conditioning and horizon vs n, per f ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for band in bands:
        pts = sorted((p for p in all_points
                      if p.get("f_nominal") == band["f_hz"]),
                     key=lambda p: p["n"])
        if not pts:
            continue
        ns = [p["n"] for p in pts]
        axes[0].semilogy(ns, [p["gram_cond"] for p in pts],
                         marker=".", label=f"f~{band['f_hz']:.0f} Hz")
        axes[1].semilogy(ns, [max(p["horizon_years"], 1e-6) for p in pts],
                         marker=".")
    axes[0].set_ylabel("Gram condition")
    axes[0].legend(fontsize=8)
    axes[1].axhline(H, color="r", ls="--", lw=1, label=f"{H:.0f} yr")
    axes[1].set_ylabel("loop horizon (years to 1% drift)")
    axes[1].set_xlabel("matrix side n")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fidelity_map.png", dpi=130)
    plt.close(fig)

    # ---- extend the BT pool from the feasible set ----
    # existing songop_* candidates are KEPT: they remain valid members of
    # the feasible set and recorded comparisons reference them by id.
    feas = sorted((p for p in all_points if p.get("feasible")),
                  key=lambda p: (p["f_hz"], p["n"]))
    idx = np.unique(np.linspace(0, len(feas) - 1,
                                min(args.pool_size, len(feas))).astype(int))
    chosen = [feas[i] for i in idx]
    w_eq = equal_weights()
    tiles, rows = [], []
    for pt in chosen:
        r, pl, op = probe(signal, T, pt["f_hz"], pt["n"])
        A0 = materialize(op)
        for clip in CLIPS:
            img = gray(A0, pct=clip)
            phi = features(img)
            loss_eq = float(scalar_loss(phi, w_eq))
            cid = f"songop_f{pt['f_hz']:.0f}_n{pt['n']}_c{clip:.1f}"
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
                "phi": {m.name: float(v) for m, v in zip(METRICS, phi)},
                "loss_vector": [float(v) for v in loss_vector(phi)],
                "loss_eq": loss_eq,
            }, indent=2))
            tiles.append((f"f={pl.f:.0f}Hz n={pl.n} rho={pl.rho:.2f} "
                          f"clip={clip} L={loss_eq:.2f} "
                          f"hz={r['horizon_years']:.0f}yr", img))
            rows.append({"id": cid, "f_hz": pl.f, "n": pl.n,
                         "rho": pl.rho, "clip": clip, "loss_eq": loss_eq,
                         "horizon_years": r["horizon_years"]})

    rows.sort(key=lambda r: r["loss_eq"])
    band_txt = "\n".join(
        (f"- f={b['f_hz']:.0f} Hz: n in [{b['n_lo']}, {b['n_hi']}], "
         f"rho in [{b['rho_lo']:.3f}, {b['rho_hi']:.3f}]")
        if b["feasible"] else
        f"- f={b['f_hz']:.0f} Hz: INFEASIBLE - {b['reason']}"
        for b in bands)
    rank = "\n".join(f"- {r['id']}: L_eq={r['loss_eq']:.3f} "
                     f"(horizon {r['horizon_years']:.0f} yr)" for r in rows)
    make_sheet(tiles, OUT / "songop_pool.png", tile_size=330,
               cols=len(CLIPS) * 2, readme=f"""\
# exp_songop_pool: fidelity-constrained nested search

Song {wav.name} ({T:.1f} s). The bounds are FIDELITY constraints, not
aesthetic knobs: f in [{args.f_lo:.0f}, {args.f_hi:.0f}] Hz (sounds like
the song), loop horizon >= {H:.0f} years to 1% drift (loops until you
die), n <= {args.n_max} (printable). rho is not free: rho = f T / n^2.

Nested scheme per f: n <= {args.n_max} forces rho up; the horizon forces
rho down; bisection finds the band; a fine n-sweep ({args.sweep_points}
points) maps it, because window length can resonate with musical
structure (n ~ integer beats -> aligned windows -> conditioning spikes).
See fidelity_map.png (Gram condition and horizon vs n, per f).

## Feasible bands
{band_txt}

## Pool ({len(rows)} candidates in runs/songop_*, do `loopviz compare`)

Equal-weight metric ranking (a prior, not the verdict):
{rank}

Every candidate is the EXACT operator; each candidate.json carries its
measured drift/pass and loop horizon in years.

Viewing note: sheet tiles are full matrices, downscaled - signed texture
cancels toward gray at reduced scale; judge candidates in `loopviz
compare` (nearest-neighbor scaling) or the presentation.png files at
native size.
""")
    (OUT / "results.json").write_text(json.dumps(
        {"bands": bands, "sweep": all_points, "pool": rows}, indent=2))
    print(f"wrote {OUT}/songop_pool.png, fidelity_map.png, "
          f"{len(rows)} candidates")


if __name__ == "__main__":
    main()
