"""Overnight sweep: a boatload of exact candidates for the real tracks.

For every wav in data/audio and every sample rate in FREQS (ordered by
your measured preference so partial runs cover all tracks at the best
rates first), the smallest matrix side n meeting the loop-stability
margin is bisected, and candidates are written at two densities (smallest n =
densest print, and ~18% roomier) x three ink clips centered on your
preferred 99.5. Everything is the EXACT operator; loop stability means
drift per pass <= --drift-tol (default 1e-9), i.e. indistinguishable
from exact under iteration - the check exists to keep a safe margin
from the conditioning boundary, not as a fidelity claim.

Resumable: bisection results are checkpointed (sweep_state.json) and
existing candidate dirs are skipped, so re-running continues where it
stopped. A time budget (--hours) stops the run gracefully.

Run: .venv/bin/python scripts/exp_overnight_sweep.py [--hours 8]
       [--drift-tol 1e-9] [--n-max 2000]
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.loss import equal_weights, loss_vector, scalar_loss
from playlistviz.matviz import gray
from playlistviz.metrics import METRICS, features
from playlistviz.render import save_png
from playlistviz.songmatrix import (Plan, build, load_audio, loop_degradation,
                                    materialize)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_overnight_sweep"
FREQS = (4000.0, 6000.0, 3000.0, 5000.0, 8000.0)   # preference-ordered
CLIPS = (99.3, 99.5, 99.7)


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
    }, pl, op


def smallest_feasible_n(signal, T, f, n_max, tol):
    if f * T / n_max ** 2 > 0.97:
        return None, "needs n beyond n_max even at full rank"
    r_top, _, _ = probe(signal, T, f, n_max)
    if r_top["drift_per_pass"] > tol:
        return None, (f"drift at n_max {r_top['drift_per_pass']:.1e} "
                      f"> {tol:.0e}")
    lo = int(np.ceil(np.sqrt(f * T / 0.97)))
    hi = n_max
    while hi - lo > 4:
        mid = (lo + hi) // 2
        r, _, _ = probe(signal, T, f, mid)
        if r["drift_per_pass"] <= tol:
            hi = mid
        else:
            lo = mid
    return hi, None


def write_candidate(pt, pl, wav, clip, img, w_eq):
    phi = features(img)
    loss_eq = float(scalar_loss(phi, w_eq))
    cid = f"songop_t{wav.stem}_f{pt['f_hz']:.0f}_n{pt['n']}_c{clip:.1f}"
    cdir = ROOT / "runs" / cid
    cdir.mkdir(parents=True, exist_ok=True)
    save_png(img, cdir / "presentation.png")
    manifest_path = ROOT / "data" / "tracks.json"
    meta = {}
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text()).get(wav.stem, {})
    (cdir / "candidate.json").write_text(json.dumps({
        "id": cid, "kind": "songop", "song": wav.name,
        "title": meta.get("title", ""), "artist": meta.get("artist", ""),
        "n": pl.n, "N": pl.N, "rho": pl.rho, "f_hz": pl.f,
        "clip_pct": clip, "playback_err": pt["playback_err"],
        "gram_cond": pt["gram_cond"],
        "loop_drift_per_pass": pt["drift_per_pass"],
        "phi": {m.name: float(v) for m, v in zip(METRICS, phi)},
        "loss_vector": [float(v) for v in loss_vector(phi)],
        "loss_eq": loss_eq,
    }, indent=2))
    return cid, loss_eq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--drift-tol", type=float, default=1e-9,
                    help="loop-stability margin: max drift per full pass")
    ap.add_argument("--n-max", type=int, default=2000)
    args = ap.parse_args()
    tol = args.drift_tol
    deadline = time.time() + args.hours * 3600
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "sweep_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    wavs = sorted((ROOT / "data" / "audio").glob("[0-9][0-9][0-9].wav"))
    print(f"{len(wavs)} tracks | budget {args.hours} h | "
          f"drift tol {tol:.0e} | n <= {args.n_max}")
    w_eq = equal_weights()
    made = skipped = errors = 0
    audio_cache = {}

    for f in FREQS:
        for wav in wavs:
            if time.time() > deadline:
                print("time budget reached - stopping gracefully")
                state_path.write_text(json.dumps(state, indent=2))
                print(f"made {made}, skipped {skipped}, errors {errors}")
                return
            key = f"{wav.stem}|{f:.0f}"
            try:
                if wav.stem not in audio_cache:
                    audio_cache[wav.stem] = load_audio([wav])
                signal, T = audio_cache[wav.stem]

                if key not in state:
                    n_min, why = smallest_feasible_n(signal, T, f,
                                                     args.n_max, tol)
                    state[key] = {"n_min": n_min, "why": why, "T": T}
                    state_path.write_text(json.dumps(state, indent=2))
                cell = state[key]
                if cell["n_min"] is None:
                    print(f"{wav.stem} f={f:.0f}: infeasible ({cell['why']})")
                    skipped += 1
                    continue

                n_min = cell["n_min"]
                n_variants = sorted({n_min,
                                     min(args.n_max, round(n_min * 1.18))})
                for n in n_variants:
                    todo = [c for c in CLIPS if not (
                        ROOT / "runs" /
                        f"songop_t{wav.stem}_f{f:.0f}_n{n}_c{c:.1f}" /
                        "candidate.json").exists()]
                    if not todo:
                        continue
                    pt, pl, op = probe(signal, T, f, n)
                    if pt["drift_per_pass"] > tol:  # resonance dip at this n
                        print(f"{wav.stem} f={f:.0f} n={n}: drift "
                              f"{pt['drift_per_pass']:.1e} > {tol:.0e} - skip")
                        continue
                    A0 = materialize(op)
                    for clip in todo:
                        img = gray(A0, pct=clip)
                        cid, loss_eq = write_candidate(pt, pl, wav, clip,
                                                       img, w_eq)
                        made += 1
                    print(f"{wav.stem} f={f:.0f} n={n} rho={pl.rho:.2f} "
                          f"drift {pt['drift_per_pass']:.0e} "
                          f"-> {len(todo)} candidates "
                          f"[{made} made, {(deadline - time.time())/3600:.1f}"
                          f" h left]")
            except Exception:
                errors += 1
                print(f"ERROR at {key}:\n{traceback.format_exc()}")

    state_path.write_text(json.dumps(state, indent=2))
    print(f"sweep complete: made {made}, skipped {skipped}, errors {errors}")


if __name__ == "__main__":
    main()
