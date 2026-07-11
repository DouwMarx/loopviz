"""Paper-first sweep: the print size is the input, f is the searched knob.

For each paper size the canvas side is fixed by physics, not searched:
n_max = (short side - 2x5 mm printer margin) / 0.25 mm pixel pitch. Per
(track, paper) the maximum stable sample rate fs_max is bisected at
n_max (loop drift per pass <= --drift-tol), then candidates are built on
a grid:

  n in {n_max, n_max-1, n_max-2}   - stepping n by 1 fully rerolls the
                                     interference texture at ~fixed rho,
                                     drift and print size (exp_moire_n)
  f linspace F_FLOOR..fs_max       - the only rho lever at fixed paper;
                                     capped at the 16 kHz source rate
                                     (upsampling manufactures redundancy)
  clip in {99.5, 99.9, 99.99}      - settled favorite, known edge, and
                                     the unexplored extreme

Every grid point is probed for drift before writing (bisection at n_max
does not vouch for neighbors on a resonance). Resumable: fs_max results
are checkpointed (sweep_state.json), existing candidate dirs skipped.

Run: .venv/bin/python scripts/exp_paper_sweep.py [--hours 60]
       [--drift-tol 1e-9] [--tracks 007,013]
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
from playlistviz.pool import candidate_dir, song_dir
from playlistviz.render import save_png
from playlistviz.songmatrix import (Plan, build, load_audio, loop_degradation,
                                    materialize)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_paper_sweep"

PITCH_MM = 0.25          # discernible nose-to-paper; printers sharp to ~0.2
MARGIN_MM = 5.0          # per side, so any printer can produce it
PAPERS = {"A4": 210.0, "A3": 297.0, "A2": 420.0}   # short side, mm
F_FLOOR = 4000.0         # below this it stops sounding like the song
F_CAP = 16000.0          # source sample rate: never upsample
RHO_CAP = 0.97           # measured practical existence limit
CLIPS = (99.5, 99.9, 99.99)
REROLLS = 3              # n_max, n_max-1, n_max-2


def n_for_paper(short_mm: float) -> int:
    return int((short_mm - 2 * MARGIN_MM) / PITCH_MM)


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


def max_feasible_f(signal, T, n, tol):
    """Largest f with drift <= tol at this n, or (None, reason)."""
    f_hi = min(F_CAP, RHO_CAP * n * n / T)
    if f_hi < F_FLOOR:
        return None, (f"needs f < {F_FLOOR:.0f} Hz even at rho={RHO_CAP} "
                      f"(song too long for this paper)")
    r, _, _ = probe(signal, T, f_hi, n)
    if r["drift_per_pass"] <= tol:
        return f_hi, None
    r, _, _ = probe(signal, T, F_FLOOR, n)
    if r["drift_per_pass"] > tol:
        return None, (f"drift {r['drift_per_pass']:.1e} > {tol:.0e} "
                      f"even at the {F_FLOOR:.0f} Hz floor")
    lo, hi = F_FLOOR, f_hi          # lo feasible, hi not
    while hi - lo > 0.02 * lo:
        mid = 0.5 * (lo + hi)
        r, _, _ = probe(signal, T, mid, n)
        if r["drift_per_pass"] <= tol:
            lo = mid
        else:
            hi = mid
    return lo, None


def cand_id(stem: str, f: float, n: int, clip: float) -> str:
    return f"songop_t{stem}_f{f:.0f}_n{n}_c{clip:g}"


def write_candidate(pt, pl, wav, paper, clip, img, w_eq):
    phi = features(img)
    loss_eq = float(scalar_loss(phi, w_eq))
    cid = cand_id(wav.stem, pt["f_hz"], pt["n"], clip)
    manifest_path = ROOT / "data" / "tracks.json"
    meta = {}
    if manifest_path.exists():
        meta = json.loads(manifest_path.read_text()).get(wav.stem, {})
    cdir = song_dir(ROOT / "runs", meta.get("title", ""), wav.name,
                    paper) / cid
    cdir.mkdir(parents=True, exist_ok=True)
    save_png(img, cdir / "presentation.png")
    (cdir / "candidate.json").write_text(json.dumps({
        "id": cid, "kind": "songop", "song": wav.name,
        "title": meta.get("title", ""), "artist": meta.get("artist", ""),
        "n": pl.n, "N": pl.N, "rho": pl.rho, "f_hz": pl.f,
        "clip_pct": clip, "paper": paper, "pitch_mm": PITCH_MM,
        "print_side_mm": pl.n * PITCH_MM,
        "playback_err": pt["playback_err"],
        "gram_cond": pt["gram_cond"],
        "loop_drift_per_pass": pt["drift_per_pass"],
        "phi": {m.name: float(v) for m, v in zip(METRICS, phi)},
        "loss_vector": [float(v) for v in loss_vector(phi)],
        "loss_eq": loss_eq,
    }, indent=2))
    return cid, loss_eq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=60.0)
    ap.add_argument("--drift-tol", type=float, default=1e-9)
    ap.add_argument("--f-points", type=int, default=5)
    ap.add_argument("--tracks", default="",
                    help="comma-separated stems to restrict to (smoke test)")
    args = ap.parse_args()
    tol = args.drift_tol
    deadline = time.time() + args.hours * 3600
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "sweep_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    wavs = sorted((ROOT / "data" / "audio").glob("[0-9][0-9][0-9].wav"))
    if args.tracks:
        keep = set(args.tracks.split(","))
        wavs = [w for w in wavs if w.stem in keep]
    print(f"{len(wavs)} tracks x {len(PAPERS)} papers "
          f"(n_max: {', '.join(f'{p}={n_for_paper(s)}' for p, s in PAPERS.items())})"
          f" | budget {args.hours} h | drift tol {tol:.0e}")
    w_eq = equal_weights()
    made = skipped = errors = 0
    audio_cache = {}

    # smallest papers first: fast feasibility verdicts + early candidates
    for paper, short_mm in sorted(PAPERS.items(), key=lambda kv: kv[1]):
        n_max = n_for_paper(short_mm)
        for wav in wavs:
            if time.time() > deadline:
                print("time budget reached - stopping gracefully")
                state_path.write_text(json.dumps(state, indent=2))
                print(f"made {made}, skipped {skipped}, errors {errors}")
                return
            key = f"{wav.stem}|{paper}"
            try:
                if wav.stem not in audio_cache:
                    audio_cache[wav.stem] = load_audio([wav])
                signal, T = audio_cache[wav.stem]

                if key not in state:
                    fs_max, why = max_feasible_f(signal, T, n_max, tol)
                    state[key] = {"fs_max": fs_max, "why": why, "T": T,
                                  "n_max": n_max}
                    state_path.write_text(json.dumps(state, indent=2))
                fs_max = state[key]["fs_max"]
                if fs_max is None:
                    print(f"{wav.stem} {paper}: infeasible "
                          f"({state[key]['why']})")
                    skipped += 1
                    continue

                f_grid = sorted({round(f) for f in
                                 np.linspace(F_FLOOR, fs_max, args.f_points)})
                for n in range(n_max, n_max - REROLLS, -1):
                    for f in f_grid:
                        todo = [c for c in CLIPS if candidate_dir(
                            ROOT / "runs",
                            cand_id(wav.stem, f, n, c)) is None]
                        if not todo:
                            continue
                        pt, pl, op = probe(signal, T, f, n)
                        # fs_max was bisected at n_max; at the n-1/n-2
                        # rerolls the same f means slightly higher rho,
                        # which can poke past the cap - enforce per n
                        if pl.rho > RHO_CAP:
                            print(f"{wav.stem} {paper} n={n} f={f}: "
                                  f"rho={pl.rho:.4f} > {RHO_CAP} - skip")
                            continue
                        if pt["drift_per_pass"] > tol:   # resonance at (n, f)
                            print(f"{wav.stem} {paper} n={n} f={f}: drift "
                                  f"{pt['drift_per_pass']:.1e} > {tol:.0e}"
                                  f" - skip")
                            continue
                        A0 = materialize(op)
                        for clip in todo:
                            img = gray(A0, pct=clip)
                            write_candidate(pt, pl, wav, paper, clip,
                                            img, w_eq)
                            made += 1
                        print(f"{wav.stem} {paper} n={n} f={f} "
                              f"rho={pl.rho:.2f} drift "
                              f"{pt['drift_per_pass']:.0e} -> {len(todo)} "
                              f"[{made} made, "
                              f"{(deadline - time.time())/3600:.1f} h left]",
                              flush=True)
            except Exception:
                errors += 1
                print(f"ERROR at {key}:\n{traceback.format_exc()}")

    state_path.write_text(json.dumps(state, indent=2))
    print(f"sweep complete: made {made}, skipped {skipped}, errors {errors}")


if __name__ == "__main__":
    main()
