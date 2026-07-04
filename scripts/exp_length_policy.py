"""Compare song-length policies under the current fitted BT weights.

For each policy (crop-shortest, pad-longest, stretch) the song matrix is
rebuilt at 1 kHz, a short ES optimizes the free part under the observer's
fitted weights, and the winner is rendered. Output: one side-by-side sheet
plus per-policy stats (D, zero fraction, Gram condition, playback error,
personal loss).

Run: .venv/bin/python scripts/exp_length_policy.py [--policy NAME]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.config import AudioConfig, OptConfig, RenderConfig, ZConfig
from playlistviz.ingest import build_song_matrix, song_durations, window_seconds
from playlistviz.loss import scalar_loss
from playlistviz.metrics import features
from playlistviz.operator import PlaylistOperator
from playlistviz.optimize import make_objective, run_es
from playlistviz.render import render, save_png
from playlistviz.zspace import generate_Z, song_envelopes, theta_to_params

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_length_policy"
SR = 1000
POLICIES = ("crop-shortest", "pad-longest", "stretch")


def run_policy(policy: str, w: np.ndarray, wavs: list[Path]) -> dict:
    seconds, stretch = window_seconds(policy, song_durations(wavs))
    cfg = AudioConfig(sample_rate=SR, excerpt_seconds=seconds, stretch=stretch)
    X = build_song_matrix(wavs, cfg)
    op = PlaylistOperator.from_songs(X)
    zero_frac = float((np.abs(X) < 1e-12).mean())

    t0 = time.time()
    zcfg, rcfg = ZConfig(), RenderConfig()
    objective = make_objective(op, w, zcfg, rcfg.opt_resolution,
                               stride=rcfg.opt_stride)
    best, _ = run_es(objective, OptConfig(generations=10, population=10, seed=3))

    params = theta_to_params(best.theta)
    env = song_envelopes(op.X)
    zf = generate_Z(params, op.D, zcfg, envelopes=env,
                    project_perp=op.project_perp)
    L, R = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)
    img = render(L, R, min(1024, op.D), params)
    phi = features(img)
    save_png(img, OUT / f"{policy}.png")
    return {
        "policy": policy,
        "D": op.D,
        "seconds": seconds,
        "zero_frac": zero_frac,
        "gram_cond": op.gram_condition(),
        "playback_err": op.playback_error(U=zf.U, Vp=zf.Vp, scale=zf.scale),
        "personal_loss": scalar_loss(phi, w),
        "es_seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=POLICIES,
                    help="run a single policy (default: all three)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    weights_file = ROOT / "runs" / "fitted_weights.json"
    w = np.asarray(json.loads(weights_file.read_text())["w_simplex"])
    wavs = sorted((ROOT / "data" / "audio").glob("[0-9][0-9][0-9].wav"))

    policies = [args.policy] if args.policy else list(POLICIES)
    results = [run_policy(p, w, wavs) for p in policies]

    print(f"\n{'policy':15s} {'D':>8s} {'zeros':>7s} {'cond':>8s} "
          f"{'playback':>10s} {'L_personal':>11s} {'ES time':>8s}")
    for r in results:
        print(f"{r['policy']:15s} {r['D']:8d} {r['zero_frac']:7.1%} "
              f"{r['gram_cond']:8.2e} {r['playback_err']:10.2e} "
              f"{r['personal_loss']:11.3f} {r['es_seconds']:7.0f}s")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    if len(policies) == 3:
        from PIL import Image
        tiles = [Image.open(OUT / f"{p}.png").resize((400, 400)) for p in policies]
        sheet = Image.new("L", (3 * 410, 400), 30)
        for k, im in enumerate(tiles):
            sheet.paste(im.convert("L"), (k * 410, 0))
        sheet.save(OUT / "side_by_side.png")
        print(f"\nwrote {OUT}/side_by_side.png ({' | '.join(policies)})")


if __name__ == "__main__":
    main()
