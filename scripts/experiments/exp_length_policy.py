"""How unequal song lengths shape A0 itself - no optimization, no Z.

For each policy (crop-shortest, pad-longest, stretch) the song matrix is
rebuilt at 1 kHz and A0's own two pictures (block mean, block energy) are
rendered directly. This isolates what the *music* part of the operator
looks like under each policy; the aesthetic layer is a separate concern.

Run: .venv/bin/python scripts/exp_length_policy.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from loopviz.config import AudioConfig
from loopviz.ingest import build_song_matrix, song_durations, window_seconds
from loopviz.operator import PlaylistOperator
from loopviz.paper.render import energy_image, mean_image
from loopviz.paper.sheet import make_sheet

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "exp_length_policy"
SR = 1000
P = 512
POLICIES = ("crop-shortest", "pad-longest", "stretch")


def unit(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-30), 0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wavs = sorted((ROOT / "data" / "audio").glob("[0-9][0-9][0-9].wav"))

    tiles, stats = [], []
    for policy in POLICIES:
        seconds, stretch = window_seconds(policy, song_durations(wavs))
        cfg = AudioConfig(sample_rate=SR, excerpt_seconds=seconds,
                          stretch=stretch)
        X = build_song_matrix(wavs, cfg)
        op = PlaylistOperator.from_songs(X)
        L, R = op.factors()  # A0 alone

        E = energy_image(L, R, P)
        tiles.append((f"{policy} | A0 block energy",
                      unit(np.log1p(E * 1e6 / max(E.mean(), 1e-30)))))
        stats.append({
            "policy": policy,
            "D": op.D,
            "zero_frac": float((np.abs(X) < 1e-12).mean()),
            "gram_cond": op.gram_condition(),
            "playback_err": op.playback_error(),
        })
        # mean picture kept for completeness; it is near-blank by theory
        tiles.append((f"{policy} | A0 block mean (near-blank)",
                      unit(mean_image(L, R, P))))

    lines = "\n".join(
        f"- {s['policy']}: D={s['D']}, zeros {s['zero_frac']:.0%}, "
        f"Gram cond {s['gram_cond']:.2f}, playback err {s['playback_err']:.1e}"
        for s in stats)
    make_sheet(tiles, OUT / "a0_by_policy.png", tile_size=P, cols=2, readme=f"""\
# exp_length_policy: A0 under each length policy (no optimization, no Z)

What is varied: only the length policy (crop-shortest / pad-longest /
stretch) at {SR} Hz. Shown: the music part A0's own block-energy picture
(left column, where its structure lives) and its block-mean picture (right
column, near-blank because audio averages out inside blocks - see
doc/math.pdf section 3).

{lines}

Reading the energy pictures: bright bands = window pairs where songs share
loudness; dark bands = padded silence (pad-longest). All policies give an
exact operator; the choice is aesthetic/semantic, not mathematical.
""")
    for f in ["crop-shortest.png", "pad-longest.png", "stretch.png",
              "side_by_side.png", "results.json"]:
        (OUT / f).unlink(missing_ok=True)  # outputs of the old ES version
    (OUT / "results.json").write_text(json.dumps(stats, indent=2))
    print(f"wrote {OUT}/a0_by_policy.png")
    for s in stats:
        print(f"  {s['policy']:15s} D={s['D']:7d} zeros {s['zero_frac']:5.1%} "
              f"cond {s['gram_cond']:.2f} err {s['playback_err']:.1e}")


if __name__ == "__main__":
    main()
