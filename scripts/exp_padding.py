"""Quick experiment: zero-padding vs center-cropping for unequal song lengths.

Variant CROP: every song center-cropped to a common excerpt (current default).
Variant PAD : longer excerpt window; songs shorter than the window keep their
              full length and are zero-padded at the end.

Reports Gram conditioning, playback error, and renders both operators with
identical theta so the visual consequence of padding is directly comparable.

Run: .venv/bin/python scripts/exp_padding.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.config import AudioConfig, ZConfig
from playlistviz.ingest import build_song_matrix
from playlistviz.operator import PlaylistOperator
from playlistviz.render import render, save_png
from playlistviz.zspace import N_PARAMS, generate_Z, song_envelopes, theta_to_params

import soundfile as sf

ROOT = Path(__file__).parent.parent
AUDIO = ROOT / "data" / "audio"
OUT = ROOT / "runs" / "exp_padding"
SR = 2000  # low rate: this experiment is about structure, not fidelity


def build_variant(name: str, seconds: float) -> PlaylistOperator:
    cfg = AudioConfig(sample_rate=SR, excerpt_seconds=seconds)
    wavs = sorted(AUDIO.glob("[0-9][0-9][0-9].wav"))
    X = build_song_matrix(wavs, cfg)
    op = PlaylistOperator.from_songs(X)
    zero_frac = float((np.abs(X) < 1e-12).mean())
    print(f"{name:5s}: window {seconds:5.0f}s  D={op.D:8d}  N={op.N}  "
          f"zero fraction {zero_frac:5.1%}  "
          f"Gram cond {op.gram_condition():8.2e}  "
          f"playback err {op.playback_error():.2e}")
    return op


def render_variant(op: PlaylistOperator, theta: np.ndarray, path: Path) -> None:
    params = theta_to_params(theta)
    zcfg = ZConfig(rank=16)
    env = song_envelopes(op.X)
    zf = generate_Z(params, op.D, zcfg, envelopes=env, project_perp=op.project_perp)
    L, R = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)
    save_png(render(L, R, 512, params, gray=True), path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lengths = []
    for w in sorted(AUDIO.glob("[0-9][0-9][0-9].wav")):
        info = sf.info(str(w))
        lengths.append(info.duration)
        print(f"  {w.name}: {info.duration:7.1f} s")
    print()

    # CROP window fits inside every song; PAD window exceeds the shortest,
    # so at least one song gets zero-padded.
    crop_s = min(lengths) * 0.9
    pad_s = min(lengths) * 2.0
    op_crop = build_variant("CROP", crop_s)
    op_pad = build_variant("PAD", pad_s)

    theta = np.random.default_rng(42).standard_normal(N_PARAMS)
    render_variant(op_crop, theta, OUT / "crop.png")
    render_variant(op_pad, theta, OUT / "pad.png")

    a = Image.open(OUT / "crop.png")
    b = Image.open(OUT / "pad.png")
    sheet = Image.new("RGB", (a.width + b.width + 10, a.height), (30, 30, 30))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (a.width + 10, 0))
    sheet.save(OUT / "side_by_side.png")
    print(f"\nwrote {OUT}/side_by_side.png (left: crop, right: pad)")


if __name__ == "__main__":
    main()
