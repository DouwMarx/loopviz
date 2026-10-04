"""Display modes for the pixel-exact song operator: how should the matrix
that IS the song be shown?

The operator is built once (song 2, n = 1000, rho = 0.95 - the practical
full-rank limit from exp_operator_sizing). Every mode shows the ENTIRE
matrix - no crops, no pooling; a crop cannot reproduce the song, so it is
not the artwork. The only freedom is the value -> ink mapping:

- image modes (1 entry = 1 pixel): black & white, diverging palettes
- glyph modes (1 entry = 1 cell of 8x8 px): Hinton diagram, bubble chart
- 3D: full wireframe height field, every row and column drawn

Run: .venv/bin/python scripts/exp_matrix_viz.py [--song 2] [--n 1000]
                                                [--rho 0.95]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from loopviz.paper import matviz
from loopviz.paper.render import save_png
from loopviz.paper.sheet import make_sheet
from loopviz.songmatrix import build, load_audio, materialize, plan

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "exp_matrix_viz"


def load_img(path: Path) -> np.ndarray:
    """Load a saved mode image back as a square float tile for the sheet."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    side = min(im.size)
    x0, y0 = (im.width - side) // 2, (im.height - side) // 2
    return np.asarray(im.crop((x0, y0, x0 + side, y0 + side)),
                      dtype=np.float64) / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", type=int, default=2)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--rho", type=float, default=0.95)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("detail_*.png"):
        stale.unlink()  # outputs of the old cropped version

    wav = ROOT / "data" / "audio" / f"{args.song:03d}.wav"
    signal, T = load_audio([wav])
    pl = plan(T, n=args.n, rho=args.rho)
    op, _ = build(signal, pl)
    err = op.playback_error()
    A0 = materialize(op)
    print(f"operator: n={pl.n} N={pl.N} f={pl.f:.0f} Hz  "
          f"playback err={err:.2e}")

    g = matviz.gray(A0)
    save_png(g, OUT / "full_gray.png")
    save_png(g, OUT / "full_gray_16bit.png", bit_depth=16)
    palettes = ("RdBu_r", "PuOr", "coolwarm")
    for cm in palettes:
        matviz.save_rgb(matviz.diverging(A0, cm), OUT / f"full_{cm}.png")
    matviz.hinton(A0, OUT / "full_hinton.png")
    matviz.bubble(A0, OUT / "full_bubble.png")
    matviz.wireframe(A0, OUT / "full_wireframe.png")

    tiles = [("black & white", g)]
    tiles += [(f"diverging {cm}", matviz.diverging(A0, cm))
              for cm in palettes]
    tiles += [("Hinton diagram (full matrix, 8px cells)",
               load_img(OUT / "full_hinton.png")),
              ("bubble chart (full matrix, 8px cells)",
               load_img(OUT / "full_bubble.png")),
              ("3D wireframe (full matrix, stride 1)",
               load_img(OUT / "full_wireframe.png"))]

    make_sheet(tiles, OUT / "viz_modes.png", tile_size=500, cols=4, readme=f"""\
# exp_matrix_viz: display modes for the pixel-exact song operator

One operator (song {wav.name}, {T:.1f} s, n={pl.n}, N={pl.N},
f={pl.f:.0f} Hz, playback err {err:.2e}), seven ways of turning its
signed entries into ink. Every mode shows the ENTIRE matrix - the artwork
claim is that the displayed object reproduces the song, and a crop does
not, so crops are banned. The only freedom is the value -> ink mapping.
(A 3D bar mode was removed: a million bars cannot actually be rendered,
and showing a subset is against the spirit of the piece.)

What is varied: only the display mode.

- full_gray.png / full_gray_16bit.png - black & white, mid-gray = 0,
  symmetric 99.5-percentile clip, 1 entry = 1 pixel. The print master.
- full_RdBu_r / full_PuOr / full_coolwarm - diverging palettes around 0
  (sign becomes hue, magnitude becomes saturation), 1 entry = 1 pixel.
- full_hinton.png - Hinton diagram, 1 entry = 1 cell of 8x8 px
  ({8 * pl.n}px image): square area ~ |entry|, white = +, black = -.
- full_bubble.png - circle area ~ |entry|, diverging color, 8px cells.
- full_wireframe.png - the whole matrix as a height field, every row and
  column drawn (stride 1). Dense, as it must be.

Viewing note: the contact sheet downscales; a signed matrix cancels
toward flat gray when downscaled, so judge the full_*.png files at
native size (100% zoom), not the sheet tiles.
""")
    print(f"wrote {OUT}/viz_modes.png")


if __name__ == "__main__":
    main()
