"""Display modes for the pixel-exact song operator: how should the matrix
that IS the song be shown?

The operator is built once (song 2, n = 1000, rho = 0.95 - the practical
full-rank limit from exp_operator_sizing). Every mode shows the SAME
signed matrix; nothing is rendered or pooled, the only freedom is the
value -> ink mapping:

- image modes (every entry, full resolution): black & white, diverging
  palettes (RdBu, PuOr, coolwarm)
- glyph modes (readable only for small sides -> detail crops): Hinton
  diagram, bubble chart
- 3D modes (detail crops): wireframe height field, 3D bar chart

Run: .venv/bin/python scripts/exp_matrix_viz.py [--song 2] [--n 1000]
                                                [--rho 0.95]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz import matviz
from playlistviz.render import save_png
from playlistviz.sheet import make_sheet
from playlistviz.songmatrix import build, load_audio, materialize, plan

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_matrix_viz"


def load_fig(path: Path) -> np.ndarray:
    """Load a saved matplotlib figure back as a square float RGB tile."""
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

    wav = ROOT / "data" / "audio" / f"{args.song:03d}.wav"
    signal, T = load_audio([wav])
    pl = plan(T, n=args.n, rho=args.rho)
    op, _ = build(signal, pl)
    err = op.playback_error()
    A0 = materialize(op)
    print(f"operator: n={pl.n} N={pl.N} f={pl.f:.0f} Hz  "
          f"playback err={err:.2e}")

    # --- image modes, full resolution, 1 entry = 1 pixel ---
    g = matviz.gray(A0)
    save_png(g, OUT / "full_gray.png")
    save_png(g, OUT / "full_gray_16bit.png", bit_depth=16)
    palettes = ("RdBu_r", "PuOr", "coolwarm")
    for cm in palettes:
        matviz.save_rgb(matviz.diverging(A0, cm), OUT / f"full_{cm}.png")

    # --- detail crops for glyph / 3D modes ---
    i, j = matviz.best_crop(A0, 96)
    C96 = A0[i:i + 96, j:j + 96]
    C64 = C96[:64, :64]
    C40 = C96[:40, :40]
    print(f"detail crop at ({i}, {j})")

    matviz.hinton(C96, OUT / "detail_hinton.png")
    matviz.bubble(C96, OUT / "detail_bubble.png")
    matviz.wireframe(C64, OUT / "detail_wireframe.png")
    matviz.bars3d(C40, OUT / "detail_bars3d.png")

    # --- contact sheet: 1:1 crops of image modes + the glyph figures ---
    c = (args.n - 500) // 2
    sl = slice(c, c + 500)
    tiles = [("black & white (500px 1:1 crop)", g[sl, sl])]
    tiles += [(f"diverging {cm} (500px 1:1 crop)",
               matviz.diverging(A0, cm)[sl, sl]) for cm in palettes]
    tiles += [(f"Hinton diagram ({C96.shape[0]}px detail)",
               load_fig(OUT / "detail_hinton.png")),
              (f"bubble chart ({C96.shape[0]}px detail)",
               load_fig(OUT / "detail_bubble.png")),
              (f"3D wireframe ({C64.shape[0]}px detail)",
               load_fig(OUT / "detail_wireframe.png")),
              (f"3D bars ({C40.shape[0]}px detail)",
               load_fig(OUT / "detail_bars3d.png"))]

    make_sheet(tiles, OUT / "viz_modes.png", tile_size=500, cols=4, readme=f"""\
# exp_matrix_viz: display modes for the pixel-exact song operator

One operator (song {wav.name}, {T:.1f} s, n={pl.n}, N={pl.N},
f={pl.f:.0f} Hz, playback err {err:.2e}), eight ways of turning its
signed entries into ink. NOTHING is rendered or pooled - every mode shows
the exact matrix; the only freedom is the value -> ink mapping.

What is varied: only the display mode.

- full_gray.png / full_gray_16bit.png - black & white, mid-gray = 0,
  symmetric 99.5-percentile clip. The print master.
- full_RdBu_r / full_PuOr / full_coolwarm - diverging palettes around 0
  (sign becomes hue, magnitude becomes saturation).
- detail_hinton.png - square area ~ |entry|, white = +, black = -
  ({C96.shape[0]}px detail crop; unreadable at full n).
- detail_bubble.png - circle area ~ |entry|, diverging color.
- detail_wireframe.png - the matrix as a height field ({C64.shape[0]}px).
- detail_bars3d.png - 3D bars, only readable at ~{C40.shape[0]}px.

The sheet shows 1:1 crops for image modes (downscaling a signed matrix
cancels toward gray - view the full_*.png files at native size).
Detail crop anchored at the maximum-energy {C96.shape[0]}px block,
({i}, {j}).
""")
    print(f"wrote {OUT}/viz_modes.png")


if __name__ == "__main__":
    main()
