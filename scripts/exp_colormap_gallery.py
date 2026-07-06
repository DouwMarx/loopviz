"""Colormap gallery: one matrix, every family of value -> color mapping.

The gray presentation image already encodes the signed matrix as
luminance (mid-gray = 0), so applying a colormap to it is exactly
equivalent to matviz.diverging() on the raw matrix. This gallery shows
the same candidate under diverging / sequential / cyclic maps, plus a
reversal pair, as a study aid for choosing the print palette.

Run: .venv/bin/python scripts/exp_colormap_gallery.py
       [--candidate songop_t002_f3998_n1322_c99.3]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.sheet import make_sheet

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_colormap_gallery"
Image.MAX_IMAGE_PIXELS = None

GROUPS = [
    ("diverging (zero = neutral midpoint - the natural family for a "
     "signed matrix)", ["RdBu_r", "coolwarm", "PuOr_r", "PiYG", "seismic",
                        "berlin", "vanimo", "managua"]),
    ("sequential (ordered low->high; treats the matrix as unsigned "
     "intensity)", ["viridis", "magma", "cividis", "gray"]),
    ("cyclic (wraps around; for phase-like data - shown for contrast)",
     ["twilight"]),
    ("reversal: the _r suffix flips which end is 'hot'",
     ["RdBu", "RdBu_r"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="songop_t002_f3998_n1322_c99.3")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    g = np.asarray(Image.open(ROOT / "runs" / args.candidate /
                              "presentation.png"), dtype=np.float64) / 255.0

    tiles = []
    for group, cmaps in GROUPS:
        for name in cmaps:
            try:
                cm = plt.get_cmap(name)
            except ValueError:
                continue   # colormap not in this matplotlib version
            rgb = cm(g)[..., :3]
            tiles.append((f"{name}  [{group.split(' (')[0]}]", rgb))

    make_sheet(tiles, OUT / "colormap_gallery.png", tile_size=400, cols=4,
               readme=f"""\
# exp_colormap_gallery: one matrix, every colormap family

Candidate: {args.candidate}. The gray presentation already maps the
signed matrix to luminance (mid-gray = 0, symmetric percentile clip), so
colormapping it is identical to colormapping the matrix.

Families shown:
- DIVERGING - two hues meeting at a neutral center. Correct for signed
  data with a meaningful zero (our case: mid = silence-level coupling,
  ends = strong +/- coupling). RdBu/coolwarm/PuOr are classics; berlin,
  vanimo, managua are modern perceptually uniform ones.
- SEQUENTIAL - one perceptual ramp, dark -> light. Treats values as
  unsigned intensity; sign information becomes invisible (both strong +
  and strong - map far from the middle of the ramp only if you remap to
  |value|). viridis/magma/cividis are perceptually uniform; gray is the
  print master.
- CYCLIC - ends meet (twilight). For angles/phases; included to show why
  it is wrong here: the two extremes of coupling become the same color.
- REVERSAL - any map + "_r" flips direction (RdBu vs RdBu_r decides
  whether positive coupling prints warm or cool).

Viewing note: tiles are downscaled full matrices; judge at native size.
""")
    print(f"wrote {OUT}/colormap_gallery.png ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
