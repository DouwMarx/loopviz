"""Show the two target-generator families side by side (fixed thetas, no ES).

What is varied: only the family pins (cloud vs cells vs mixed) on top of the
same base parameters - the visual identity of each generative process.

Run: .venv/bin/python scripts/exp_families.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.sheet import make_sheet
from playlistviz.targets import (FAMILIES, N_PARAMS_2D, apply_family,
                                 generate_target, theta2d_to_params)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_families"


def main() -> None:
    base = theta2d_to_params(np.zeros(N_PARAMS_2D))
    tiles = []
    for family in FAMILIES:
        img = generate_target(apply_family(base, family), 400)
        tiles.append((f"family: {family}", img))
    make_sheet(tiles, OUT / "families.png", tile_size=400, cols=3, readme="""\
# exp_families: the two generative processes

What is varied: only the family pins, on identical base parameters.

- cloud: spectral 1/f synthesis + domain warp + optional flow smear ->
  nebula / ink-wash character. Fractal, scale-free.
- cells: Worley F2-F1 crack network over a quiet spectral fill ->
  honeycomb / cell-wall character. Compositional, distance-field based.
- mixed: all layers free to blend (exploration only; usually muddier
  than either pure family).

`playlistviz embed --family {cloud,cells,mixed}` selects one; default cloud.
""")
    print(f"wrote {OUT}/families.png")


if __name__ == "__main__":
    main()
