"""The anatomy of the artwork operator: A = A0 + Z P_perp, where Z carries
an embedded target image (the actual production path).

Shows block-mean and block-energy pictures of the music part A0, the free
part U (P_perp V)^T, and their sum.

Run: .venv/bin/python scripts/exp_decompose.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.embed import embed_image
from playlistviz.ingest import load_matrix
from playlistviz.operator import PlaylistOperator
from playlistviz.render import energy_image, mean_image
from playlistviz.sheet import make_sheet
from playlistviz.targets import N_PARAMS_2D, generate_target, theta2d_to_params

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_decompose"
P = 512


def unit(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-30), 0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, _ = load_matrix(ROOT / "data" / "songs.npz")
    op = PlaylistOperator.from_songs(X)

    # a representative cloud target (neutral theta), embedded
    T = generate_target(theta2d_to_params(np.zeros(N_PARAMS_2D)), P)
    res = embed_image(op, T, rank=300)

    LA, RA = op.factors()                                     # A0 alone
    LZ, RZ = res.U, res.Vp                                    # Z P_perp alone
    LS, RS = op.factors(U=res.U, Vp=res.Vp, scale=res.scale)  # the sum

    parts = {"A0 (music)": (LA, RA),
             "Z P_perp (embedded image)": (LZ, RZ),
             "A0 + Z P_perp (the artwork)": (LS, RS)}
    tiles = [(f"block mean | {name}", unit(mean_image(L, R, P)))
             for name, (L, R) in parts.items()]
    for name, (L, R) in parts.items():
        E = energy_image(L, R, P)
        tiles.append((f"block energy | {name}",
                      unit(np.log1p(E * 1e6 / max(E.mean(), 1e-30)))))

    err = op.playback_error(U=res.U, Vp=res.Vp, scale=res.scale)
    make_sheet(tiles, OUT / "decomposition.png", tile_size=P, cols=3, readme=f"""\
# exp_decompose: the anatomy of A

What is varied: nothing - one embedded cloud target; the three columns show
the additive decomposition A = A0 + Z P_perp (production path: Z carries
the embedded image, rank {res.rank}, image error {res.rel_error:.1%}).

- Row 1: block-MEAN pictures. A0's is near-blank (audio averages out inside
  blocks); the embedded image owns this channel - it IS the artwork.
- Row 2: block-ENERGY pictures. Here A0's song-block grid is visible
  (loudness correlations, silence bands from zero-padding).

Playback error with the embedded image in place: {err:.2e}.
""")
    print(f"image rel error {res.rel_error:.2%}, playback error {err:.2e}")
    print(f"wrote {OUT}/decomposition.png")


if __name__ == "__main__":
    main()
