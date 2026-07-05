"""Show the anatomy of A: the music part A0, the free part Z P_perp, and
their sum, in both pixel statistics (block mean and block energy).

Run: .venv/bin/python scripts/exp_decompose.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.config import ZConfig
from playlistviz.ingest import load_matrix
from playlistviz.operator import PlaylistOperator
from playlistviz.render import energy_image, mean_image
from playlistviz.zspace import N_PARAMS, generate_Z, theta_to_params

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_decompose"
P = 512


def unit(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / max(hi - lo, 1e-30), 0, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X, meta = load_matrix(ROOT / "data" / "songs.npz")
    op = PlaylistOperator.from_songs(X)

    # a hand-picked organic theta (the "smooth_local" regime)
    params = dict(theta_to_params(np.zeros(N_PARAMS)),
                  beta_u=2.6, beta_v=2.6, beta_spread=0.3, locality=0.9,
                  log_width=-1.3, width_spread=0.6, ridge_amount=0.15,
                  log_amp=0.0)
    zf = generate_Z(params, op.D, ZConfig(), project_perp=op.project_perp)

    LA, RA = op.factors()                                    # A0 alone
    LZ, RZ = zf.scale * zf.U, zf.Vp                          # Z P_perp alone
    LS, RS = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)    # the sum

    from playlistviz.sheet import make_sheet

    tiles = []
    for name, (L, R) in {"A0 (music)": (LA, RA),
                         "Z P_perp (free)": (LZ, RZ),
                         "A0 + Z P_perp": (LS, RS)}.items():
        tiles.append((f"block mean | {name}", unit(mean_image(L, R, P))))
    for name, (L, R) in {"A0 (music)": (LA, RA),
                         "Z P_perp (free)": (LZ, RZ),
                         "A0 + Z P_perp": (LS, RS)}.items():
        E = energy_image(L, R, P)
        tiles.append((f"block energy | {name}",
                      unit(np.log1p(E * 1e6 / max(E.mean(), 1e-30)))))

    err = op.playback_error(U=zf.U, Vp=zf.Vp, scale=zf.scale)
    make_sheet(tiles, OUT / "decomposition.png", tile_size=P, cols=3, readme=f"""\
# exp_decompose: the anatomy of A

What is varied: nothing - one fixed organic Z theta; the three columns show
the additive decomposition A = A0 + Z P_perp.

- Row 1: block-MEAN picture (average of matrix entries per pixel block).
  A0's mean picture is near zero because raw audio oscillates within any
  block; the free part Z dominates this channel.
- Row 2: block-ENERGY picture (mean square per block, log-toned). Here A0's
  song-block grid is visible, including dark bands from zero-padding.

Playback error with this Z: {err:.2e} (the free part is inaudible).
""")
    print(f"playback error with this Z: {err:.2e}")
    print(f"wrote {OUT}/decomposition.png")


if __name__ == "__main__":
    main()
