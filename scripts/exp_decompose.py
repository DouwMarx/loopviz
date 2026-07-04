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
from playlistviz.zspace import N_PARAMS, generate_Z, song_envelopes, theta_to_params

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
    env = song_envelopes(op.X)
    zf = generate_Z(params, op.D, ZConfig(), envelopes=env,
                    project_perp=op.project_perp)

    LA, RA = op.factors()                                    # A0 alone
    LZ, RZ = zf.scale * zf.U, zf.Vp                          # Z P_perp alone
    LS, RS = op.factors(U=zf.U, Vp=zf.Vp, scale=zf.scale)    # the sum

    tiles = {}
    for name, (L, R) in {"A0": (LA, RA), "Z": (LZ, RZ), "sum": (LS, RS)}.items():
        tiles[f"mean_{name}"] = unit(mean_image(L, R, P))
        tiles[f"energy_{name}"] = unit(np.log1p(
            energy_image(L, R, P) * 1e6 / max(energy_image(L, R, P).mean(), 1e-30)))

    sheet = Image.new("L", (3 * (P + 10), 2 * (P + 10)), 30)
    order = [["mean_A0", "mean_Z", "mean_sum"],
             ["energy_A0", "energy_Z", "energy_sum"]]
    for row, names in enumerate(order):
        for col, name in enumerate(names):
            im = Image.fromarray((tiles[name] * 255).astype(np.uint8), "L")
            sheet.paste(im, (col * (P + 10) + 5, row * (P + 10) + 5))
    sheet.save(OUT / "decomposition.png")

    err = op.playback_error(U=zf.U, Vp=zf.Vp, scale=zf.scale)
    print(f"columns: A0 | Z P_perp | A0 + Z P_perp   rows: block mean | block energy")
    print(f"playback error with this Z: {err:.2e}")
    print(f"wrote {OUT}/decomposition.png")


if __name__ == "__main__":
    main()
