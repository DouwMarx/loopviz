"""Design your own diverging colormap - the knobs, demonstrated.

A diverging map is two sequential ramps glued at a neutral center. Built
here in OKLab (a perceptually uniform color space: equal steps look
equal), each arm is defined by four knobs:

- hue_neg / hue_pos : the two endpoint hues (degrees on the color wheel)
- L_center, L_end   : lightness of the center vs the ends - the CENTER
                      is the field color of the print (most matrix
                      entries are near zero), so this knob decides
                      paper-light vs ink-dark artwork
- C_end             : chroma (saturation) at the ends
- drift             : hue rotation along each arm (0 = plain two-hue;
                      berlin-like richness comes from nonzero drift)

Run: .venv/bin/python scripts/exp_custom_diverging.py
       [--candidate songop_t002_f3998_n1322_c99.3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.sheet import make_sheet

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_custom_diverging"
Image.MAX_IMAGE_PIXELS = None


def oklab_to_srgb(L, a, b):
    """OKLab -> sRGB in [0,1] (vectorized, gamut-clipped)."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    lin = np.clip(np.stack([r, g, bl], axis=-1), 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * lin ** (1 / 2.4) - 0.055)


def make_diverging(hue_neg, hue_pos, L_center, L_end, C_end,
                   drift=0.0, n=511):
    """A diverging palette as an (n, 3) sRGB lookup table."""
    t = np.linspace(-1.0, 1.0, n)
    at = np.abs(t)
    L = L_center + (L_end - L_center) * at
    C = C_end * at ** 0.85                       # ease chroma toward 0
    h = np.where(t < 0, hue_neg + drift * at, hue_pos + drift * at)
    h = np.deg2rad(h)
    return oklab_to_srgb(L, C * np.cos(h), C * np.sin(h))


PALETTES = {
    "ember (dark field, blue/orange, berlin-ish)":
        dict(hue_neg=250, hue_pos=55, L_center=0.13, L_end=0.85, C_end=0.13),
    "ember, drifted arms (+40 deg hue rotation)":
        dict(hue_neg=250, hue_pos=55, L_center=0.13, L_end=0.85, C_end=0.13,
             drift=40),
    "ember, muted (half chroma)":
        dict(hue_neg=250, hue_pos=55, L_center=0.13, L_end=0.85, C_end=0.065),
    "paper (light field, same hue pair reversed roles)":
        dict(hue_neg=250, hue_pos=55, L_center=0.97, L_end=0.35, C_end=0.11),
    "wine-moss (analogous, mid-gray field)":
        dict(hue_neg=10, hue_pos=140, L_center=0.55, L_end=0.25, C_end=0.09),
    "riso duotone (saturated inks, white field)":
        dict(hue_neg=330, hue_pos=200, L_center=0.98, L_end=0.55, C_end=0.17),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="songop_t002_f3998_n1322_c99.3")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    g = np.asarray(Image.open(ROOT / "runs" / args.candidate /
                              "presentation.png"), dtype=np.float64) / 255.0

    tiles = []
    for name, knobs in PALETTES.items():
        lut = make_diverging(**knobs)
        idx = (g * (len(lut) - 1)).astype(int)
        tiles.append((name, lut[idx]))
    make_sheet(tiles, OUT / "custom_diverging.png", tile_size=430, cols=3,
               readme=f"""\
# exp_custom_diverging: designing diverging palettes in OKLab

Candidate: {args.candidate}. Each palette is generated from six knobs
(see script docstring); interpolation is in OKLab so lightness and
chroma behave perceptually, then gamut-clipped to sRGB.

What is varied per tile: the design knobs -
- ember: dark center (ink field), blue/orange ends
- drifted: same but each arm rotates hue 40 deg outward (richer, like
  matplotlib's berlin)
- muted: chroma halved (pigment-print friendly)
- paper: same hue pair with a white center (paper field)
- wine-moss: low-chroma analogous pair on a mid-gray field
- riso duotone: two saturated inks on white, in the spirit of
  risograph two-color prints

The center color owns ~95% of the print area (most entries are near
zero), so choose the field first, accents second. Tiles are downscaled
full matrices; judge at native size.
""")
    print(f"wrote {OUT}/custom_diverging.png ({len(tiles)} tiles)")


if __name__ == "__main__":
    main()
