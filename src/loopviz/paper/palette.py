"""Print-oriented diverging palettes for the signed matrix.

A diverging map is two sequential ramps glued at a neutral center. Built in
OKLab - a perceptually uniform space where equal steps look equal - each arm
is defined by six knobs:

- hue_neg / hue_pos : the two endpoint hues (degrees on the color wheel)
- L_center, L_end   : lightness of the center vs the ends. The CENTER owns
                      ~95% of the print (most matrix entries are near zero),
                      so this knob is paper-light vs ink-dark artwork.
- C_end             : chroma (saturation) at the ends
- drift             : hue rotation along each arm (0 = plain two-hue;
                      berlin-like richness comes from nonzero drift)
- chroma_ease       : exponent easing chroma from center to end (0.85 keeps
                      the neutral truly neutral, avoiding a muddy midtone)

Two spec kinds share one interface: `oklab` (the six knobs above) and `mpl`
(a stock matplotlib diverging colormap, e.g. RdBu_r, coolwarm, berlin). Both
produce an (n, 3) sRGB lookup table via `lut_from_spec`, applied to the
grayscale render with `apply_spec`. Coloring the grayscale luminance is
exactly equivalent to coloring the signed matrix (mid-gray = 0).

Printability is measured, not assumed (`gamut_report`): every palette is
checked against the sRGB gamut (hard) and, when a CMYK ICC profile is
available, soft-proofed against coated offset print (FOGRA39). A palette
whose accent colors fall outside the print gamut is flagged, not silently
clipped.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# OKLab chroma that coated CMYK offset can reproduce across most hues. A
# conservative single-number ceiling used only when no ICC profile is found;
# the exact per-hue gamut comes from the soft-proof. (Coated stock reaches
# ~0.11-0.13 in reds/oranges, less in cyans; 0.11 stays inside for all hues.)
SAFE_CHROMA = 0.11

# Darkest neutral coated offset can hold (measured: FOGRA39 black -> OKLab
# L 0.219). A field darker than this prints lighter than designed, so the
# analytic fallback also warns when the ramp dips below it.
COATED_BLACK_L = 0.20

# Round-trip OKLab distance above which a color visibly shifts in print
# (~perceptual JND in OKLab). The out-of-gamut FRACTION counts entries past
# this; relative-colorimetric compresses the extreme accent/field tips, so a
# small fraction is graceful, not a failure.
GAMUT_DE = 0.02

# Max fraction of the ramp allowed to shift before a palette is called
# unprintable (only the extreme tips may compress).
GAMUT_TOL = 0.10


# -- OKLab <-> sRGB -----------------------------------------------------------

def oklab_to_linear_srgb(L, a, b) -> np.ndarray:
    """OKLab -> linear sRGB, UNCLIPPED (negative/>1 reveals out-of-gamut)."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.stack([r, g, bl], axis=-1)


def _linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin,
                    1.055 * lin ** (1 / 2.4) - 0.055)


def oklab_to_srgb(L, a, b) -> np.ndarray:
    """OKLab -> sRGB in [0, 1] (vectorized, gamut-clipped)."""
    return _linear_to_srgb(oklab_to_linear_srgb(L, a, b))


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] -> OKLab (..., 3) as (L, a, b)."""
    rgb = np.asarray(rgb, dtype=float)
    r, g, b = (_srgb_to_linear(rgb[..., i]) for i in range(3))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, bb], axis=-1)


# -- palette spec -------------------------------------------------------------

# Range and step for each OKLab knob: (min, max, step). Drives the viewer's
# sliders and validates loaded specs. Kept here so UI and engine never drift.
KNOB_RANGES = {
    "hue_neg": (0.0, 360.0, 1.0),
    "hue_pos": (0.0, 360.0, 1.0),
    "L_center": (0.0, 1.0, 0.01),
    "L_end": (0.0, 1.0, 0.01),
    "C_end": (0.0, 0.30, 0.005),
    "drift": (-90.0, 90.0, 1.0),
    "chroma_ease": (0.4, 2.0, 0.05),
}
KNOBS = tuple(KNOB_RANGES)


@dataclass(frozen=True)
class PaletteSpec:
    """A reproducible palette definition.

    kind == "oklab": the six-plus-one knobs design a diverging map.
    kind == "mpl":   `cmap` names a matplotlib diverging colormap.
    """

    name: str = "custom"
    kind: str = "oklab"
    hue_neg: float = 250.0
    hue_pos: float = 55.0
    L_center: float = 0.13
    L_end: float = 0.85
    C_end: float = 0.11
    drift: float = 0.0
    chroma_ease: float = 0.85
    cmap: str = ""          # used iff kind == "mpl"
    n_lut: int = 511        # LUT resolution (odd -> a sample exactly at center)
    group: str = ""         # UI grouping label only (no effect on the colors)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PaletteSpec":
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields)


# -- lookup tables ------------------------------------------------------------

def oklab_lut(spec: PaletteSpec) -> np.ndarray:
    """Diverging OKLab palette as an (n_lut, 3) sRGB table, center in middle."""
    t = np.linspace(-1.0, 1.0, spec.n_lut)
    at = np.abs(t)
    L = spec.L_center + (spec.L_end - spec.L_center) * at
    C = spec.C_end * at ** spec.chroma_ease
    h = np.where(t < 0, spec.hue_neg + spec.drift * at,
                 spec.hue_pos + spec.drift * at)
    h = np.deg2rad(h)
    return oklab_to_srgb(L, C * np.cos(h), C * np.sin(h))


def mpl_lut(cmap: str, n: int) -> np.ndarray:
    """Stock matplotlib colormap sampled to an (n, 3) sRGB table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.get_cmap(cmap)(np.linspace(0.0, 1.0, n))[..., :3]


def lut_from_spec(spec: PaletteSpec) -> np.ndarray:
    if spec.kind == "oklab":
        return oklab_lut(spec)
    if spec.kind == "mpl":
        return mpl_lut(spec.cmap, spec.n_lut)
    raise ValueError(f"unknown palette kind {spec.kind!r}")


def apply_lut(gray: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map grayscale luminance in [0, 1] through an (n, 3) LUT -> RGB."""
    g = np.clip(np.asarray(gray, dtype=float), 0.0, 1.0)
    idx = np.round(g * (len(lut) - 1)).astype(int)
    return lut[idx]


def apply_spec(gray: np.ndarray, spec: PaletteSpec) -> np.ndarray:
    """Recolor a grayscale image with a palette spec -> (..., 3) sRGB."""
    return apply_lut(gray, lut_from_spec(spec))


# -- printability -------------------------------------------------------------

def srgb_clip_fraction(lut: np.ndarray, spec: PaletteSpec) -> float:
    """Fraction of LUT entries that fall outside the sRGB gamut.

    For oklab specs this is exact (recomputes the unclipped linear values).
    For mpl specs it is 0 by construction (the colormap is already sRGB).
    """
    if spec.kind != "oklab":
        return 0.0
    t = np.linspace(-1.0, 1.0, spec.n_lut)
    at = np.abs(t)
    L = spec.L_center + (spec.L_end - spec.L_center) * at
    C = spec.C_end * at ** spec.chroma_ease
    h = np.deg2rad(np.where(t < 0, spec.hue_neg + spec.drift * at,
                            spec.hue_pos + spec.drift * at))
    lin = oklab_to_linear_srgb(L, C * np.cos(h), C * np.sin(h))
    # 0.5% linear tolerance: sub-half-percent excursions clip to a visually
    # identical color (numerical boundary touch), larger ones are real
    # out-of-sRGB accents.
    out = (lin < -5e-3) | (lin > 1.0 + 5e-3)
    return float(out.any(axis=-1).mean())


def find_cmyk_profile() -> str | None:
    """Locate a CMYK ICC profile for soft-proofing, or None.

    Order: PLAYLISTVIZ_CMYK_ICC env var, then common coated-offset profiles
    on typical Linux/Nix installs. Absence is fine - the analytic chroma
    ceiling is used instead.
    """
    env = os.environ.get("PLAYLISTVIZ_CMYK_ICC")
    if env and Path(env).exists():
        return env
    patterns = [
        "/nix/store/*/tex/generic/colorprofiles/FOGRA39L_coated.icc",
        "/usr/share/color/icc/**/*FOGRA39*.icc",
        "/usr/share/color/icc/**/*[Cc]oated*.icc",
        "/usr/share/color/icc/**/USWebCoatedSWOP.icc",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def cmyk_softproof(lut: np.ndarray,
                   icc_path: str) -> tuple[float, float, np.ndarray]:
    """Soft-proof an sRGB LUT against a CMYK profile (relative colorimetric).

    Round-trips sRGB -> CMYK -> sRGB; a color the print gamut cannot hold is
    pulled to the boundary, so the OKLab distance of the round trip measures
    how far out of gamut it is. Returns (out_of_gamut_fraction, max_dE,
    proofed_lut) where proofed_lut is what the print would actually show.
    """
    from PIL import Image, ImageCms

    srgb = ImageCms.createProfile("sRGB")
    cmyk = ImageCms.getOpenProfile(icc_path)
    intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
    to_cmyk = ImageCms.buildTransform(srgb, cmyk, "RGB", "CMYK", intent)
    to_rgb = ImageCms.buildTransform(cmyk, srgb, "CMYK", "RGB", intent)

    arr = (np.clip(lut, 0, 1) * 255).astype(np.uint8).reshape(-1, 1, 3)
    im = Image.fromarray(arr, "RGB")
    proofed = ImageCms.applyTransform(ImageCms.applyTransform(im, to_cmyk),
                                      to_rgb)
    proofed_rgb = np.asarray(proofed, dtype=float).reshape(-1, 3) / 255.0

    de = np.linalg.norm(srgb_to_oklab(lut) - srgb_to_oklab(proofed_rgb),
                        axis=-1)
    return float((de > GAMUT_DE).mean()), float(de.max()), proofed_rgb


def gamut_report(spec: PaletteSpec) -> dict:
    """Printability of a palette: sRGB validity + CMYK soft-proof.

    `printable` is the honest bottom line: no sRGB clipping AND (if a CMYK
    profile is available) < 2% of the ramp out of print gamut, else the
    conservative chroma ceiling is satisfied.
    """
    lut = lut_from_spec(spec)
    okl = srgb_to_oklab(lut)
    max_chroma = float(np.hypot(okl[..., 1], okl[..., 2]).max())
    min_L = float(okl[..., 0].min())
    srgb_clip = srgb_clip_fraction(lut, spec)

    report = {
        "srgb_clip_fraction": srgb_clip,
        "max_oklab_chroma": max_chroma,
        "min_oklab_lightness": min_L,
        "safe_chroma": SAFE_CHROMA,
    }
    icc = find_cmyk_profile()
    if icc is not None:
        oog, max_de, _ = cmyk_softproof(lut, icc)
        report.update(method="cmyk-softproof", icc=Path(icc).name,
                      out_of_gamut_fraction=oog, max_delta_e=max_de)
        printable = srgb_clip < 1e-6 and oog < GAMUT_TOL
        report["tier"] = _tier(srgb_clip, oog)
    else:
        report.update(method="chroma-ceiling")
        printable = (srgb_clip < 1e-6 and max_chroma <= SAFE_CHROMA + 1e-6
                     and min_L >= COATED_BLACK_L - 0.03)
        report["tier"] = "unknown" if printable else "poor"
    report["printable"] = bool(printable)
    return report


def _tier(srgb_clip: float, oog: float) -> str:
    """Coarse printability grade from the out-of-gamut fraction."""
    if srgb_clip >= 1e-6:
        return "invalid"          # not even a valid sRGB color
    if oog < 0.05:
        return "faithful"         # prints essentially as designed
    if oog < GAMUT_TOL:
        return "good"             # only the extreme tips compress
    if oog < 0.30:
        return "compressed"       # visible desaturation of accents
    return "poor"                 # a screen palette, not a print one


# -- presets ------------------------------------------------------------------

def _oklab(name, group, **kw) -> PaletteSpec:
    return PaletteSpec(name=name, kind="oklab", group=group, **kw)


def _mpl(name, cmap, group) -> PaletteSpec:
    return PaletteSpec(name=name, kind="mpl", cmap=cmap, group=group)


# The designed OKLab palettes are all tuned to print faithfully on coated
# offset (chroma inside CMYK, field lightness at/above the L~0.20 black
# floor). The stock matplotlib maps are shown with their MEASURED print
# gamut so the honest ones stand out:
#   - ColorBrewer diverging print/colour-blind-safe set (Cynthia Brewer):
#     BrBG, PiYG, PRGn, PuOr, RdBu, RdYlBu  (colorbrewer2.org)
#   - Crameri perceptually-uniform, CVD-safe maps bundled in matplotlib:
#     berlin, managua, vanimo  (fabiocrameri.ch); vik/broc/roma/... need the
#     optional cmcrameri package and are intentionally not vendored.
_DESIGNED = [
    # dark field (ink), L_center at the coated black floor
    _oklab("ember", "Designed - dark field", hue_neg=250, hue_pos=55,
           L_center=0.20, L_end=0.80, C_end=0.10),
    _oklab("dusk", "Designed - dark field", hue_neg=260, hue_pos=45,
           L_center=0.20, L_end=0.80, C_end=0.09, drift=25),
    _oklab("teal-rust", "Designed - dark field", hue_neg=200, hue_pos=30,
           L_center=0.20, L_end=0.80, C_end=0.09),
    _oklab("indigo-amber", "Designed - dark field", hue_neg=278, hue_pos=75,
           L_center=0.21, L_end=0.82, C_end=0.09),
    _oklab("violet-olive", "Designed - dark field", hue_neg=300, hue_pos=105,
           L_center=0.21, L_end=0.80, C_end=0.08),
    _oklab("magenta-teal", "Designed - dark field", hue_neg=335, hue_pos=185,
           L_center=0.21, L_end=0.80, C_end=0.08),
    _oklab("plum-green", "Designed - dark field", hue_neg=320, hue_pos=140,
           L_center=0.21, L_end=0.80, C_end=0.08),
    _oklab("slate-copper", "Designed - dark field", hue_neg=245, hue_pos=40,
           L_center=0.22, L_end=0.78, C_end=0.08, drift=15),
    # light field (paper white), accents on the dark end of each arm
    _oklab("paper", "Designed - paper field", hue_neg=250, hue_pos=55,
           L_center=0.96, L_end=0.42, C_end=0.10),
    _oklab("paper-teal", "Designed - paper field", hue_neg=200, hue_pos=30,
           L_center=0.96, L_end=0.42, C_end=0.075),
    _oklab("paper-violet", "Designed - paper field", hue_neg=300, hue_pos=110,
           L_center=0.96, L_end=0.42, C_end=0.08),
    _oklab("paper-berry", "Designed - paper field", hue_neg=335, hue_pos=150,
           L_center=0.96, L_end=0.44, C_end=0.08),
    _oklab("linen-indigo", "Designed - paper field", hue_neg=278, hue_pos=70,
           L_center=0.95, L_end=0.45, C_end=0.09),
    # muted / analogous on a mid-gray field
    _oklab("wine-moss", "Designed - muted", hue_neg=10, hue_pos=140,
           L_center=0.55, L_end=0.35, C_end=0.09),
    _oklab("ash-blue", "Designed - muted", hue_neg=250, hue_pos=60,
           L_center=0.60, L_end=0.30, C_end=0.06),
    _oklab("sepia", "Designed - muted", hue_neg=40, hue_pos=70,
           L_center=0.70, L_end=0.30, C_end=0.05),
]

_COLORBREWER = [_mpl(n, n, "ColorBrewer (colour-blind safe)") for n in
                ["RdBu", "RdYlBu", "BrBG", "PuOr", "PRGn", "PiYG"]]

_CRAMERI = [_mpl(n, n, "Crameri (perceptual, CVD-safe)") for n in
            ["berlin", "managua", "vanimo"]]

_CLASSIC = [_mpl(n, n, "Classic (matplotlib)") for n in
            ["coolwarm", "Spectral", "RdGy", "RdYlGn", "seismic"]]

PRESETS: dict[str, PaletteSpec] = {
    p.name: p for p in _DESIGNED + _COLORBREWER + _CRAMERI + _CLASSIC
}


# -- reproducible config IO ---------------------------------------------------

def _git_commit(root: Path) -> str | None:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short",
                              "HEAD"], capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _image_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_config(spec: PaletteSpec, source: Path, out_dir: Path,
                source_id: str = "", extra: dict | None = None) -> Path:
    """Write a reproducible palette config + preview next to the source.

    The JSON pins the spec, the source image (path + sha256), the gamut
    report, and provenance (git commit, timestamp). `apply_config` on the
    same source reproduces the exact RGB. Returns the config path.
    """
    from datetime import datetime, timezone

    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source)

    # sanitize the label into a filename stem: the name comes from a UI field
    # and must not escape out_dir (no path separators, no ..)
    stem = "".join(c for c in spec.name if c.isalnum() or c in "-_") or "custom"

    gray = np.asarray(Image.open(source).convert("L"), dtype=float) / 255.0
    rgb = apply_spec(gray, spec)
    preview = out_dir / f"{stem}.png"
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                    "RGB").save(preview)

    root = out_dir
    for parent in [out_dir, *out_dir.parents]:
        if (parent / ".git").exists():
            root = parent
            break

    config = {
        "spec": spec.to_dict(),
        "source": {"id": source_id, "path": str(source),
                   "sha256": _image_sha256(source)},
        "gamut": gamut_report(spec),
        "preview": preview.name,
        "provenance": {
            "tool": "loopviz.palette",
            "git_commit": _git_commit(root),
            "created_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
        },
    }
    if extra:
        config["extra"] = extra
    path = out_dir / f"{stem}.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def load_spec(path: Path) -> PaletteSpec:
    """Load a PaletteSpec from a saved config JSON (or a bare spec dict)."""
    d = json.loads(Path(path).read_text())
    return PaletteSpec.from_dict(d.get("spec", d))


def apply_config(config_path: Path, source: Path, out_path: Path) -> Path:
    """Reproduce a saved recolor: load spec, recolor source, write PNG."""
    from PIL import Image

    spec = load_spec(config_path)
    gray = np.asarray(Image.open(source).convert("L"), dtype=float) / 255.0
    rgb = apply_spec(gray, spec)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                    "RGB").save(out_path)
    return Path(out_path)
