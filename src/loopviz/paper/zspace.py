"""Parametric generator for the free part Z = U V^T of the playlist operator.

The math. All exact playlist operators form the family

    A = A0 + Z P_perp,     A0 = X_next G^{-1} X^T,   P_perp = I - X G^{-1} X^T.

P_perp annihilates every song, so Z (a D x D matrix, D(D-r) free dimensions)
is *inaudible*: it only acts on vectors outside the song span. We spend it on
the picture. Materializing a D x D Z is impossible, so we take a rank-q slice

    Z = U V^T,   U, V in R^{D x q},

and generate U, V procedurally from a small parameter vector theta.

Column model (the image is built from outer products col_U * col_V^T, so
column structure is directly visible texture):

    col_k = window_k(t) * noise_k(t), then ridge-sharpened

- noise_k: 1/f^beta_k spectral noise; beta_k = beta_center +- beta_spread
  (per-column offsets cached), so columns span a range of roughnesses.
- window_k: mixture of Gaussian bumps at cached random centers with widths
  set by theta. locality=0 -> global support (full-length streaks in the
  image); locality=1 -> compact blobs. This is the anti-"line-ey" lever:
  a localized column contributes a local patch, not a full-width line.

Z depends only on (seed, theta), never on the audio: the artwork's claim is
that A plays the playlist exactly, and the free part stays mathematically
independent of it.

Everything is deterministic given (seed, theta). The theta-independent
randomness (white spectra, bump centers, per-column offsets) is cached in a
ZGenerator; each theta evaluation costs one batched irfft plus elementwise
work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ZConfig

# theta layout: name -> (low, high). Values arrive in [0, 1] from the
# optimizer's sigmoid map and are affinely mapped into these ranges.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    # Z column structure
    "beta_u": (0.2, 3.6),          # spectral slope center, U columns
    "beta_v": (0.2, 3.6),          # spectral slope center, V columns
    "beta_spread": (0.0, 1.2),     # per-column slope diversity
    "locality": (0.0, 0.97),       # 0 global streaks -> 1 compact blobs
    "log_width": (-3.3, -0.7),     # log10 bump width as fraction of D
    "width_spread": (0.0, 1.0),    # per-column width diversity (decades)
    "log_amp": (-2.5, 1.5),        # log10 amplitude of Z relative to A0
    "ridge_amount": (0.0, 1.0),    # sparsifying tanh mix
    "ridge_sharp": (1.0, 16.0),
    # rendering / tone (grayscale)
    "energy_mix": (0.0, 1.0),      # blend of block-energy vs signed block-mean
    "log_compress": (0.0, 2.0),    # log10 of log1p compression factor
    "gamma": (0.4, 2.4),
    "vignette": (0.0, 0.6),
    # free nonmonotonic tone curve: y = t + sum_i b_i cos(2 pi (c_i t + d_i))
    "tone_b1": (0.0, 0.35), "tone_c1": (0.25, 2.0), "tone_d1": (0.0, 1.0),
    "tone_b2": (0.0, 0.35), "tone_c2": (0.25, 2.0), "tone_d2": (0.0, 1.0),
}

PARAM_NAMES = list(PARAM_RANGES)
N_PARAMS = len(PARAM_NAMES)

_BUMPS_PER_COLUMN = 3


def theta_to_params(theta: np.ndarray) -> dict[str, float]:
    """Map raw theta in R^n through sigmoid into named parameter ranges."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (N_PARAMS,):
        raise ValueError(f"theta must have shape ({N_PARAMS},), got {theta.shape}")
    unit = 1.0 / (1.0 + np.exp(-theta))
    out = {}
    for u, name in zip(unit, PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        out[name] = float(lo + (hi - lo) * u)
    return out


@dataclass
class ZFactors:
    U: np.ndarray      # (D, q)
    Vp: np.ndarray     # (D, q), already P_perp-projected
    scale: float       # amplitude applied when composing A = A0 + scale * U Vp^T


class ZGenerator:
    """Caches the theta-independent randomness for one (seed, rank, D)."""

    def __init__(self, D: int, cfg: ZConfig):
        self.D, self.cfg = D, cfg
        q = cfg.rank
        freqs = np.fft.rfftfreq(D)
        with np.errstate(divide="ignore"):
            self.logf = np.log(np.where(freqs > 0, freqs, 1.0))
        self.spectra: dict[str, np.ndarray] = {}
        self.centers: dict[str, np.ndarray] = {}
        self.beta_off: dict[str, np.ndarray] = {}
        self.width_off: dict[str, np.ndarray] = {}
        for role in ("u", "v"):
            rng = np.random.default_rng([cfg.seed, ord(role)])
            self.spectra[role] = (rng.standard_normal((q, freqs.size))
                                  + 1j * rng.standard_normal((q, freqs.size)))
            self.spectra[role][:, 0] = 0.0
            self.centers[role] = rng.uniform(0, D, size=(q, _BUMPS_PER_COLUMN))
            self.beta_off[role] = rng.uniform(-1, 1, size=q)
            self.width_off[role] = rng.uniform(-1, 1, size=q)

    def _columns(self, role: str, params: dict[str, float]) -> np.ndarray:
        D, q = self.D, self.cfg.rank
        p = params

        beta = np.clip(p[f"beta_{role}"] + p["beta_spread"] * self.beta_off[role],
                       0.05, 4.0)
        power = np.exp(-0.5 * beta[:, None] * self.logf[None, :])
        power[:, 0] = 0.0
        cols = np.fft.irfft(self.spectra[role] * power, n=D, axis=1)  # (q, D)
        sd = cols.std(axis=1, keepdims=True)
        cols /= np.where(sd > 0, sd, 1.0)

        loc = p["locality"]
        if loc > 0:
            sigma = D * 10.0 ** (p["log_width"]
                                 + p["width_spread"] * self.width_off[role])
            t = np.arange(D)
            bump = np.zeros((q, D))
            for b in range(_BUMPS_PER_COLUMN):
                d = t[None, :] - self.centers[role][:, b:b + 1]
                bump += np.exp(-0.5 * (d / sigma[:, None]) ** 2)
            bump /= bump.max(axis=1, keepdims=True) + 1e-12
            cols = cols * ((1.0 - loc) + loc * bump)

        r = p["ridge_amount"]
        if r > 0:
            cols = (1.0 - r) * cols + r * np.tanh(p["ridge_sharp"] * cols)

        sd = cols.std(axis=1, keepdims=True)
        cols /= np.where(sd > 0, sd, 1.0)
        return cols.T  # (D, q)

    def __call__(self, params: dict[str, float],
                 project_perp=None) -> ZFactors:
        U = self._columns("u", params)
        V = self._columns("v", params)
        Vp = project_perp(V) if project_perp is not None else V
        scale = 10.0 ** params["log_amp"] / np.sqrt(self.D * self.cfg.rank)
        return ZFactors(U=U, Vp=Vp, scale=scale)


_GENERATOR_CACHE: dict[tuple[int, int, int], ZGenerator] = {}


def get_generator(D: int, cfg: ZConfig) -> ZGenerator:
    key = (D, cfg.seed, cfg.rank)
    if key not in _GENERATOR_CACHE:
        _GENERATOR_CACHE.clear()  # keep at most one; they can be large
        _GENERATOR_CACHE[key] = ZGenerator(D, cfg)
    return _GENERATOR_CACHE[key]


def generate_Z(params: dict[str, float], D: int, cfg: ZConfig,
               project_perp=None) -> ZFactors:
    """Build the low-rank free part from named params (cached generator)."""
    return get_generator(D, cfg)(params, project_perp=project_perp)
