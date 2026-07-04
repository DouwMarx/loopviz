"""Parametric generator for the free part Z = U V^T of the playlist operator.

Z lives in the D(D-r)-dimensional family that is invisible to playback
(it is always applied through P_perp). We parameterize a low-rank slice of it
with a small theta vector: spectral 1/f^beta noise columns, optionally
amplitude-modulated by the song envelopes so the free part stays visually
tied to the actual audio, with smooth phase warping and ridge sparsification.

Everything is deterministic given (seed, theta). The white spectra and warp
displacement fields depend only on (seed, rank, D), so they are cached in a
ZGenerator and each theta evaluation costs one batched irfft plus cheap
elementwise work (this was the ES bottleneck before caching).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ZConfig

# theta layout: name -> (low, high). Values arrive in [0, 1] from the
# optimizer's sigmoid map and are affinely mapped into these ranges.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    # Z structure
    "beta_u": (0.2, 3.0),        # spectral slope of U noise columns
    "beta_v": (0.2, 3.0),        # spectral slope of V noise columns
    "log_amp": (-2.5, 1.5),      # log10 amplitude of Z relative to A0 factor scale
    "env_mix_u": (0.0, 1.0),     # envelope modulation of U by the songs
    "env_mix_v": (0.0, 1.0),
    "warp_u": (0.0, 0.8),        # smooth phase warp strength
    "warp_v": (0.0, 0.8),
    "ridge_amount": (0.0, 1.0),  # sparsifying ridge mix
    "ridge_sharp": (1.0, 16.0),
    # rendering / tone
    "energy_mix": (0.0, 1.0),    # blend of block-energy vs signed block-mean image
    "log_compress": (0.0, 2.0),  # log10 of log1p compression factor
    "gamma": (0.4, 2.4),
    "vignette": (0.0, 0.6),
    # cosine palette rgb = a + b cos(2 pi (c t + d)), 3 channels each.
    # In gray mode the palette collapses to a free nonmonotonic tone curve.
    "pal_a_r": (0.2, 0.8), "pal_a_g": (0.2, 0.8), "pal_a_b": (0.2, 0.8),
    "pal_b_r": (0.0, 0.5), "pal_b_g": (0.0, 0.5), "pal_b_b": (0.0, 0.5),
    "pal_c_r": (0.25, 1.5), "pal_c_g": (0.25, 1.5), "pal_c_b": (0.25, 1.5),
    "pal_d_r": (0.0, 1.0), "pal_d_g": (0.0, 1.0), "pal_d_b": (0.0, 1.0),
}

PARAM_NAMES = list(PARAM_RANGES)
N_PARAMS = len(PARAM_NAMES)


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


def song_envelopes(X: np.ndarray, smooth: int = 2048) -> np.ndarray:
    """Smoothed |x| envelope per song, normalized to mean 1. Shape (D, N)."""
    env = np.abs(X)
    smooth = min(smooth, X.shape[0])
    kernel = np.ones(smooth) / smooth
    for j in range(env.shape[1]):
        env[:, j] = np.convolve(env[:, j], kernel, mode="same")
        m = env[:, j].mean()
        if m > 0:
            env[:, j] /= m
    return env


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
        self.freqs = np.fft.rfftfreq(D)
        F = self.freqs.size
        self.spectra: dict[str, np.ndarray] = {}
        self.warp_disp: dict[str, np.ndarray] = {}
        for role in ("u", "v"):
            rng = np.random.default_rng([cfg.seed, ord(role), 0])
            self.spectra[role] = (rng.standard_normal((q, F))
                                  + 1j * rng.standard_normal((q, F)))
            wrng = np.random.default_rng([cfg.seed, ord(role), 1])
            wspec = (wrng.standard_normal((q, F))
                     + 1j * wrng.standard_normal((q, F)))
            with np.errstate(divide="ignore"):
                wscale = np.where(self.freqs > 0, self.freqs ** -1.25, 0.0)
            disp = np.fft.irfft(wspec * wscale, n=D, axis=1)
            disp /= np.abs(disp).max(axis=1, keepdims=True) + 1e-12
            self.warp_disp[role] = disp

    def _columns(self, role: str, beta: float, warp: float,
                 ridge_amount: float, ridge_sharp: float, env_mix: float,
                 envelopes: np.ndarray | None) -> np.ndarray:
        D, q = self.D, self.cfg.rank
        with np.errstate(divide="ignore"):
            scale = np.where(self.freqs > 0, self.freqs ** (-beta / 2.0), 0.0)
        cols = np.fft.irfft(self.spectra[role] * scale, n=D, axis=1)  # (q, D)
        sd = cols.std(axis=1, keepdims=True)
        cols /= np.where(sd > 0, sd, 1.0)

        if warp > 0:
            pos = np.clip(np.arange(D) + warp * 0.05 * D * self.warp_disp[role],
                          0, D - 1)
            i0 = pos.astype(np.int64)
            i1 = np.minimum(i0 + 1, D - 1)
            frac = pos - i0
            rows = np.arange(q)[:, None]
            cols = cols[rows, i0] * (1.0 - frac) + cols[rows, i1] * frac

        cols = (1.0 - ridge_amount) * cols + ridge_amount * np.tanh(ridge_sharp * cols)

        if envelopes is not None and env_mix > 0:
            env = envelopes[:, np.arange(q) % envelopes.shape[1]].T  # (q, D)
            cols = cols * (1.0 - env_mix + env_mix * env)

        sd = cols.std(axis=1, keepdims=True)
        cols /= np.where(sd > 0, sd, 1.0)
        return cols.T  # (D, q)

    def __call__(self, params: dict[str, float],
                 envelopes: np.ndarray | None = None,
                 project_perp=None) -> ZFactors:
        U = self._columns("u", params["beta_u"], params["warp_u"],
                          params["ridge_amount"], params["ridge_sharp"],
                          params["env_mix_u"], envelopes)
        V = self._columns("v", params["beta_v"], params["warp_v"],
                          params["ridge_amount"], params["ridge_sharp"],
                          params["env_mix_v"], envelopes)
        Vp = project_perp(V) if project_perp is not None else V
        scale = 10.0 ** params["log_amp"] / np.sqrt(self.D * self.cfg.rank)
        return ZFactors(U=U, Vp=Vp, scale=scale)


_GENERATOR_CACHE: dict[tuple[int, int, int], ZGenerator] = {}


def get_generator(D: int, cfg: ZConfig) -> ZGenerator:
    key = (D, cfg.seed, cfg.rank)
    if key not in _GENERATOR_CACHE:
        _GENERATOR_CACHE.clear()  # keep at most one; they are ~100 MB each
        _GENERATOR_CACHE[key] = ZGenerator(D, cfg)
    return _GENERATOR_CACHE[key]


def generate_Z(params: dict[str, float], D: int, cfg: ZConfig,
               envelopes: np.ndarray | None = None,
               project_perp=None) -> ZFactors:
    """Build the low-rank free part from named params (cached generator)."""
    return get_generator(D, cfg)(params, envelopes=envelopes,
                                 project_perp=project_perp)
