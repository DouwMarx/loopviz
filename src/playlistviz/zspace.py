"""Parametric generator for the free part Z = U V^T of the playlist operator.

Z lives in the D(D-r)-dimensional family that is invisible to playback
(it is always applied through P_perp). We parameterize a low-rank slice of it
with a small theta vector: spectral 1/f^beta noise columns, optionally
amplitude-modulated by the song envelopes so the free part stays visually
tied to the actual audio, with smooth phase warping and ridge sparsification.

Everything is deterministic given (seed, theta).
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
    # cosine palette rgb = a + b cos(2 pi (c t + d)), 3 channels each
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


def _spectral_noise(rng: np.random.Generator, n: int, beta: float) -> np.ndarray:
    """1/f^beta noise of length n, unit std."""
    freqs = np.fft.rfftfreq(n)
    spec = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(freqs.size)
    with np.errstate(divide="ignore"):
        scale = np.where(freqs > 0, freqs ** (-beta / 2.0), 0.0)
    x = np.fft.irfft(spec * scale, n=n)
    sd = x.std()
    return x / sd if sd > 0 else x


def _smooth_warp(rng: np.random.Generator, n: int, strength: float) -> np.ndarray:
    """Monotone-ish smooth warp of sample positions in [0, n)."""
    if strength <= 0:
        return np.arange(n, dtype=np.float64)
    disp = _spectral_noise(rng, n, beta=2.5)
    disp = disp / (np.abs(disp).max() + 1e-12)
    return np.clip(np.arange(n) + strength * 0.05 * n * disp, 0, n - 1)


def _ridge(x: np.ndarray, amount: float, sharp: float) -> np.ndarray:
    """Mix in a tanh-sharpened copy: a soft sparsifying nonlinearity."""
    return (1.0 - amount) * x + amount * np.tanh(sharp * x)


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


def generate_Z(params: dict[str, float], D: int, cfg: ZConfig,
               envelopes: np.ndarray | None = None,
               project_perp=None) -> ZFactors:
    """Build the low-rank free part from named params.

    project_perp: callable applying P_perp to a (D, q) matrix; if None, V is
    used unprojected (only acceptable in tests).
    """
    q = cfg.rank
    U = np.empty((D, q))
    V = np.empty((D, q))
    for role, mat, beta, warp, env_mix in (
        ("u", U, params["beta_u"], params["warp_u"], params["env_mix_u"]),
        ("v", V, params["beta_v"], params["warp_v"], params["env_mix_v"]),
    ):
        for k in range(q):
            rng = np.random.default_rng([cfg.seed, ord(role), k])
            col = _spectral_noise(rng, D, beta)
            pos = _smooth_warp(rng, D, warp)
            col = np.interp(pos, np.arange(D), col)
            col = _ridge(col, params["ridge_amount"], params["ridge_sharp"])
            if envelopes is not None and env_mix > 0:
                env = envelopes[:, k % envelopes.shape[1]]
                col = col * (1.0 - env_mix + env_mix * env)
            sd = col.std()
            mat[:, k] = col / sd if sd > 0 else col

    Vp = project_perp(V) if project_perp is not None else V
    scale = 10.0 ** params["log_amp"] / np.sqrt(D * q)
    return ZFactors(U=U, Vp=Vp, scale=scale)
