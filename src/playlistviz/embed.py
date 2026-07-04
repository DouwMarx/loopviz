"""Embed an arbitrary target image into the playlist operator's free part.

The block-mean picture of A is linear in Z:

    M = M0 + pool(U) . pool(P_perp V)^T,     M0 = picture of A0.

A P x P image has P^2 numbers; Z has ~D^2 free dimensions. So ANY target T
is reachable: factor the residual T_amp - M0 by SVD, lift the pixel-level
singular vectors to sample-level columns with a pooling-consistent
upsampler, and absorb the (small, rank-N) distortion from P_perp and from
smooth lifting with a few fixed-point iterations. Playback stays exact by
construction - V is always projected through P_perp.

The aesthetic problem then moves up a level: choose T (procedurally,
optimized, or any artwork); the operator carries it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operator import PlaylistOperator
from .render import mean_image, pool_rows


def lift_const(pooled: np.ndarray, D: int, P: int) -> np.ndarray:
    """Piecewise-constant lift: exact right inverse of pool_rows."""
    k = D // P
    out = np.zeros((D, pooled.shape[1]))
    out[: P * k] = np.repeat(pooled, k, axis=0)
    return out


def lift(pooled: np.ndarray, D: int, P: int) -> np.ndarray:
    """(P, q) pixel-level columns -> (D, q) sample-level.

    Smooth linear-interpolation base plus one piecewise-constant correction,
    so pool_rows(lift(u)) == u exactly while staying smooth except for the
    (small) interpolation residual.
    """
    k = D // P
    centers = (np.arange(P) + 0.5) * k
    t = np.arange(D, dtype=np.float64)
    out = np.empty((D, pooled.shape[1]))
    for j in range(pooled.shape[1]):
        out[:, j] = np.interp(t, centers, pooled[:, j])
    out += lift_const(pooled - pool_rows(out, P), D, P)
    return out


@dataclass
class EmbedResult:
    U: np.ndarray
    Vp: np.ndarray            # P_perp-projected
    scale: float              # always 1.0; amplitudes are baked into U, Vp
    achieved: np.ndarray      # pooled mean image of A0 + U Vp^T at P
    target_amp: np.ndarray    # the amplitude-mapped target it was aiming at
    rel_error: float          # ||achieved - target_amp|| / ||target_amp||
    rank: int


def embed_image(op: PlaylistOperator, T: np.ndarray, rank: int = 160,
                amp_factor: float = 4.0, n_iter: int = 3) -> EmbedResult:
    """Make the block-mean picture of A equal (an affine map of) T.

    T: (P, P) target in [0, 1]. rank: SVD truncation of the residual;
    amp_factor: embedded amplitude relative to A0's picture, so the target
    dominates visually without erasing the music's own structure.
    """
    T = np.asarray(T, dtype=np.float64)
    P = T.shape[0]
    if T.shape != (P, P):
        raise ValueError("target must be square")
    if P > op.D:
        raise ValueError(f"target resolution {P} exceeds D={op.D}")

    L1, R1 = op.factors()
    M0 = mean_image(L1, R1, P)

    Tc = T - T.mean()
    m = np.abs(Tc).max()
    if m < 1e-12:
        raise ValueError("target image is constant")
    target_amp = Tc * (amp_factor * np.abs(M0).max() / m)

    resid = target_amp - M0
    W, s, Vt = np.linalg.svd(resid, full_matrices=False)
    rank = min(rank, P)
    energy = float((s[:rank] ** 2).sum() / (s**2).sum())
    Wq = W[:, :rank] * np.sqrt(s[:rank])
    Vq = Vt[:rank].T * np.sqrt(s[:rank])

    U = lift(Wq, op.D, P)  # exact: pool(U) == Wq

    # lift V and absorb the P_perp correction, which is rank-N and therefore
    # small and contractive under this iteration
    V = lift(Vq, op.D, P)
    Vp = op.project_perp(V)
    for _ in range(n_iter):
        V += lift(Vq - pool_rows(Vp, P), op.D, P)
        Vp = op.project_perp(V)

    achieved = M0 + mean_image(U, Vp, P)
    rel = float(np.linalg.norm(achieved - target_amp)
                / np.linalg.norm(target_amp))
    _ = energy  # rank energy retained; exposed via rel_error which subsumes it
    return EmbedResult(U=U, Vp=Vp, scale=1.0, achieved=achieved,
                       target_amp=target_amp, rel_error=rel, rank=rank)


def display(achieved: np.ndarray) -> np.ndarray:
    """Affine-normalize an achieved mean image back to [0, 1] for output."""
    lo, hi = np.percentile(achieved, [0.5, 99.5])
    if hi - lo < 1e-30:
        return np.zeros_like(achieved)
    return np.clip((achieved - lo) / (hi - lo), 0.0, 1.0)
