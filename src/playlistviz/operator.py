"""The cyclic playlist operator, kept in factored form.

A = A0 + Z P_perp with
    A0     = X_next G^{-1} X^T          (min-norm exact solution, rank N)
    P_perp = I - X G^{-1} X^T           (projector onto the orthogonal complement of span X)

A is never materialized. Everything streams through the factors:
applying A costs O(DN); rendering costs O(pixels * rank^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlaylistOperator:
    """Factored representation of the playlist operator.

    X       : (D, N) song matrix, columns are songs in playlist order
    Ginv    : (N, N) inverse Gram matrix (X^T X)^{-1}
    """

    X: np.ndarray
    Ginv: np.ndarray

    @classmethod
    def from_songs(cls, X: np.ndarray, rcond: float = 1e-10) -> "PlaylistOperator":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be (D, N)")
        D, N = X.shape
        if D < N:
            raise ValueError(f"need D >= N for independent songs, got D={D} < N={N}")
        G = X.T @ X
        # pinv guards near-duplicate songs (the finite-precision shadow of the
        # existence condition); exact inverse when well conditioned.
        Ginv = np.linalg.pinv(G, rcond=rcond)
        return cls(X=X, Ginv=Ginv)

    # -- basic properties ---------------------------------------------------

    @property
    def D(self) -> int:
        return self.X.shape[0]

    @property
    def N(self) -> int:
        return self.X.shape[1]

    @property
    def Xnext(self) -> np.ndarray:
        """Columns advanced one playlist step, cyclically."""
        return np.roll(self.X, -1, axis=1)

    def gram_condition(self) -> float:
        G = self.X.T @ self.X
        return float(np.linalg.cond(G))

    # -- operator actions ---------------------------------------------------

    def coeffs(self, v: np.ndarray) -> np.ndarray:
        """Least-squares song coefficients of v: G^{-1} X^T v."""
        return self.Ginv @ (self.X.T @ v)

    def apply_A0(self, v: np.ndarray) -> np.ndarray:
        """A0 v = X_next G^{-1} X^T v."""
        return self.Xnext @ self.coeffs(v)

    def project_perp(self, M: np.ndarray) -> np.ndarray:
        """P_perp M = M - X G^{-1} X^T M (columns projected off the song span)."""
        return M - self.X @ (self.Ginv @ (self.X.T @ M))

    def apply(self, v: np.ndarray, U: np.ndarray | None = None,
              Vp: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
        """(A0 + scale * U Vp^T) v, where Vp is already P_perp-projected."""
        out = self.apply_A0(v)
        if U is not None and Vp is not None:
            out = out + scale * (U @ (Vp.T @ v))
        return out

    # -- verification ---------------------------------------------------------

    def playback_error(self, U: np.ndarray | None = None,
                       Vp: np.ndarray | None = None, scale: float = 1.0) -> float:
        """Max relative error of A x_n vs x_{n+1} over the cyclic playlist.

        Must be ~machine epsilon for any Z: the free part sees only P_perp x_n = 0.
        """
        Xn = self.Xnext
        errs = []
        for n in range(self.N):
            pred = self.apply(self.X[:, n], U=U, Vp=Vp, scale=scale)
            errs.append(np.linalg.norm(pred - Xn[:, n]) / max(np.linalg.norm(Xn[:, n]), 1e-30))
        return float(max(errs))

    # -- rendering factors ----------------------------------------------------

    def factors(self, U: np.ndarray | None = None, Vp: np.ndarray | None = None,
                scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (L, R) with A = L R^T, rank N (+ rank(Z) if given).

        L = [X_next G^{-1} | scale * U],  R = [X | P_perp V].
        """
        L = self.Xnext @ self.Ginv
        R = self.X
        if U is not None and Vp is not None:
            L = np.concatenate([L, scale * U], axis=1)
            R = np.concatenate([R, Vp], axis=1)
        return L, R
