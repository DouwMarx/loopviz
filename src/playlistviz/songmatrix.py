"""One song (or a playlist) as a single n x n matrix, one entry per pixel.

The audio is cut into N consecutive windows of n samples (columns of W,
n x N) and the cyclic window-advance operator

    A0 w_k = w_{k+1},  A0 w_N = w_1,     A0 = W S G^{-1} W^T

is n x n and displayed exactly, one entry per pixel. Seeded with window 1,
iterating A0 plays the audio and repeats. No free part, no pooling.

Sizing math
-----------
The canvas side n, sample rate f, total duration T and rank fraction
rho = N/n are tied by one equation:

    samples  L = f T = N n = rho n^2      =>      n = sqrt(f T / rho)

so only three of (n, f, T, rho) are free. rho is the information density:
the fraction of the n^2 pixels that carry independent audio content. At
rho = 1 (N = n, "just barely full rank") every pixel is worth exactly one
sample - but real audio windows go nearly linearly dependent there
(adjacent windows correlate), the Gram becomes ill-conditioned and
playback breaks. plan() lets you pick any consistent combination and
exp_operator_sizing.py measures where the boundary actually is.

Print math: a matrix of side n printed at pixel pitch p mm occupies
n*p mm. Human 20/20 acuity resolves ~0.15 mm at 50 cm viewing distance;
comfortably *discernible* pixels want ~3x that, so p ~ 0.4-0.6 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import soundfile as sf

from .operator import PlaylistOperator

# Comfortable pixel pitch (mm) for individually discernible entries at
# ~50 cm viewing distance (about 3x the 20/20 acuity limit of ~0.15 mm).
DISCERNIBLE_PITCH_MM = 0.5


@dataclass(frozen=True)
class Plan:
    """A consistent (n, N, f, T) choice for a pixel-exact song operator."""

    n: int            # window length = matrix side = image side (pixels)
    N: int            # number of windows = rank of A0
    f: float          # implied sample rate (Hz)
    T: float          # audio duration covered (s)

    @property
    def rho(self) -> float:
        """Information density N/n = fraction of pixels carrying audio."""
        return self.N / self.n

    @property
    def samples(self) -> int:
        return self.N * self.n

    def print_side_mm(self, pitch_mm: float = DISCERNIBLE_PITCH_MM) -> float:
        return self.n * pitch_mm


def plan(T: float, *, n: int | None = None, f: float | None = None,
         rho: float = 1.0) -> Plan:
    """Solve f*T = rho*n^2 for the missing quantity.

    Give the duration T plus exactly one of n (canvas side) or f (sample
    rate); rho = N/n is the rank fraction (1.0 = full rank).
    """
    if (n is None) == (f is None):
        raise ValueError("give exactly one of n= or f=")
    if n is None:
        n = round(sqrt(f * T / rho))
    N = max(1, round(rho * n))
    return Plan(n=n, N=N, f=N * n / T, T=T)


def full_rank_side(T: float, f: float) -> int:
    """Smallest square canvas that holds T seconds at f Hz: n = ceil(sqrt(fT)).

    At this size the operator is just barely full rank and each pixel
    carries exactly one audio sample. Works for a playlist too - T is the
    total duration.
    """
    return int(np.ceil(sqrt(f * T)))


def load_audio(paths: list[Path]) -> tuple[np.ndarray, float]:
    """Concatenate wav files to one mono signal; return (signal, duration_s)."""
    parts, T = [], 0.0
    for p in paths:
        data, sr = sf.read(str(p), dtype="float64")
        if data.ndim == 2:
            data = data.mean(axis=1)
        parts.append(data)
        T += data.size / sr
    return np.concatenate(parts), T


def build(signal: np.ndarray, pl: Plan,
          dither_db: float = -70.0) -> tuple[PlaylistOperator, np.ndarray]:
    """Resample signal to pl.samples, window it, return (operator, W).

    dither_db: inaudible noise floor added before windowing (relative to
    the song's RMS, deterministic seed). Required whenever the audio has
    digitally silent stretches: a linear operator cannot map the zero
    vector to the music that follows it, so silent windows must be given
    a floor to advance from. -70 dB leaves playback exact to ~1e-8.
    """
    from scipy.signal import resample

    song = resample(signal, pl.samples)
    if dither_db is not None:
        rng = np.random.default_rng(7)
        song = song + (10 ** (dither_db / 20) * np.std(song)
                       * rng.standard_normal(song.size))
    W = song.reshape(pl.N, pl.n).T          # column k = window k
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    return PlaylistOperator.from_songs(W), W


def materialize(op: PlaylistOperator) -> np.ndarray:
    """The exact n x n matrix A0 (fine at pixel-print sizes, n <~ 4000)."""
    L, R = op.factors()
    return L @ R.T
