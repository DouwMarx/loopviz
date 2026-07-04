"""Central configuration dataclasses. Everything tunable lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 8000       # Hz, mono
    excerpt_seconds: float = 12.0  # centered excerpt per song
    normalize: bool = True         # unit L2 norm per song

    @property
    def dim(self) -> int:
        return int(self.sample_rate * self.excerpt_seconds)


@dataclass(frozen=True)
class ZConfig:
    rank: int = 16          # rank of the free perturbation Z = U V^T
    seed: int = 7           # base seed; noise fields are deterministic given (seed, theta)


@dataclass(frozen=True)
class RenderConfig:
    opt_resolution: int = 384          # px, square, used inside the ES loop
    presentation_resolution: int = 1536  # px, used for candidates shown in comparisons
    print_resolution: int = 4096       # px, final print render


@dataclass(frozen=True)
class OptConfig:
    generations: int = 14
    population: int = 12       # (1+lambda)-ES
    sigma0: float = 1.0
    sigma_decay: float = 0.82
    subspace_rank: int = 8     # search in a random rank-k subspace of theta (0 = full)
    seed: int = 0


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=lambda: Path.cwd())

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def audio(self) -> Path:
        return self.data / "audio"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def comparisons(self) -> Path:
        return self.runs / "comparisons.jsonl"
