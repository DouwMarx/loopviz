"""Audio ingest: youtube links -> wav files -> song matrix X (D x N).

Downloads bestaudio via yt-dlp, converts to mono wav at the configured
sample rate via ffmpeg, takes a centered excerpt, and L2-normalizes each
song so the Gram matrix is well scaled.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import AudioConfig


def download(links: list[str], out_dir: Path, sample_rate: int) -> list[Path]:
    """Download each link in order to numbered wav files. Returns paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, link in enumerate(links):
        stem = f"{i:03d}"
        target = out_dir / f"{stem}.wav"
        if target.exists():
            print(f"[{stem}] already downloaded, skipping")
            paths.append(target)
            continue
        print(f"[{stem}] downloading {link}")
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--js-runtimes", "node",
            "--no-playlist",
            "-f", "bestaudio/best",
            "-x", "--audio-format", "wav",
            "--postprocessor-args",
            f"ExtractAudio:-ac 1 -ar {sample_rate}",
            "-o", str(out_dir / f"{stem}.%(ext)s"),
            "--print-to-file", "%(title)s", str(out_dir / f"{stem}.title.txt"),
            link,
        ]
        for attempt in range(3):
            result = subprocess.run(cmd)
            if result.returncode == 0 and target.exists():
                break
            print(f"[{stem}] attempt {attempt + 1} failed, retrying...")
        else:
            raise RuntimeError(f"download failed for {link}")
        paths.append(target)
    return paths


def load_excerpt(path: Path, cfg: AudioConfig) -> np.ndarray:
    """Load a wav, mixdown to mono, resample, then make length uniform.

    Length policy: with cfg.stretch, the song is time-stretched (resampled)
    to exactly D samples; otherwise it is center-cropped if longer than D
    and zero-padded at the end if shorter.
    """
    data, sr = sf.read(str(path), dtype="float64")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != cfg.sample_rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, cfg.sample_rate)
        data = resample_poly(data, cfg.sample_rate // g, sr // g)
    D = cfg.dim
    if cfg.stretch and data.size != D:
        from scipy.signal import resample
        x = resample(data, D)
    elif data.size >= D:
        start = (data.size - D) // 2
        x = data[start:start + D]
    else:
        x = np.zeros(D)
        x[: data.size] = data
    if cfg.normalize:
        n = np.linalg.norm(x)
        if n > 0:
            x = x / n
    return x


def song_durations(wav_paths: list[Path]) -> list[float]:
    """Duration in seconds of each wav."""
    return [sf.info(str(p)).duration for p in wav_paths]


def window_seconds(policy: str, durations: list[float]) -> tuple[float, bool]:
    """Map a length policy to (window seconds, stretch flag).

    crop-shortest: window = shortest song, longer songs center-cropped
    pad-longest  : window = longest song, shorter songs zero-padded
    stretch      : window = longest song, every song time-stretched to fit
    """
    if policy == "crop-shortest":
        return min(durations), False
    if policy == "pad-longest":
        return max(durations), False
    if policy == "stretch":
        return max(durations), True
    raise ValueError(f"unknown length policy {policy!r}")


def build_song_matrix(wav_paths: list[Path], cfg: AudioConfig) -> np.ndarray:
    """Stack excerpts into X (D, N) in playlist order."""
    cols = [load_excerpt(p, cfg) for p in wav_paths]
    return np.stack(cols, axis=1)


def save_matrix(X: np.ndarray, titles: list[str], path: Path, cfg: AudioConfig,
                length_policy: str = "crop") -> None:
    np.savez_compressed(
        path, X=X.astype(np.float32),
        meta=json.dumps({
            "titles": titles,
            "sample_rate": cfg.sample_rate,
            "excerpt_seconds": cfg.excerpt_seconds,
            "length_policy": length_policy,
        }),
    )


def load_matrix(path: Path) -> tuple[np.ndarray, dict]:
    z = np.load(path, allow_pickle=False)
    return z["X"].astype(np.float64), json.loads(str(z["meta"]))


def read_titles(audio_dir: Path, wav_paths: list[Path]) -> list[str]:
    titles = []
    for p in wav_paths:
        t = Path(str(p).replace(".wav", ".title.txt"))
        titles.append(t.read_text().strip() if t.exists() else p.stem)
    return titles
