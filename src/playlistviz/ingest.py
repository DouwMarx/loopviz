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
    """Load a wav, mixdown to mono, resample if needed, center-crop/pad to D."""
    data, sr = sf.read(str(path), dtype="float64")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != cfg.sample_rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, cfg.sample_rate)
        data = resample_poly(data, cfg.sample_rate // g, sr // g)
    D = cfg.dim
    if data.size >= D:
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


def build_song_matrix(wav_paths: list[Path], cfg: AudioConfig) -> np.ndarray:
    """Stack excerpts into X (D, N) in playlist order."""
    cols = [load_excerpt(p, cfg) for p in wav_paths]
    return np.stack(cols, axis=1)


def save_matrix(X: np.ndarray, titles: list[str], path: Path, cfg: AudioConfig) -> None:
    np.savez_compressed(
        path, X=X.astype(np.float32),
        meta=json.dumps({
            "titles": titles,
            "sample_rate": cfg.sample_rate,
            "excerpt_seconds": cfg.excerpt_seconds,
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
