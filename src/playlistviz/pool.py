"""Candidate pool on disk.

Candidates live one directory each (candidate.json + presentation.png),
grouped per song and per paper size:

    runs/songs/<song-slug>_t<stem>/<paper>/<candidate-id>/

<paper> is A4/A3/A2 for paper-sweep candidates and "pre" for the earlier
pre-paper pool (sized by n, not by a sheet). The legacy flat layout
runs/<candidate-id>/ is still read (playlist-level optimizer candidates
have no song and stay flat), so both discovery and id -> directory
resolution go through here.
"""

from __future__ import annotations

import re
from pathlib import Path


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:n].lower() or "untitled"


def song_dir(runs_dir: Path, title: str, song: str, paper: str = "pre") -> Path:
    """Folder for one song's candidates at one paper size,
    e.g. runs/songs/piano-man_t033/A2."""
    stem = Path(song).stem if song else "unknown"
    return (Path(runs_dir) / "songs" / f"{slug(title or stem)}_t{stem}"
            / (paper or "pre"))


def iter_candidate_files(runs_dir: Path) -> list[Path]:
    """Every candidate.json in the pool, both layouts, sorted by id."""
    runs_dir = Path(runs_dir)
    files = [f for f in runs_dir.glob("*/candidate.json")
             if not f.parent.name.startswith("exp_")]
    files += runs_dir.glob("songs/*/*/*/candidate.json")
    return sorted(files, key=lambda f: f.parent.name)


def candidate_dir(runs_dir: Path, cand_id: str) -> Path | None:
    """Resolve a candidate id to its directory, or None."""
    runs_dir = Path(runs_dir)
    flat = runs_dir / cand_id
    if (flat / "candidate.json").exists():
        return flat
    hits = [d for d in runs_dir.glob(f"songs/*/*/{cand_id}")
            if (d / "candidate.json").exists()]
    return hits[0] if hits else None
