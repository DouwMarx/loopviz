"""A loop is an excerpt of one audio file with a name: loops/<name>.json.

    {"audio": "data/audio_3d/song.wav", "start_s": 324.68, "end_s": 336.705,
     "name": "armed_man", "title": "...", "notes": "..."}

`loopviz plate build --loop loops/armed_man.json` and every `loopviz
relief` command take the spec instead of --audio/--start/--end. Paths in
the spec are relative to the working directory, like every other CLI path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class LoopSpec:
    audio: Path
    start_s: float
    end_s: float
    name: str
    title: str = ""
    notes: str = ""

    def __post_init__(self):
        object.__setattr__(self, "audio", Path(self.audio))
        if not self.end_s > self.start_s:
            raise ValueError(f"loop end {self.end_s} must be after start {self.start_s}")

    @property
    def T(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["audio"] = str(self.audio)
        return d

    @classmethod
    def load(cls, path: Path | str) -> LoopSpec:
        d = json.loads(Path(path).read_text())
        return cls(**d)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1) + "\n")
        return path

    def signal(self, out_wav: Path | None = None) -> tuple[np.ndarray, float, int]:
        """(mono float64 signal, T, sr) of the excerpt."""
        return load_loop(self.audio, self.start_s, self.end_s, out_wav=out_wav)


def load_loop(audio: Path, start: float | None, end: float | None,
              out_wav: Path | None = None) -> tuple[np.ndarray, float, int]:
    """The excerpt [start, end) seconds of a wav as (signal, T, sr)."""
    x, sr = sf.read(str(audio), dtype="float64")
    if x.ndim == 2:
        x = x.mean(axis=1)
    a = 0 if start is None else round(start * sr)
    b = x.size if end is None else round(end * sr)
    if not 0 <= a < b <= x.size:
        raise SystemExit(f"loop [{start}, {end}) outside audio of {x.size / sr:.2f} s")
    seg = x[a:b]
    if out_wav is not None:
        sf.write(str(out_wav), seg, sr)
    return seg, seg.size / sr, sr


# -- CLI ----------------------------------------------------------------------------

def add_loop_args(q, required: bool = True) -> None:
    """--loop SPEC.json, or --audio WAV [--start S] [--end S]; the two are exclusive."""
    g = q.add_mutually_exclusive_group(required=required)
    g.add_argument("--loop", help="loop spec JSON (loops/<name>.json): audio, start, end, name")
    g.add_argument("--audio", help="wav file; with --start/--end the excerpt")
    q.add_argument("--start", type=float, default=None, help="loop start (s), with --audio")
    q.add_argument("--end", type=float, default=None, help="loop end (s), with --audio")


def resolve_loop(args) -> LoopSpec:
    """The LoopSpec named by the CLI flags. --audio without --start/--end
    is the whole file; its name is the file stem."""
    if args.loop:
        if args.start is not None or args.end is not None:
            raise SystemExit("--loop already fixes the excerpt; drop --start/--end")
        return LoopSpec.load(args.loop)
    audio = Path(args.audio)
    start = 0.0 if args.start is None else float(args.start)
    end = sf.info(str(audio)).duration if args.end is None else float(args.end)
    return LoopSpec(audio, start, end, name=audio.stem)

