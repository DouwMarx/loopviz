# loopviz

![The World Breathes with Me, as the matrix that plays it](doc/example_the-world-breathes-with-me.png)

This is "The World Breathes with Me" (Caligula's Horse). Not a picture of
the song: the matrix that plays it.

Cut a song into N consecutive windows `w_1 ... w_N` of n samples. There is
an n x n matrix A that advances the song one window at a time and closes
the loop:

```
A w_k = w_{k+1},        A w_N = w_1
```

Iterating A plays the song exactly, forever: its eigenvalues sit on the
unit circle by construction, so the loop never decays. The print shows A
itself, one matrix entry per dot, never cropped, never resampled: the
artwork IS the operator. The visible weave is whatever redundancy in the
song A could not whiten. One equation sizes everything:

```
f T = rho n^2      (sample rate x duration = density x pixels)
```

paper, sample rate, song length and density trade off. `doc/math.pdf` derives everything.

## Pipeline

```
loopviz download --csv musiek_van_die_maand.csv    # audio, 16 kHz
scripts/exp_paper_sweep.py                         # candidates per paper size
loopviz compare                                    # click what you prefer
loopviz fit                                        # Bradley-Terry preference model
scripts/make_prints.py                             # print-exact PDFs, 100% scale only
```

Candidates carry 12 image metrics; your pairwise choices fit a
Bradley-Terry utility (quadratic + interactions) that ranks the pool per
(song, paper).

## Setup

```
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

Requires `ffmpeg` (and `node` for YouTube downloads).
