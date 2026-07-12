# loopviz

![a song as the matrix that plays it](doc/example_the-world-breathes-with-me.png)

The matrix that plays ([The World Breathes with Me](https://www.youtube.com/watch?v=Dc2CtNFOqV4)) on a loop.

Cut a song into N consecutive windows `w_1 ... w_N` of n samples. There is
an n x n matrix A that advances the song one window at a time and closes
the loop:

```
A w_k = w_{k+1},        A w_N = w_1
```

Iterating A plays the song exactly, forever. A is the exact
[dynamic mode decomposition](https://en.wikipedia.org/wiki/Dynamic_mode_decomposition)
operator of the window sequence, with zero residual and no rank
truncation; the cyclic closure pins its nonzero eigenvalues to the unit
circle, so the loop never decays.

See `doc/math.pdf` for a derivation.

## Pipeline

```
loopviz download --csv tracks.csv     # audio, 16 kHz
scripts/exp_paper_sweep.py            # exact candidates per paper size
loopviz compare                       # pairwise choices in the browser
loopviz fit                           # preference model
scripts/make_prints.py                # print-exact PDFs, 100% scale only
```

## Choosing by comparison

A song has many exact operators (sample rate, density, ink clip and
window count change texture, not playback); which to print is taste, so
taste is measured: pairwise clicks in the compare UI fit a Bradley-Terry
utility on 12 image metrics that ranks the pool per (song, paper).

## Setup

```
uv venv && uv pip install -e ".[dev]" && uv run pytest
```

Requires `ffmpeg` (and `node` for YouTube downloads).
