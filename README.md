# loopviz

![a song as the matrix that plays it](doc/example_the-world-breathes-with-me.png)

Not a picture of a song: the matrix that plays it ([The World Breathes with Me](https://www.youtube.com/watch?v=Dc2CtNFOqV4)).

Cut a song into N consecutive windows `w_1 ... w_N` of n samples. There is
an n x n matrix A that advances the song one window at a time and closes
the loop:

```
A w_k = w_{k+1},        A w_N = w_1
```

Iterating A plays the song exactly, forever: eigenvalues on the unit
circle by construction. The print shows A at one entry per dot, never
cropped, never resampled: the artwork is the operator. One equation
sizes everything:

```
f T = rho n^2      (sample rate x duration = density x pixels)
```

paper, sample rate, song length and density trade off. `doc/math.pdf` derives everything.

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
