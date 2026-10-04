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
loopviz download --csv tracks.csv            # audio, 16 kHz
scripts/experiments/exp_paper_sweep.py       # exact candidates per paper size
loopviz compare                              # pairwise choices in the browser
loopviz fit                                  # preference model
scripts/make_prints.py                       # print-exact PDFs, 100% scale only
loopviz relief sweep|build|demo              # legacy: stepped relief, one bed
loopviz plate plan|build|demo                # one big smooth relief, strips printed on edge
loopviz plate export                         # package: audio, spec sheet, STLs, renders
```

A loop is named once in `loops/<name>.json` (audio file, start, end, title,
notes) and passed as `--loop loops/armed_man.json` to every `plate` and
`relief` command in place of `--audio --start --end`.

## Layout

```
src/loopviz/           core: operator.py (A from windows), songmatrix.py (one song as n x n),
                       ingest.py, config.py, loopspec.py (loops/<name>.json), cli.py
src/loopviz/paper/     paper prints: render, metrics, loss, optimize, zspace, palette,
                       compare_server, bt (Bradley-Terry), pool, sheet, matviz
src/loopviz/print3d/   3D prints: surface, slab, strippack, plate, export (the strip plate);
                       relief, relief_cli, reliefviz (legacy stepped design)
scripts/               make_prints, slice_metrics, pitch_sheet, render_stl; experiments/exp_*.py
doc/                   math.pdf (derivation), plate.md (strip plate), relief.md (legacy)
archive/               relief_testtile (calibration tiles of the stepped design), early specs
```

## 3D print

Two designs, both watertight STL (stereolithography mesh) files built
directly from the matrix, no CAD kernel. `loopviz relief` is the first
one: a stepped height field on one bed (up to 305 mm), one column per
entry, levels set by the layer height; `doc/relief.md` has sizing, printer
limits and what the quantized object still plays, `archive/relief_testtile/`
the calibration tiles. `loopviz plate` is bigger than the bed: the matrix
interpolated to a smooth surface, 660 mm on a 340 mm bed, cut into 2 x 7
strips that stand on edge and print side by side in one job, the relief
traced in XY and its amplitude set by the overhang limit and the strip
thickness. `plan` enumerates strip layouts, `build` writes a print-ready
`bed.stl` with engraved labels, assembly notes and metrics, `demo` makes
tiny test plates, `export` packages a finished build (the loop and its
playback by the exact and the as-printed operator as wavs, a spec sheet
with every parameter and the commit, STLs, previews, renders, README).
The shipped builds use Fourier interpolation and no clipping
(`--interp fourier --mapping none`), at rho 0.6 and 0.8. See
`doc/plate.md` for the numbers and the design decisions.

```
loopviz relief build --loop loops/armed_man.json --pitch 1.5
loopviz plate plan
loopviz plate build --loop loops/armed_man.json --pitch 1.5 --rho 0.6 --side 660 --cols 2 --rows 7 \
    --thickness 13 --relief auto --interp fourier --mapping none --subdiv 4 \
    --out runs/plate/armed_man_p1.5_rho0.6_fourier_t13
loopviz plate export --build runs/plate/armed_man_p1.5_rho0.6_fourier_t13 --out export/armed_man_p1.5_rho0.6 --zip
```

## Choosing by comparison

A song has many exact operators (sample rate, density, ink clip and
window count change texture, not playback); which to print is taste, so
taste is measured: pairwise clicks in the compare UI fit a Bradley-Terry
utility on 12 image metrics that ranks the pool per (song, paper).

## Setup

```
uv venv && uv pip install -e ".[dev]" && uv run pytest
uv run pytest -m visual
```

The visual tests render tiny plates and have headless Claude judge the
pictures (orientation, labels, smoothness, overhang colouring); they are
skipped by default and need the `claude` CLI.

Requires `ffmpeg` (and `node` for YouTube downloads).
