# playlistviz

A playlist you can hang on a wall.

Given N songs in a cyclic order, there is a matrix **A** that *is* the
playlist: `A x_n = x_{n+1}`, `A x_N = x_1`. One matrix-multiply advances the
music; iterating A plays it forever. But the exact solutions form a huge
family

```
A = A0 + Z P_perp
```

where `A0 = X_next G^-1 X^T` is pinned down by the music and `Z P_perp` is a
`D(D-r)`-dimensional free part that the playlist literally cannot hear
(`P_perp` annihilates every song). This project spends that freedom on
*beauty*: the free part and the rendering are optimized so that the picture
of A — the thing you print — scores well on quantitative proxies for human
aesthetic preference, with the proxy weights fitted **to you** via pairwise
comparisons (Bradley–Terry). Everything is grayscale: the audio produces one
scalar field per pixel, and honesty demands we print exactly that.

## Pipeline

```
playlistviz download --links-file links.txt   # direct youtube URLs (or ytsearch1: queries)
playlistviz build                             # 1 kHz, pad-longest by default
playlistviz optimize --replicates 2 --jobs 4  # Dirichlet weight sweep x ES-seed replicates
playlistviz compare                           # web UI: click the render you prefer
playlistviz fit                               # Bradley-Terry fit of your metric weights
playlistviz optimize --weights runs/fitted_weights.json --replicates 3 --jobs 3
playlistviz render <candidate> --resolution 4096 --bits 16  # print export
playlistviz report                            # all candidates on the equal-weight yardstick
```

Key knobs:
- `build --length-policy {crop-shortest, pad-longest, stretch}` — how songs
  of different lengths become one matrix (zero-padding is provably harmless
  to the operator; stretching changes the audio itself).
- `build --sample-rate` — playback fidelity only; the operator is exact at
  any rate. Default 1000 Hz while iterating.
- `optimize --replicates N` — N ES seeds per weight draw, separating weight
  effects from basin luck (the loss landscape is multimodal; replicate
  spread is routinely 1.5-3x in loss).
- `optimize --z-rank Q` — capacity of the free part (texture richness).
  This does NOT grow the search space: all Q columns share one theta.
- `optimize --scalarization chebyshev` — augmented weighted-Chebyshev
  scalarization; reaches non-convex parts of the Pareto front that weighted
  sums cannot.

## How A is rendered without materializing it

At audio scale A is dense `D x D`; it is never formed. A stays factored as
`A = L R^T` with rank `N + Q`, and pixel statistics stream through the
factors:

- **block mean** `M_pq = (pool L)_p · (pool R)_q` — signed, low-frequency
- **block energy** `E_pq = tr(G_p H_q)` via per-block cross-Grams — where
  the loudness structure of the songs shows up

Cost scales with output pixels and rank², never `D²`. Inside the ES loop
renders subsample rows (`opt_stride`); presentation and print renders are
exact. Tone mapping (log compression, gamma, free nonmonotonic tone curve,
vignette) turns the two fields into the printed grayscale image.

## The pixel-exact song operator (current focus)

The second construction drops the playlist, the free part, and ALL
rendering: one song is cut into N windows of n samples (columns of W) and
the cyclic window-advance operator `A0 = W S G^-1 W^T` is an n x n matrix
displayed 1 entry = 1 pixel. What you print is exactly the matrix that
plays the song. One equation ties every design knob together:

```
samples  L = f T = N n = rho n^2      =>      n = sqrt(f T / rho)
```

- `n` — canvas side: paper side / pixel pitch (~0.5 mm/px is comfortably
  discernible at 50 cm; `songmatrix.DISCERNIBLE_PITCH_MM`)
- `f` — implied sample rate; `T` — song duration
- `rho = N/n` — rank fraction = information density (fraction of pixels
  carrying independent audio). Full rank (`rho = 1`, one sample per pixel)
  sits exactly on the existence boundary: adjacent windows of real audio
  correlate, the Gram degenerates, playback breaks. Measured practical
  limit: `rho ~ 0.95` (exp_operator_sizing).

Digital silence is a hard wall — a linear map cannot send the zero vector
to the music that follows it — so `songmatrix.build` adds -70 dB
deterministic dither (inaudible; playback stays exact to ~1e-8).

Capacity at 8 kHz: a 2-minute song is a ~1000 px (50 cm) print; the full
27-minute playlist would need ~3600 px (1.8 m). It is a one-song-per-print
construction. Display modes for the materialized matrix (grayscale,
diverging palettes, Hinton, bubble, wireframe, 3D bars) live in
`matviz.py` / `exp_matrix_viz.py`.

An example, straight from the print queue — "The World Breathes with Me"
(Caligula's Horse) as the exact 1638 x 1638 operator that plays it
(f = 4130 Hz, rho = 0.32, clip 99.5, an A2 print at 0.25 mm/entry). The
horizontal weave is the song's pitch content interfering with the window
length; every pixel is one matrix entry:

![The World Breathes with Me as its own playback operator](doc/example_the-world-breathes-with-me.png)

## Choosing the print palette

The grayscale render encodes the signed matrix as one luminance axis
(mid-gray = 0), so recoloring the render is exactly recoloring the matrix.
`playlistviz palette [candidate]` launches an interactive viewer to pick a
diverging scheme and see it on the actual render. With no argument it opens
your highest-rated candidate (by fitted preference weights if you have run
`fit`, else the equal-weight yardstick):

- default schemes: ~30 presets in grouped menus - 16 OKLab-designed and
  tuned to print faithfully (dark-field, paper-field, muted), plus sourced
  stock maps for comparison: ColorBrewer colour-blind-safe diverging
  (RdBu/BrBG/PuOr/PRGn/PiYG/RdYlBu, colorbrewer2.org), Crameri
  perceptually-uniform maps bundled in matplotlib (berlin/managua/vanimo,
  fabiocrameri.ch), and classics (coolwarm/Spectral/...). Each carries a
  live print verdict, so the honestly-printable ones stand out from the
  saturated screen maps.
- design your own: six OKLab knobs - `hue_neg`/`hue_pos` (the two accent
  hues), `L_center`/`L_end` (field-vs-accent lightness; the center owns ~95%
  of the print), `C_end` (accent chroma), `drift` (per-arm hue rotation,
  berlin-like richness).

Every palette is checked for printability, not assumed: the sRGB gamut
(hard) and, when a CMYK ICC profile is found (`PLAYLISTVIZ_CMYK_ICC` or
auto-discovered), a coated-offset soft-proof (FOGRA39) reporting the fraction
of the ramp outside the print gamut. The designed presets are tuned to the
coated medium (dark field at the OKLab L~0.20 black floor, chroma inside
CMYK); saturated screen maps like coolwarm correctly flag as out of gamut.
Saving writes a reproducible config (spec + source hash + gamut + git commit)
and a preview; `playlistviz palette <candidate> --apply <config>.json`
re-renders it exactly. The engine is `palette.py`; the viewer is
`palette_server.py`.

An accepted playback error was investigated as a third design axis and
REJECTED. The SVD of A0 prices any component attenuation in closed form
(`err_k = sqrt(sum_i (a_i s_i v_i.w_k)^2)`), so a tolerance is a budget —
but (a) loop drift is ~850x the static tolerance (truncation moves
eigenvalues off the unit circle; a statically-fine 1e-2 approximation
explodes within one pass — exp_error_budget), and (b) an ES that was free
to spend the budget chose not to: within faithful-loop budgets,
attenuation is cosmetically invisible. So candidates are EXACT only, and
"loops forever" is structural (eigenvalues on the unit circle by
construction). Feasibility uses a plain stability margin — measured drift
per full pass <= 1e-9, i.e. indistinguishable from exact under iteration
— which is how the conditioning boundary is located per (song, rate);
it is a safety margin, not a fidelity claim. The honest aesthetic
parameters are rho, the ink
map, and the song — enumerated as a grid of comparison candidates by
exp_songop_pool for pairwise BT preference collection (`playlistviz
compare`). Crops are banned in all display modes: a crop cannot
reproduce the song, so it is not the artwork. Perturbations `Y P_perp`
with range(Y) inside window space would be exact AND provably loop-stable
(doc/math.pdf section 5) but are deliberately not used — simplicity.

## The math, didactically

`doc/math.pdf` derives everything in ~3 pages: why A_0 = X_next G^{-1} X^T
solves A X = X_next (the Gram matrix G unmixes song correlations), why the
full solution family is A_0 + Z P_perp, how the block-mean/block-energy
pictures stream through the factors.

## The free part Z (what the Z-space optimizer shapes)

`Z = U V^T`, rank Q (default 48), always applied through `P_perp` so it is
inaudible. Each column is

```
window(t) * noise(t)
```

- `noise`: 1/f^beta spectral noise, per-column beta spread (texture mix)
- `window`: Gaussian bumps at cached random positions; the `locality` theta
  interpolates global support (full-width streaks) to compact blobs. This is
  the anti-"line-ey" lever: localized columns contribute local patches.

All theta-independent randomness is cached (`ZGenerator`); one theta
evaluation costs a batched irfft plus elementwise work.

## Aesthetic metrics (12, grayscale)

Seven structural metrics from the aesthetic-optimization spec (spectral
slope β→2.0, fractal D→1.4, luminance entropy→5 bits, edge density→0.08,
gradient Gini→0.75, mirror symmetry→0.3, RMS contrast→0.2) plus four added
after a literature review and one construction-specific one:

| metric | source | target |
|---|---|---|
| edge-orientation entropy | Redies/Brachmann/Wagemans 2017 | 0.95 (normalized) |
| compression complexity | Forsythe et al. 2011 — zlib ratio | 0.50 |
| luminance skewness | Graham & Redies 2010 | 0.0 |
| balance (DCM) | Hübner & Fillinger 2016 | 0.05 |
| moiré salience | log peak of the annular-median-whitened 2D spectrum: strength of the pitch-interference weave/diagonals the operator produces (noise floor ~1.3, strong weave 4–6) | 1.5 |

Weight-independent barriers (contrast floor, entropy floor) guard against
reward hacking. Measurement is two-scale (image + 2x downsample). Targets
are population means; the whole point of the BT loop is to correct them for
one observer (fitted weights can go negative, which flags a wrong target or
direction — reported, never silently clipped).

## Personalization (Bradley–Terry)

Candidates store phi (metric values); loss vectors are rebuilt from phi by
name, so pools survive metric-set changes. The compare UI presents pairs
(D-optimal active selection once ≥5 choices exist); the fit is logistic
regression on loss-vector differences, L2-regularized toward uniform:
`P(i ≻ j) = σ(w · (ℓ_j − ℓ_i))`.

## Guarantees

- Aesthetic optimization **cannot** change playback: Z is applied through
  `P_perp`; `playback_error()` is asserted after every run (~1e-14).
- Renders are deterministic given theta; the print render is the comparison
  render, just sharper (and computed without stride approximation).
- Sign-flip sanity check (`exp_free_aesthetics.py --negate`): maximizing the
  loss produces structureless static, confirming the metric set is not
  vacuous.

## Layout

```
src/playlistviz/
  config.py     knobs (audio, Z rank, ES budget, resolutions, stride)
  ingest.py     yt-dlp + ffmpeg -> song matrix X, length policies
  operator.py   factored A0, P_perp, playback verification
  zspace.py     theta -> low-rank Z (localized spectral columns), cached
  render.py     factored block-mean / block-energy images, tone curve, 16-bit gray PNG
  metrics.py    the 12-metric feature map, two-scale
  loss.py       weighted / Chebyshev loss, barriers, Dirichlet sampling
  optimize.py   (1+lambda)-ES, optional low-rank search subspace
  bt.py         Bradley-Terry fit + D-optimal pair selection
  compare_server.py  stdlib web UI for pairwise choices
  cli.py        the pipeline commands
  sheet.py      labeled contact sheets for experiment outputs
  songmatrix.py one song as an n x n operator: sizing math, dither, build,
                SVD error pricing, loop-degradation measurement
  matviz.py     full-matrix display modes (gray, diverging, hinton, bubble,
                wireframe) - every entry shown, crops banned
  palette.py    OKLab diverging-palette engine + printability (sRGB gamut,
                CMYK soft-proof), presets, reproducible config IO
  palette_server.py  interactive palette designer web UI (playlistviz palette)
doc/
  math.tex/.pdf didactic derivation of the whole construction
scripts/        each writes a README.md into its runs/exp_* output folder
  exp_free_aesthetics.py  no-music-constraint baseline; --negate sanity check
  exp_length_policy.py    A0's own pictures under crop/pad/stretch (no ES)
  exp_song_matrix.py      one song as an n x n matrix, 1 entry = 1 pixel
  exp_operator_sizing.py  rank fraction vs conditioning; capacity tables
  exp_matrix_viz.py       display modes for the pixel-exact operator
  exp_error_budget.py     what a playback tolerance buys; loop stability
  exp_songop_pool.py      exact-only (rho x ink) grid -> BT candidates
tests/          88 tests, no network needed (synthetic songs)
```

## Setup

```
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

Requires `ffmpeg` on PATH (and `node` or `deno` for YouTube extraction).
