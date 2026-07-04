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
comparisons (Bradley–Terry).

## Pipeline

```
playlistviz download --links-file links.txt   # direct youtube URLs (or ytsearch1: queries)
playlistviz build [--sample-rate 8000 --seconds 12]   # X (D x N), factored A0, verification
playlistviz optimize --replicates 2 --jobs 7  # Dirichlet weight sweep x ES-seed replicates
playlistviz compare                           # web UI: click the render you prefer
playlistviz fit                               # Bradley-Terry fit of your metric weights
playlistviz optimize --weights runs/fitted_weights.json   # re-optimize under YOUR weights
playlistviz render <candidate> --resolution 4096 --bits 16  # print export
playlistviz report                            # all candidates on the equal-weight yardstick
```

Optimization is **grayscale by default** (color metrics excluded from the
loss, palette degenerates to a free tone curve); pass `--color` to optimize
in color. `--replicates N` reruns each weight draw under N different ES
seeds so a BT preference can be attributed to the weights rather than one
lucky basin. `--z-rank` trades texture richness against compute.

Quick experiments live in `scripts/`:
- `exp_padding.py` — zero-padding vs center-cropping for unequal song lengths
- `exp_free_aesthetics.py` — pure aesthetic optimization with no music
  constraint (the reference for what the constraint costs visually)

## How A is rendered without materializing it

At audio scale A is dense `D x D` (tens of GB); it is never formed. A stays
factored as `A = L R^T` with rank `N + q`, and pixel statistics stream
through the factors:

- **block mean** `M_pq = (pool L)_p · (pool R)_q` — signed, low-frequency
- **block energy** `E_pq = tr(G_p H_q)` via per-block `(N+q) x (N+q)`
  cross-Grams — where the loudness structure of the songs shows up

Cost scales with output pixels and rank², never `D²`, so 4096² print renders
are cheap. Tone mapping (log-compression, gamma, vignette) and a cosine
palette turn the two fields into RGB; those parameters sit in the same theta
the optimizer searches.

## Aesthetic metrics (14)

Ten from the original aesthetic-optimization spec (spectral slope β→2.0,
fractal D→1.4, Hasler–Süsstrunk colorfulness, hue dispersion, luminance
entropy, edge density, gradient Gini, mirror symmetry, RMS contrast, mean
saturation) plus four added after a literature review, chosen for strongest
evidence and least overlap with the existing set:

| metric | source | target |
|---|---|---|
| edge-orientation entropy | Redies/Brachmann/Wagemans 2017 — artworks sit near max EOE | 0.95 (normalized) |
| compression complexity | Forsythe et al. 2011 — zlib ratio, Berlyne inverted-U | 0.50 |
| luminance skewness | Graham & Redies 2010 — art has near-zero skew | 0.0 |
| balance (DCM) | Hübner & Fillinger 2016 — r ≈ −0.84 with liking | 0.05 |

Weight-independent barriers (contrast floor, entropy floor, colorfulness
ceiling) guard against reward hacking. Measurement is two-scale (image +
2× downsample) to penalize scale-fragile solutions.

Metrics considered and deferred: PHOG self-similarity and anisotropy
(largely redundant with EOE + fractal D + β), Ou–Luo color harmony (needs
constants from the paper, validated only on color pairs), corner density /
curvature (mostly captured by 2nd-order EOE), Birkhoff/Rigau order-complexity
(weak direct preference evidence).

## Personalization (Bradley–Terry)

Every optimization run stores its winner's per-metric loss vector. The
compare UI presents pairs of renders (D-optimal active selection once ≥5
choices exist); choices append to `runs/comparisons.jsonl`. The fit is
logistic regression on loss-vector differences, L2-regularized toward
uniform: `P(i ≻ j) = σ(w · (ℓ_j − ℓ_i))`. Negative fitted components are
flagged — they mean you *like* what the population target penalizes.
A few dozen comparisons suffice for 14 weights.

## Guarantees

- The aesthetic optimization **cannot** change playback: Z is applied through
  `P_perp`, and `playback_error()` is asserted after every run (~1e-12).
- Renders are deterministic given theta (noise seeds fixed per role), so the
  print render is the comparison render, just sharper.

## Layout

```
src/playlistviz/
  config.py     knobs (sample rate, excerpt length, Z rank, ES budget, resolutions)
  ingest.py     yt-dlp + ffmpeg -> song matrix X
  operator.py   factored A0, P_perp, playback verification
  zspace.py     theta -> low-rank Z (spectral noise, envelope-modulated)
  render.py     factored block-mean / block-energy images, palette, 16-bit PNG
  metrics.py    the 14-metric feature map, two-scale
  loss.py       weighted loss, barriers, Dirichlet weight sampling
  optimize.py   (1+lambda)-ES, optional low-rank search subspace
  bt.py         Bradley-Terry fit + D-optimal pair selection
  compare_server.py  stdlib web UI for pairwise choices
  cli.py        the pipeline commands
scripts/        quick one-off experiments (padding, unconstrained baseline)
tests/          60 tests, no network needed (synthetic songs)
```

## Setup

```
uv venv && uv pip install -e ".[dev]"
uv run pytest
```

Requires `ffmpeg` on PATH (and `node` or `deno` for YouTube extraction).
