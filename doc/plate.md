# The operator as one big smooth plate

`loopviz plate` prints the exact n x n window-advance operator A as a
single smooth relief, bigger than the printer bed, cut into strips that
stand on edge and print side by side in one job. `doc/relief.md` is the
stepped predecessor and keeps the sizing (`f = rho n^2 / T`,
`n = floor(side / pitch)`), the operator maths and the playback analysis.

## Why on edge

A 660 mm plate on a 340 mm bed needs cutting. Flat, the tiles each take a
print and the relief is built from layers: every height is a layer
boundary and every slope is a staircase. On edge the strip's back is a
vertical wall, its long bottom edge sits on the bed, and the relief is
traced by the XY motion of the nozzle inside every layer, so it is as
smooth as the toolpath. One job prints all strips. The price: the plate's
y axis becomes the printer's Z, so a slope in y is an overhang. Fused
deposition modelling (FDM) tolerates about 45 degrees from vertical, and
that caps the relief amplitude (see "Surface" below).

## Packing model

`src/loopviz/print3d/strippack.py`, the hand-over note's model with one added
constraint. With L the usable bed length, Hz the printer height, g the
gap between strips, k the largest safe slenderness h/t and t the strip
thickness, the plate side S of a cols x rows cut satisfies

    S <= cols * L
    S <= rows * min(Hz, k t)
    cols * rows <= slots(t) = floor((L + g) / (t + g))
    t >= relief_mm + body_min_mm

The last line is new: the hand-over omits the body under the relief. k
comes from the cantilever model in the hand-over (first natural frequency
f1 = 270 t / h^2 Hz for PLA, polylactic acid, t and h in metres; keep f1
above 100 Hz). Enumeration over t and all (cols, rows) solves it. Material
is a shell model (3 walls of 0.45 mm, relief band solid, core at 15 %
infill, PLA 1.24 g/cm^3) and time is volume / (10 mm^3/s x 0.6 duty),
labelled crude: slice for real numbers.

`uv run loopviz plate plan` with defaults (bed 340, margin 5 so L = 330,
zmax 325, gap 8, k 8, relief 4 + body 3 so t >= 7) prints the best S per
thickness, then this frontier:

```
Pareto frontier, S versus material (step: % material per % side)
    t layout            S    strip l x h   slots   h/t  f1 Hz seams     kg  kg/m2 crude h  binds   step
  7.5 2x10          600.0  300.0 x  60.0  20/21   8.00    562    10   2.17   6.02      81  kt
  7.0 2x11          616.0  308.0 x  56.0  22/22   8.00    603    11   2.24   5.90      84  kt     1.2
  8.0 2x10          640.0  320.0 x  64.0  20/21   8.00    527    10   2.51   6.14      94  kt     3.2
  9.0 2x9           648.0  324.0 x  72.0  18/19   8.00    469     9   2.67   6.37     100  kt     5.1
  8.5 2x10          660.0  330.0 x  66.0  20/20   7.76    527    10   2.73   6.26     102  len     1.0
 28.0 3x3           672.0  224.0 x 224.0   9/9    8.00    151     4   4.67  10.33     174  kt    39.2
 28.5 3x3           684.0  228.0 x 228.0   9/9    8.00    148     4   4.88  10.43     182  kt     2.6
 29.0 3x3           696.0  232.0 x 232.0   9/9    8.00    145     4   5.10  10.52     190  kt     2.6
 29.5 3x3           708.0  236.0 x 236.0   9/9    8.00    143     4   5.32  10.62     199  kt     2.6

recommended: t = 8.5 mm, 2x10 strips of 330.0 x 66.0 x 8.5 mm, S = 660 mm, 2.73 kg, ~102 h (crude)
```

The recommendation rule: from the cheapest frontier layout, jump to the
largest S costing at most 5 % material per % side, repeat. Material grows
at least as S^2, so 2 % per % is the floor; the jump from 660 to 672 mm
costs 39 % per %, which is where the returns stop. `--side 660` lists the
lightest thickness per grid at a fixed side: 2x10 at 8.5 mm is 2.73 kg,
2x3 at 27.5 mm is 4.42 kg, so fewer seams are bought with mass.

Three corrections to the hand-over, all in `tests/test_strippack.py`:

- t = 10 gives 2 x 9 strips of 330 x 73.3 mm, not 330 x 80. 80 mm is the
  limit k t; a 660 mm square only needs 660 / 9.
- The plateau at 2L = 660 from t = 10 upward is not unbroken: 3 x 3 strips
  beat it for t in 28 to 29.5 mm (672 to 708 mm).
- The t = 5 row (S = 520) is infeasible with any relief: a strip needs
  relief plus body, 7 mm at the defaults.

The hand-over calls two strips end to end per bed row an extension. On
this bed it beats the one-per-row optimum: `--per-row 2` gives 5 x 4
strips of 161 x 201 x 25.5 mm, S = 805 mm (20 strips in 10 bed rows,
6.43 kg) against 708 mm. `plate build` places one strip per bed row, so
this stays a plan-only option.

## Surface

`src/loopviz/print3d/surface.py`. Cell (i, j) sits at ((j + 0.5) p, (i + 0.5) p),
row 0 at the bottom. Values v = clip(A / m, -1, 1) with
m = percentile(|A|, 99.5), the clip the paper prints use; optional
Gaussian blur in cells, renormalised to peak 1. Fine-grid nodes at
k p / subdiv (subdiv 3: 0.667 mm at 2 mm pitch, 0.5 mm at 1.5 mm).
Interpolation is a plug-in with four methods; on a 24 x 24 patch around
the roughest spot of the n = 330 matrix (`runs/exp_smooth/RESULTS.md`,
Q3, slopes in range per pitch):

| method | overshoot | undershoot | max dh/dy | max dh/dx |
|---|---|---|---|---|
| nearest | 0 | 0 | 2.8 | 6 |
| linear | 0 | 0 | 0.47 | 1.0 |
| cubic | 0.14 | 0.12 | 0.61 | 1.34 |
| fourier | 0.28 | 0.50 | 0.73 | 1.81 |

Nearest is the old stepped look. Linear is a tent field with creases
along every cell row and column. Cubic (ndimage order 3, C2) is the
default: smooth ridges, no creases, 14 % overshoot next to spikes.
Fourier (mirror-padded zero-pad FFT, the only band-limited one) rings
with halos around every spike.

Overshoot means the printed range exceeds the nominal relief. The
surface measures the range after interpolation; with `max_range_mm`
(thickness minus body, 7 mm at t = 10 and body 3) it scales the relief
down so the printed range fits, never clips, and shifts the top to z = t.
Both armed-man builds hit this cap (scale 0.646 and 0.489).

Overhang on edge. A facet with gradient (hx, hy) has print-frame normal
(-hx, -1, -hy), so it hangs only when hy > 0 (the other sign leans back
like a roof) and its angle from vertical is

    tan(theta) = hy / sqrt(1 + hx^2)

A facet steep in x runs obliquely and hangs less. Interpolation is
linear in the heights, so one surface at 1 mm relief gives every facet
its critical relief, R^2 (hy^2 - tan^2(a) hx^2) = tan^2(a), infinite when
hy <= tan(a) |hx|. `fit_relief` returns the (1 - q) quantile over the
footprint, q = 0.99 by default (`--area-quantile`), and `--rule y` keeps
the simpler |dh/dy| <= tan(a) for comparison. The 45 degree limit is the
FDM rule of thumb: each layer steps out by layer x tan(alpha), so the
unsupported fraction of a 0.45 mm line is u = layer tan(alpha) / w; at
0.2 mm layers 45 degrees is u = 0.44, and the u = 0.5 point is 66, 56 and
48 degrees at 0.10, 0.15 and 0.20 mm layers (RESULTS.md Q5; sources:
https://3dx.info/beyond-basics-how-layer-height-and-line-width-impact-overhang-performance-in-3d-printing/,
https://www.snapmaker.com/blog/45-degree-rule-3d-printing/,
https://www.convertools.net/3d-printing/max-overhang-angle/).

## The matrix is anisotropic, and y is the smooth axis

RESULTS.md Q2 at rho 0.95 (r1 = lag-1 autocorrelation, hf = energy
fraction in the outer half of the band):

| pitch | n | r1 along y | r1 along x | hf along y | hf along x |
|---|---|---|---|---|---|
| 2.0 | 330 | 0.68 | -0.57 | 0.06 | 0.84 |
| 1.5 | 440 | 0.86 | -0.77 | 0.01 | 0.98 |

Why: A = W_next G^-1 W^T. Column j of A is a combination of the next
windows, so along i (row index, plate y) the matrix inherits the music's
spectrum resampled to f, and the music sits far below the grid Nyquist.
Row i of A is a combination of the dual basis G^-1 W^T = V S^-1 U^T,
which weights every singular pattern by 1 / s_r, so the weakest, near
Nyquist components of the windows dominate: along j (plate x) the entries
alternate. The same overhang rule applied to x allows 2.1 mm of relief
at 2 mm pitch where y allows 6.3 (Q4). On edge y is the printer's Z and
only the y slope hangs, so matrix row to plate y is the right
orientation (the default; `--transpose` flips it). The x chatter becomes
a 2-cell, 4 mm side-to-side ripple inside each layer, which XY motion
prints. The builds confirm it: at 2 mm pitch the 99th percentile of
|dh/dx| is 2.58 against 0.89 for |dh/dy|, and 24 % of the footprint
exceeds 45 degrees in x against 0.4 % in y (`metrics.json`, slopes).

## Cutting, labels, print orientation

Cuts snap to the nearest fine node, so neighbours share their seam nodes
exactly and strips may differ by one fine cell (73 and 73.5 mm at 1.5 mm
pitch). Labels: column letter plus row number, A1 bottom left, engraved
0.6 mm into the back near the strip's top-left corner, 6 mm cap height
from Pillow's bundled font, mirrored in x so they read when the back is
viewed from behind, with a filled triangle pointing to plate +y; the
build refuses a label whose ink would reach a boundary node.

Print orientation is the rotation (x, y, z) to (x, -z, y), determinant
+1, never a mirror. The relief faces the printer front (-Y), the back is
a vertical wall, strip k (label order A1 .. A9, B1 .. B9) is the k-th slot
from the front at Y in [y_k - t, y_k], y_k = margin + t + k (t + gap). The
18 strips of the armed-man builds end at Y = 321 mm of 340. `--orientation
flat` writes tiles for one job each instead.

## Verification chain

Every build runs, and fails loudly on:

- Closed 2-manifold per strip: every undirected edge in exactly two
  faces, every directed edge once, positive volume (`manifold_check`;
  `tests/test_slab.py` and `tests/test_plate.py` also assert
  `trimesh.is_watertight`, a dev dependency).
- Volume equal to the analytic integral of the triangulated height field
  to 1e-9 relative.
- Layout: inside the bed and under zmax, h/t <= k, no two bed boxes
  overlap.
- Reassembly in memory: the inverse bed transform returns the plate
  vertices to 1e-6 mm.
- Reassembly from disk (`verify_build`): every strips/<label>.stl read
  back, inverse-transformed, and matched to plate.stl by nearest
  neighbour both ways within 1e-3 mm (STL is float32). Measured 4.1e-5 mm
  at 2 mm pitch and 1.5e-5 mm at 1.5 mm.

Visual tests judged by headless Claude (`tests/visual_judge.py`,
`tests/test_visual.py`): 8 tests marked `visual`, deselected by default
so `uv run pytest` stays offline (174 pass in 29 s). Run them with

```
uv run pytest -m visual
```

Each test pairs a numeric assertion with a picture judgement of tiny
geometry from `loopviz plate demo`: a raised F from the front and after
reassembly from bed strips (not mirrored), six upright strips on the
bed, labels B2 and A3 read from behind with the triangle up, a spike as
a flat column (nearest) against a rounded bump (cubic), a ridge strip
with red only under the bulge. One harness test checks the judge can say
no (an L judged as a T must fail). The judge runs `claude -p` with the
Read tool, the nesting guard removed, model from `LOOPVIZ_VISUAL_MODEL`
(default sonnet), and logs prompt, answer and cost under
`runs/visual/<test>/`: 0.02 to 0.10 USD per call on sonnet.

## Outputs of a build

| file | what | 2 mm pitch | 1.5 mm pitch |
|---|---|---|---|
| plate.stl | the uncut plate, plate frame, flat back | 98.6 MB | 175.0 MB |
| bed.stl | every strip placed and oriented on the bed: print this | 198.2 MB | 351.4 MB |
| strips/<label>.stl | the same, one file per strip | 198.2 MB total | 351.4 MB total |
| layout.json | bed, per-strip bboxes, 4x4 transforms, overhang per strip | 29 KB | 28 KB |
| metrics.json | plate totals, slopes, overhang histograms, relief choice | 3 KB | 3 KB |
| assembly.md | label grid as hung, seams, print settings | 2 KB | 2 KB |
| preview_*.png | plate hillshade with seams; bed drawing plus oblique mesh; worst strip coloured by overhang | | |
| A.npy, surface_Z.npy | the matrix and the fine node heights | 0.9 + 7.9 MB | 1.5 + 14 MB |

`--write-cut` adds plate_cut.stl (all strips in the plate frame, labels
engraved, about the size of bed.stl) and `--plate-strips` the same per
strip; the visual tests use them. `--subdiv 2` quarters the face count.

## Export package

`loopviz plate export --build runs/plate/<name> --out export/<name> [--zip]`
(`src/loopviz/print3d/export.py`) assembles a self-contained directory from
a finished build: `audio/` (the loop at the source rate, three times, the
exact operator played at round(f) Hz from window 1 for three passes with
the window norms restored, and the same with the operator the plate
carries, clipped and quantised to 0.05 mm height steps, which is degraded
by design, see `doc/relief.md`), `spec_sheet.md` and `.json` (every
parameter, overhang, mass, slicer numbers if `slice_*.json` exists, drift
per pass, commit and version), `analysis/` (previews, metrics, layout,
assembly notes, the design study, renders), `stl/` (plate, bed, strips),
`renders/` and a README. The build records its loop spec
(`loops/<name>.json`, see `src/loopviz/loopspec.py`) in `metrics.json`;
older builds with only audio/start/end still export.

## Metrics

Overhang histogram (`slab.mesh_metrics`): per face with normal against
bed +Z, excluding faces lying on the bed, alpha = asin(-n . up) in
degrees (0 vertical wall, 90 ceiling), area-weighted into bins 0, 30, 45,
50, 60, 90. `overhang_bed` counts every face of every strip, including
engraving walls; `overhang_relief_face` only the relief. Mass is volume
times 1.24 g/cm^3, so solid; the shell estimate sits next to it. Slopes
are quantiles over fine cells of |dh/dx|, |dh/dy| and the facet angle.

`scripts/slice_metrics.py` slices an STL or a strips directory headlessly
with PrusaSlicer (presets generic04 and slow04: 0.2 mm layers, 3 walls,
15 % gyroid, 5 mm brim, rear seam, 0.2 mm elephant foot compensation)
and reports the slicer's manifold flag, time, filament and layer count
into slice_<preset>.json. Neither armed-man build has been sliced yet;
the hours below are the crude model and the hand-over's 35 to 50 h is
unverified.

## The armed-man builds

Loop 324.68 to 338.18 s of data/audio_3d/song.wav (T = 13.5 s), 2 x 9
strips of 330 x 73 x 10 mm, cubic, subdiv 3, clip 99.5, no blur, labels
on, relief auto under the facet rule (parameters from layout.json):

```
uv run loopviz plate build --audio data/audio_3d/song.wav --start 324.68 --end 338.18 \
    --pitch 2.0 --side 660 --cols 2 --rows 9 --thickness 10 --out runs/plate/armed_man_p2
uv run loopviz plate build --audio data/audio_3d/song.wav --start 324.68 --end 338.18 \
    --pitch 1.5 --side 660 --cols 2 --rows 9 --thickness 10 --out runs/plate/armed_man_p1.5
```

| | 2.0 mm pitch | 1.5 mm pitch |
|---|---|---|
| n, N, rho | 330, 314, 0.952 | 440, 418, 0.950 |
| f (Hz), drift per pass | 7676, 5.7e-10 | 13624, 3.6e-08 |
| nodes, faces | 991 x 991, 3.96 M | 1321 x 1321, 7.03 M |
| relief fit, facet rule, 99 % under 45 deg | 8.97 mm | 12.59 mm |
| relief chosen (scaled to the 7 mm cap) | 5.79 mm | 6.16 mm |
| printed range, body under the trough | 3.0 to 10.0 mm, 3.0 mm | 3.0 to 10.0 mm, 3.0 mm |
| relief face over 45 deg | 0.20 % (12 of 5829 cm^2), max 63 deg, strip B3 | 0.05 % (4 of 6921 cm^2), max 62 deg, strip B7 |
| mass, solid / shell estimate | 3.49 / 3.38 kg | 3.47 / 3.39 kg |
| crude time | 126 h | 126 h |
| plate.stl, bed.stl, strips/ | 98.6, 198.2, 198.2 MB | 175.0, 351.4, 351.4 MB |
| build time | 12 s | 19 s |

Two readings. The overhang rule is not what limits these plates: the
facet fit would allow 9 and 12.6 mm, the 10 mm strip allows 7 mm of
printed range, and the cubic overshoot (1.21x and 1.14x the nominal)
eats the rest. A thicker strip buys relief. And finer pitch is smoother
in y (Q2), so the 1.5 mm plate has less than a third of the over-45 area at
higher relief, at the price of 1.8x the file size.

## What is lost on edge

- No colour change by layer. Flat, a single M600 at mid relief painted
  the sign of every entry; on edge a layer is a horizontal band of the
  artwork.
- Elephant's foot on every strip's bottom edge, which is a seam for all
  but row 1: compensate 0.2 mm in the slicer or sand before assembly.
- 17 seams between neighbouring strips (16 of 330 mm and one of 660 mm,
  9 cut lines, 5.9 m in total) against 4 for a flat 2 x 2 cut.
- Layer lines run across the relief along x at the layer height, and
  back-leaning facets print as stairs of width layer / tan(alpha).
- The engraving walls are 42 degree overhangs at 2 mm pitch and 50
  degrees at 1.5 mm (0.6 mm deep over one 0.5 mm node), a few cm^2.
- One job of 18 tall thin walls: a failure late in the print costs the
  whole plate. Slenderness 7.3 against a limit of 8, f1 about 500 Hz.
