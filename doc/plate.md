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
row 0 at the bottom. The A to height mapping is a plug-in (`--mapping`):
`clip`, v = clip(A / m, -1, 1) with m = percentile(|A|, `--clip`, 99.5),
the clip the paper prints use; `tanh`, a soft knee at the same m; `none`,
v = A / max|A|, so the largest entry takes the full range and nothing is
flattened. The final builds use `none`. Optional Gaussian blur in cells,
renormalised to peak 1. Fine-grid nodes at k p / subdiv (subdiv 4 at
1.5 mm pitch: 0.375 mm; subdiv 3 at 2 mm: 0.667 mm).
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
with halos around every spike. The final builds use Fourier (see "Design
decisions").

Overshoot means the printed range exceeds the nominal relief. The
surface measures the range after interpolation; with `max_range_mm`
(thickness minus body, 10 mm at t = 13 and body 3) it scales the relief
down so the printed range fits, never clips, and shifts the top to z = t.
Both final builds hit this cap (scale 0.709 at rho 0.6 and 0.597 at
rho 0.8).

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
prints. The final builds confirm it (`metrics.json`, slopes): at rho 0.6
the 99th percentile of |dh/dx| is 2.49 against 0.98 for |dh/dy|, and
27 % of the footprint exceeds 45 degrees in x against 0.9 % in y; at
rho 0.8 it is 4.11 against 0.96 and 50 % against 0.7 %.

## Cutting, labels, print orientation

Cuts snap to the nearest fine node, so neighbours share their seam nodes
exactly and strips may differ by one fine cell (94.1 and 94.5 mm in the
final builds, one 0.375 mm node). Labels: column letter plus row number, A1 bottom left, engraved
0.6 mm into the back near the strip's top-left corner, 6 mm cap height
from Pillow's bundled font, mirrored in x so they read when the back is
viewed from behind, with a filled triangle pointing to plate +y; the
build refuses a label whose ink would reach a boundary node.

Print orientation is the rotation (x, y, z) to (x, -z, y), determinant
+1, never a mirror. The relief faces the printer front (-Y), the back is
a vertical wall, strip k (label order A1 .. A7, B1 .. B7) is the k-th slot
from the front at Y in [y_k - t, y_k], y_k = margin + t + k (t + gap). The
14 strips of the final builds end at Y = 291 mm of 340. `--orientation
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
  neighbour both ways within 1e-3 mm (STL is float32). Measured 1.5e-5 mm
  in both final builds.

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

Sizes are the final builds (pitch 1.5, subdiv 4, 1761 x 1761 nodes,
12.45 M faces); both rho values give the same sizes.

| file | what | size |
|---|---|---|
| plate.stl | the uncut plate, plate frame, flat back | 310.8 MB |
| bed.stl | every strip placed and oriented on the bed: print this | 622.7 MB |
| strips/<label>.stl | the same, one file per strip | 622.7 MB total |
| layout.json | bed, per-strip bboxes, 4x4 transforms, overhang per strip | 24 KB |
| metrics.json | plate totals, slopes, overhang histograms, relief choice, loop spec | 4 KB |
| assembly.md | label grid as hung, seams, print settings | 3 KB |
| preview_*.png | plate hillshade with seams; bed drawing plus oblique mesh; worst strip coloured by overhang | |
| render/*.png | front, three-quarter, close-up, bed and strip B3 views from `scripts/render_stl.py` | |
| A.npy, surface_Z.npy | the matrix and the fine node heights | 1.5 + 24.8 MB |
| loop.wav | the excerpt at the source rate | 0.4 MB |

`--write-cut` adds plate_cut.stl (all strips in the plate frame, labels
engraved, about the size of bed.stl) and `--plate-strips` the same per
strip; the visual tests use them. Face count scales with subdiv squared:
subdiv 3 at the same pitch gives 7.0 M faces and a 351 MB bed.stl.

## Export package

`loopviz plate export --build runs/plate/<name> --out export/<name> [--zip]`
(`src/loopviz/print3d/export.py`) assembles a self-contained directory from
a finished build:

- `audio/loop_clean.wav`: the excerpt at the source rate (16 kHz,
  12.025 s); `loop_clean_x3.wav` plays it three times.
- `audio/loop_reconstructed_x3.wav`: the exact operator played at
  round(f) Hz (9660 and 12880 Hz). Window 1 seeds it, A is applied N
  times per pass for three passes, every window scaled back by its norm.
- `audio/loop_as_printed_x3.wav`: the same with the operator the plate
  carries as the export models it: |A| clipped at the clip percentile
  under the `clip` mapping, every entry carried under `none`, and heights
  in 0.05 mm steps (164 levels at rho 0.6). Degraded by design, see
  `doc/relief.md`: for both final builds it diverges and is cut to
  silence (after 45 windows, 2.05 s, at rho 0.6; after 7 windows at
  rho 0.8).
- `spec_sheet.md` and `.json`: every parameter, overhang, mass, slicer
  numbers if `slice_*.json` exists in the build, drift per pass, commit,
  version, build and export dates.
- `analysis/`: previews, metrics, layout, assembly notes, the design
  study (`--design-study`, RESULTS.md and figures) and `render/`.
- `stl/plate_full.stl` (the uncut plate, reference geometry),
  `stl/bed_print_as_is.stl` (print this), `stl/strips/<label>.stl`.
- `renders/`: the build's `render/*.png`, else the previews.
- `README.md`: what the object is, what each file is, and the assembly
  notes.

The build records its loop spec (`loops/<name>.json`, see
`src/loopviz/loopspec.py`) in `metrics.json`; older builds with only
audio/start/end still export. The shipped packages are
`export/armed_man_p1.5_rho0.6.zip` (595 MB) and
`export/armed_man_p1.5_rho0.8.zip` (598 MB), 48 files and 1.6 GB each
unpacked; the STLs dominate.

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
into slice_<preset>.json. The final builds were not sliced: the printer
owner slices with their own profile. The only sliced geometry is the
earlier pitch 2 bicubic bed (`runs/plate/smoke_p2/bed.stl`, 18 strips,
3.96 M faces, the same geometry as the pitch 2 build below), with
PrusaSlicer 2.9.4 and the generic04 preset:

```
manifold=True facets=3963960 time=4d 17h 43m 28s (113.7 h) filament=2095.4 g / 1689.84 cm3 layers=367 maxZ=73.4 slice=1160.5 s
warning: Detected print stability issues: Low bed adhesion. Consider enabling supports.
```

So 114 h and 2.1 kg at generic speeds. The build's shell estimate for
that bed (`material_estimate` in `metrics.json`: 3 walls of 0.45 mm under
every face, 15 % infill of the rest) is 2.11 kg and 79 crude hours, so
the mass model is within 1 % of the slicer and the time model is 1.45x
too optimistic. The hand-over's 35 to 50 h is not reached. The final
builds are 4.09 and 4.30 kg solid, 2.21 and 2.46 kg in the shell
estimate, 82 and 92 crude hours; scaled by the same 1.45x that is about
120 and 135 h at generic speeds.

## The armed-man builds

Loop `loops/armed_man.json`: The Armed Man: A Mass for Peace, XII.
Benedictus (Karl Jenkins), 324.68 to 336.705 s of data/audio_3d/song.wav,
T = 12.025 s, 16 kHz source. The loop ends on the timpani hit 12.025 s
after the start, found by cross-correlating the 40 to 200 Hz envelope.
Two final candidates, pitch 1.5 mm at rho 0.6 and 0.8, Fourier
interpolation, mapping none (no clipping), subdiv 4, 2 x 7 strips of
330 x 94.3 x 13 mm on edge, 10 mm printed range, no blur, labels on,
relief auto under the facet rule:

```
uv run loopviz plate build --loop loops/armed_man.json --pitch 1.5 --rho 0.6 --side 660 \
    --cols 2 --rows 7 --thickness 13 --relief auto --interp fourier --mapping none --subdiv 4 \
    --out runs/plate/armed_man_p1.5_rho0.6_fourier_t13
uv run loopviz plate build --loop loops/armed_man.json --pitch 1.5 --rho 0.8 --side 660 \
    --cols 2 --rows 7 --thickness 13 --relief auto --interp fourier --mapping none --subdiv 4 \
    --out runs/plate/armed_man_p1.5_rho0.8_fourier_t13
```

All numbers from each build's `metrics.json`:

| | rho 0.6 | rho 0.8 |
|---|---|---|
| n, N | 440, 264 | 440, 352 |
| f (Hz), Nyquist (Hz) | 9660, 4830 | 12880, 6440 |
| drift per pass, Gram condition | 7.7e-12, 6.2e3 | 7.2e-10, 2.2e5 |
| nodes, faces | 1761 x 1761, 12.45 M | 1761 x 1761, 12.45 M |
| relief fit, facet rule, 99 % under 45 deg | 11.47 mm (y rule 8.30) | 16.66 mm (y rule 10.39) |
| relief chosen (scaled to the 10 mm cap) | 8.13 mm, scale 0.709 | 9.95 mm, scale 0.597 |
| thickness the fit would need | 17.1 mm | 19.7 mm |
| printed range, body under the trough | 3.0 to 13.0 mm, 3.0 mm | 3.0 to 13.0 mm, 3.0 mm |
| relief face over 45 deg | 0.18 % (10.4 of 5933 cm^2), max 65 deg, strip B4 | 0.08 % (5.7 of 7456 cm^2), max 64 deg, strip B5 |
| whole bed over 45 deg, with engraving | 11.9 cm^2 | 7.2 cm^2 |
| mass, solid / shell estimate | 4.09 / 2.21 kg | 4.30 / 2.46 kg |
| crude time | 82 h | 92 h |
| plate.stl, bed.stl, strips/ | 310.8, 622.7, 622.7 MB | 310.8, 622.7, 622.7 MB |
| build time, reassembly deviation | 32 s, 1.5e-5 mm | 32 s, 1.5e-5 mm |

Strip sizes 330 x 94.1 and 330 x 94.5 x 13 mm, seams at x = 330 and
y = 94.1, 188.6, 282.8, 377.2, 471.4, 565.9; slenderness h/t = 7.2
(limit 8); the strips end at Y = 291 mm of the 340 mm bed.

The overhang rule is not what limits these plates either: the facet fit
would allow 11.5 and 16.7 mm of nominal relief, the 13 mm strip allows
10 mm of printed range, and the relief is scaled down to fit. The
measured printed range is 1.23x the nominal relief at rho 0.6 and 1.00x
at rho 0.8, so without clipping the Fourier overshoot is small on these
matrices. History: the first builds were pitch 2 and 1.5 at rho 0.95 on
a 13.5 s loop (324.68 to 338.18 s), bicubic, clip 99.5, subdiv 3, 2 x 9
strips of 330 x 73 x 10 mm with a 7 mm printed range, about 3.5 kg solid,
scaled by cubic overshoot of 1.21x and 1.14x to 5.8 and 6.2 mm of relief
(`runs/plate/armed_man_p2`, `armed_man_p1.5`).

## Design decisions

- Fourier interpolation. It is the canonical band-limited kernel for a
  sampled field: the ringing around a spike is the true shape of the
  band-limited surface, not an artefact of a local polynomial. The
  price on the Q3 patch is printed range per nominal relief of 1.78x
  (1 + 0.28 overshoot + 0.50 undershoot) against 1.26x for bicubic. On
  the final matrices, unclipped, it is 1.23x and 1.00x.
- No clipping (`--mapping none`). The tail of this operator is short:
  on the clip-study matrix (pitch 1.5, rho 0.95) max |A| is 1.6x the
  99.5th percentile, 1.86x and 1.75x on the rho 0.6 and 0.8 matrices.
  So not clipping costs about a quarter of the bulk relief and nothing
  is flattened. From `runs/plate/clip_study/clip_interp_grid.png` (80 mm
  window, 10 mm printed range everywhere, bulk is the p10 to p90 height
  spread, over 45 is the share of y facets):

  | mapping | bicubic bulk, over 45 | Fourier bulk, over 45 |
  |---|---|---|
  | clip 99.5 | 3.0 mm, 0.8 % | 2.4 mm, 0.2 % |
  | soft tanh | 3.5 mm, 1.5 % | 2.7 mm, 0.3 % |
  | none | 2.2 mm, 0.1 % | 2.3 mm, 0.1 % |

- Thickness 13 mm. The overhang rule allows 11.5 and 16.7 mm of
  nominal relief, which with the Fourier overshoot would need 17.1 and
  19.7 mm strips. Thickness minus the 3 mm body caps the printed range
  at 10 mm. The amplitude study at pitch 2 (`runs/plate/amplitude/stats.json`)
  gives 7 / 10 / 14 mm printed range (strips of 10 / 13 / 17 mm, 2 x 9 /
  2 x 7 / 2 x 5) as 0.3 / 2.1 / 5.5 % of the hanging relief facets over
  45 degrees (0.16 / 0.99 / 2.7 % of the whole footprint, max 63 / 70 /
  76 degrees). 10 mm is the step before the over-45 area takes off.
- Subdiv 4. The ridges in x have a 3 mm period (2 cells). A 3 mm
  peak-to-peak sinusoid of that period sampled at 0.5 mm nodes (subdiv 3)
  has a chord error of 0.20 mm at the crest, 0.11 mm at 0.375 mm nodes
  (subdiv 4): below the layer height and a quarter of the 0.45 mm line
  width. The cost is 1.8x the faces and file size.
- Pitch 1.5 and rho 0.6 / 0.8, chosen by eye from
  `runs/plate/pitch_sheet/pitch_sheet.md` (bicubic, clip 99.5, 10 mm
  printed range). Pitches 1.2 and 1.35 at rho 0.95 have no operator on a
  16 kHz source: f = 23.9 and 18.8 kHz exceeds the source rate, the Gram
  matrix is singular and the drift is 1.0. The over-45 fraction tracks
  f/2 relative to the music, not the pitch:

  | pitch, rho | f kHz | over 45 (footprint) |
  |---|---|---|
  | 1.35, 0.8 | 15.8 | 0.23 % |
  | 1.5, 0.95 | 15.3 | 0.22 % |
  | 1.2, 0.6 | 15.1 | 0.51 % |
  | 1.5, 0.8 | 12.9 | 0.44 % |
  | 1.7, 0.95 | 11.9 | 0.38 % |
  | 1.5, 0.6 | 9.7 | 1.37 % |
  | 2.0, 0.95 | 8.6 | 0.73 % |
  | 2.5, 0.95 | 5.5 | 1.31 % |

  Lower f means fewer windows and a wider texture; rho 0.6 and 0.8 at
  pitch 1.5 bracket the look. Unclipped and Fourier, the final builds
  come in far under these sheet values (0.18 and 0.08 %).

## What is lost on edge

- No colour change by layer. Flat, a single M600 at mid relief painted
  the sign of every entry; on edge a layer is a horizontal band of the
  artwork.
- Elephant's foot on every strip's bottom edge, which is a seam for all
  but row 1: compensate 0.2 mm in the slicer or sand before assembly.
- 13 seams between neighbouring strips (12 of 330 mm and one of 660 mm,
  7 cut lines, 4.6 m in total) against 4 for a flat 2 x 2 cut.
- Layer lines run across the relief along x at the layer height, and
  back-leaning facets print as stairs of width layer / tan(alpha).
- The engraving walls are 58 degree overhangs at subdiv 4 (0.6 mm deep
  over one 0.375 mm node); they add 1.5 cm^2 over 45 degrees to the
  relief face's 10.4 and 5.7 cm^2 in the final builds.
- One job of 14 tall thin walls: a failure late in the print costs the
  whole plate. Slenderness 7.2 against a limit of 8, f1 about 390 Hz
  from the cantilever model above.
