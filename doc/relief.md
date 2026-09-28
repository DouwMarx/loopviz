# The operator as a 3D print

`loopviz relief` turns the exact n x n window-advance operator A0 into a
stepped relief: one square column per entry, top height = entry. Same
object as the paper prints, displayed in height instead of ink.

## Sizing

Three numbers tie everything (songmatrix.py): `f T = N n = rho n^2`.
On a bed of side S with cell pitch p, `n = floor(S / p)`, so for a fixed
loop of T seconds the implied sample rate is

    f = rho n^2 / T = rho (S / p)^2 / T

Halving the pitch quadruples f. The height axis is the other budget: a
column top lands on a layer boundary, and a one-layer step hides in the
top-surface noise (about +-0.05 mm between neighbours), so a level is two
layers: with relief height R and layer height h there are `R / 2h + 1`
distinguishable heights, about 5 bits on FDM. The grid is one solid, so
a cell is a free-standing column only over the part that rises above its
tallest neighbour; `max_protrusion` measures that and the build warns
when it exceeds `max_aspect` pitches.

`loopviz relief sweep --probe` prints this table for a loop and printer,
bisects the largest rank fraction rho whose exact operator still loops
(drift per pass <= tol), and measures how the quantized object plays.

Benedictus loop 5:24.68 to 5:38.18 (T = 13.5 s), 300 mm usable, rho =
0.95 (all stable: the exact operator loops with drift <= 3e-8 per pass,
rho 0.97 still fine), 6 mm relief, 2-layer levels:

| pitch mm | nozzle | n | f Hz | Nyquist Hz | levels | squares |
|---|---|---|---|---|---|---|
| 1.2 | 0.2 | 250 | 4407 | 2204 | 51 | crisp only with a 0.2 mm nozzle |
| 1.5 | 0.25 | 200 | 2815 | 1407 | 61 | crisp with 0.25 mm |
| 2.0 | 0.4 | 150 | 1578 | 789 | 31 | marginal with 0.4 mm (corner radius ~0.22 mm) |
| 2.5 | 0.4 | 120 | 1013 | 507 | 31 | crisp with 0.4 mm |
| 3.0 | 0.4 | 100 | 704 | 352 | 31 | bass only |

The source is 16 kHz; the paper prints run 4 to 16 kHz. On one 300 mm
plate the loop lands at 1 to 4.4 kHz depending on nozzle: telephone
bandwidth at best, kick-and-timpani-only at 2.5 mm. Two ways out, both
in the CLI:

- `--tiles 2`: four 300 mm prints assembled into 600 mm. n doubles, f
  quadruples: 2.0 mm pitch gives n = 300, f = 6.3 kHz; 2.5 mm gives
  n = 240, f = 4.1 kHz. Tiles are cut on cell edges and written as
  separate watertight STLs.
- a shorter loop: 2 bars (6 s, 5:24.68 to 5:30.68) at 2.0 mm gives
  f = 3.6 kHz on one print; at 1.5 mm with a 0.25 mm nozzle, 6.3 kHz.

## Printer limits (see sources below)

Defaults in `relief.Printer`, overridable by flag or environment
(`LOOPVIZ_BED_MM`, `LOOPVIZ_NOZZLE_MM`, `LOOPVIZ_LAYER_MM`,
`LOOPVIZ_STEP_LAYERS`, `LOOPVIZ_MIN_PITCH_MM`, `LOOPVIZ_MAX_ASPECT`,
`LOOPVIZ_MAX_RELIEF_MM`, `LOOPVIZ_MARGIN_MM`).

| preset | nozzle | layer | layers/level | min crisp pitch | max protrusion / pitch | bed |
|---|---|---|---|---|---|---|
| fdm04 | 0.4 mm | 0.10 mm | 2 | 2.0 mm (2.5 safe) | 5 | 305 mm |
| fdm025 | 0.25 mm | 0.05 mm | 2 | 1.5 mm | 5 | 305 mm |
| fdm02 | 0.2 mm | 0.06 mm | 2 | 1.2 mm | 5 | 305 mm |
| msla | 43 um pixel | 0.05 mm | 1 | 0.5 mm | 8 | 185 mm |

Where the numbers come from (researched 2026-09, no published pillar
test exists for stepped grids; a 60 mm test tile before the 300 mm print
is the honest check):

- Crisp pitch: a flat top needs >= 2 perimeters plus top fill, so >= 4
  line widths (1.8 mm at 0.45 mm lines, 1.1 mm at 0.28); the outer
  corner radius is at least half a line width plus deceleration ooze, and
  keeping it under ~10 % of the side gives 2.2 to 2.5 mm (0.4) and 1.4 to
  1.5 mm (0.25). Design guides put pins at >= 4 line widths, 3 mm safe.
  Arachne's minimum feature size (25 % of nozzle) only stops geometry
  being dropped. Input shaping on fast printers rounds small details;
  slow the outer wall (~40 mm/s) or use a 0.2 mm nozzle.
  https://blog.prusa3d.com/everything-about-nozzles-with-a-different-diameter_8344/
  https://support.3dverkstan.se/article/38-designing-for-3d-printing
  https://xometry.pro/en/articles/fdm-design-tips/
  https://help.prusa3d.com/article/arachne-perimeter-generator_352769
  https://forum.bambulab.com/t/what-settings-to-prevent-small-details-edges-from-being-rounded/12933
- Layers: 0.05 to 0.30 mm at 0.4 (max 80 % of nozzle), 0.05 to 0.20 at
  0.25, 0.04 to 0.14 at 0.2. Layers are global, so variable layer height
  cannot give per-cell sub-layer resolution; it only allows thin layers
  in the relief band and thick ones in the plinth. Print the relief band
  solid (100 % rectilinear infill or >= 3 top layers) so every top is
  flat. Prusa MK4 input-shaper profiles ship 0.10 to 0.20 mm only.
  https://help.prusa3d.com/article/layers-and-perimeters_1748
  https://help.prusa3d.com/article/variable-layer-height-function_1750
  https://github.com/prusa3d/PrusaSlicer-settings/issues/240
  https://github.com/bambulab/BambuStudio/issues/6647
- Protrusion: rules of thumb only. 8:1 for unsupported walls, columns
  >= 2 to 3 mm, 4 mm x 35 mm prints fine, failure is nozzle-drag wobble
  and cooling, not adhesion. 5 pitches is the conservative reading.
  https://www.core77.com/posts/74401/Design-Rules-for-3D-Printing
  https://formlabs.com/blog/minimum-wall-thickness-3d-printing/
  https://forum.prusa3d.com/forum/original-prusa-i3-mk3s-mk3-how-do-i-print-this-printing-help/recommendations-rules-of-thumb-for-printing-tall-thin-objects/
- Z accuracy: measured height deviation on 3 mm PLA specimens 2.3 to
  12.2 % (worst at 0.20 mm layers, best at 0.10 to 0.15); +-0.1 mm
  absolute, ~+-0.05 mm between adjacent cells. First layer is squished
  and elephant-foot compensated (0.2 mm XY at 0.4), so the first 0.5 mm
  above the bed is not usable relief: the 2 mm base plinth covers it.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10223179/
  https://help.prusa3d.com/article/elephant-foot-compensation_114487
  https://www.rapiddirect.com/blog/3d-printing-tolerances-how-accurate-is-3d-printing/
- Resin: the largest consumer MSLA plates are 16:9 (Phrozen Sonic Mega
  8K S 330 x 185 mm at 43 um, Elegoo Jupiter 2 302 x 162, Anycubic M7 Max
  298 x 164); the Peopoly Phenom XXL V2 is 527 x 296 mm at 137 um. No
  single plate takes 305 x 305; two 305 x 152 tiles fit the Mega 8K S.
  Resin gives crisp 0.5 mm cells but large flat slabs warp and peel.
  https://phrozen3d.com/en/products/sonic-mega-8k-s
  https://www.elegoo.com/pages/elegoo-jupiter-2
  https://store.anycubic.com/products/photon-mono-m7-max
  https://peopoly.net/products/phenom-xxl-v2-by-peopoly
- Colour change at a layer boundary is a single-extruder feature (M600
  pause) in PrusaSlicer, OrcaSlicer and Bambu Studio.
  https://help.prusa3d.com/article/color-change_1687
  https://www.orcaslicer.com/wiki/printer_settings/multimaterial/printer_multimaterial_setup

## Does the printed object still play?

`scripts/exp_relief_quantization.py` measures the object as printed:
Q(A0), the entry-wise quantization of the exact operator to L heights,
applied to the loop's windows. Step error is `|Q(A0) w_k - w_(k+1)| /
|w_(k+1)|` (1 = as wrong as silence). Findings for this loop:

- Clipping the top 0.5 % of |entries| (the paper prints' default) is
  fatal at any bit depth: at n = 250 the step error stays >= 12 % at rho
  0.7 and >= 100 % at rho 0.95 even with 65536 levels (61 % at n = 150).
  The largest entries carry playback.
- With no clipping, playback needs about 12 bits at rho 0.95 (4096
  levels: 1.2 %) and about 10 bits at rho 0.5 (1024 levels: 0.4 %). FDM
  offers 5 to 6 bits.
- At 5 to 6 bits the best the object can do is 3 to 5 % step error, with
  rho 0.35 to 0.5 and the free part of the operator used to absorb the
  quantization (alternating projections between the height grid and the
  exact-solution set; `alt-proj` columns). Iterated over a full loop the
  error compounds to order 1.

So the relief is a faithful portrait of the operator, not a machine that
plays from calipers. The paper prints are in the same position: 8-bit
ink with 99.5 % clip is >= 7 % per step at rho 0.5 and unplayable at
rho 0.95. The exactness claim lives in the matrix; every physical display
of it is quantized. `relief.json` records both numbers for each build.

## Mesh

`relief.heightfield_mesh` writes the stepped solid directly (numpy, no
CAD kernel): cell tops, a base grid, vertical walls only where neighbours
differ, the outer skirt. Wall edges are split wherever a neighbouring
cell top meets them, so there are no T-junctions: every edge is shared by
exactly two faces, except vertical edges where two columns touch only
diagonally (four faces). That is a closed solid and PrusaSlicer slices
it cell-for-cell correctly (verified per layer from the G-code), but
strict two-manifold checkers flag it: `trimesh.is_watertight` is false,
PrusaSlicer shows an "auto-repaired" badge (`--info` says manifold=no)
without changing a single facet, and OpenSCAD's CGAL kernel refuses the
mesh for booleans. Do boolean edits (frames, hanging holes, text) on the
height map before meshing, not on the STL.
`check_mesh` verifies closure, orientation and the analytic volume
`pitch^2 * sum(H)`; the tests do this for every neighbour configuration.

Rendering: `mesh_views` draws the STL's own triangles (n <= 60);
`hillshade` lights the height map for any n; `heightmap16.png` is a
16-bit height image for slicers that import one (they interpolate; the
STL is the authoritative stepped object). `loopviz relief demo` writes
2x2, 3x3, 4x4 and 12x12 cases to `runs/relief/demo/` for inspection.

## Print settings

- STL units are mm, z = 0 is the bed, no supports needed (every
  overhang is a vertical wall).
- Layer height must equal the `layer_mm` used at build time so every
  column top lands on a layer boundary; first layer height is part of the
  2 mm base and does not matter. Verified with PrusaSlicer 2.9.4 CLI
  (`nix build nixpkgs#prusa-slicer`) on the demo meshes and the 300 mm
  build: every layer's toolpath matches the height map cell for cell.
  Sliced at 0.1 mm layers, 100 % rectilinear, the 2.0 mm / 150 x 150
  version is 79 layers, 563 g of PLA and 2 to 3.5 days depending on
  speeds: time is dominated by perimeters around 22 500 islands per
  layer, not by volume. `--layer 0.2 --step-layers 1` keeps the same 31
  levels and halves it, at the cost of rougher tops.
- Print the relief band solid (100 % rectilinear) so every column top is
  a full top surface; sparse infill leaves small enclosed cells hollow.
- Z-seam "rear" or "aligned" so the per-layer seam blobs line up on one
  face of each step.
- Colour change at z = base + relief / 2 (single-extruder M600) paints
  positive entries in the second colour: sign in colour, magnitude in
  height.
