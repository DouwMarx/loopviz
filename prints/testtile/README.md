# test tiles: print these before the 300 mm relief

One STL per cell pitch. Each is 12 cells wide; the bottom 12 rows (y from
0) are the actual corner of the full-size print at that pitch, with the
same 2.0 mm base, 6.0 mm relief and 0.2 mm level
step, so what you see is what the big print will look like. Above it,
from y = 12 cells upward:

- rows 0-11: song corner
- rows 13 : stair 1 layer/cell
- rows 14 : stair 2 layers/cell
- rows 15 : stair 4 layers/cell
- rows 16 : checker 1 step
- rows 17 : checker 5 steps
- rows 18 : spikes to full relief

Print with layer height 0.1 mm (the levels are 2 layers
each), 100 % rectilinear infill, no supports, seam aligned. Then judge:
do the squares read as squares (corner rounding), which staircase step
is the smallest you can see and feel, do the spikes print clean.

| pitch mm | n of full print | f Hz | tile mm | levels | max protrusion in full print (mm) | file |
|---|---|---|---|---|---|---|
| 1.5 | 200 | 2815 | 18 x 27 x 8 | 31 | 3.8 | testtile_p1.5.stl |
| 2 | 150 | 1578 | 24 x 36 x 8 | 31 | 4.2 | testtile_p2.stl |
| 2.5 | 120 | 1013 | 30 x 45 x 8 | 31 | 3.6 | testtile_p2.5.stl |
| 3 | 100 | 704 | 36 x 54 x 8 | 31 | 3.2 | testtile_p3.stl |

Loop 324.68 to 338.18 s of song.wav; printer preset
fdm04 (0.4 mm nozzle).
