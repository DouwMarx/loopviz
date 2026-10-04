"""`loopviz relief`: size, build and preview a 3D-printed song operator.

  loopviz relief sweep --audio song.wav --start 324.68 --end 338.18
      (or --loop loops/armed_man.json for every command that takes audio)
      pitch -> (n, f, levels, bits) table for the printer; with --probe
      also the largest stable f and the playback of the quantized object
  loopviz relief build --audio song.wav --start 324.68 --end 338.18 --pitch 1.2
      STL + previews + relief.json in runs/relief/<name>/
  loopviz relief demo
      tiny hand-made reliefs (2x2, 3x3, 4x4, 12x12 from the song) to
      inspect the mesh construction before trusting a 300 mm print
  loopviz relief testtile --audio song.wav --start 324.68 --end 338.18
      one calibration tile per pitch: staircases, checkerboards, spikes
      and a 12x12 corner of the real print, to check the printer before
      the 300 mm run
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..loopspec import (  # noqa: F401  load_loop re-exported for scripts
    add_loop_args,
    load_loop,
    resolve_loop,
)
from ..songmatrix import Plan, build, loop_degradation, materialize
from .relief import (
    PRINTERS,
    ReliefPlan,
    check_mesh,
    heightfield_mesh,
    heights,
    max_protrusion,
    printer,
    quantized_matrix,
    relief_plan,
    sweep,
    write_stl,
)
from .reliefviz import heightmap_png, hillshade, mesh_views, sheet

DEFAULT_PITCHES = (0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)
RHO_CAP = 0.97          # measured existence limit (exp_operator_sizing)
RHO_FLOOR = 0.3


def drift_of(A, W, loops: int = 1) -> float:
    return loop_degradation(A, W, loops=loops)[0]


def probe(signal: np.ndarray, T: float, n: int, rho: float) -> dict:
    """Build the exact operator at (n, rho) and measure its loop drift."""
    N = max(2, round(rho * n))
    pl = Plan(n=n, N=N, f=N * n / T, T=T)
    op, W = build(signal, pl)
    return {"plan": pl, "op": op, "W": W,
            "drift": drift_of(op.factors(), W),
            "gram_cond": op.gram_condition()}


def max_feasible_rho(signal: np.ndarray, T: float, n: int, tol: float,
                     rho_cap: float = RHO_CAP, iters: int = 10) -> dict | None:
    """Largest rho with loop drift <= tol at this n (bisection), or None."""
    lo, hi = RHO_FLOOR, rho_cap
    best = None
    r = probe(signal, T, n, lo)
    if r["drift"] > tol:
        return None
    best = r
    r = probe(signal, T, n, hi)
    if r["drift"] <= tol:
        return r
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = probe(signal, T, n, mid)
        if r["drift"] <= tol:
            lo, best = mid, r
        else:
            hi = mid
    return best


def quantized_playback(A: np.ndarray, W: np.ndarray, levels: int,
                       clip_pct: float) -> dict:
    """How well the object as printed (clipped, `levels` heights) plays."""
    Aq = quantized_matrix(A, levels, clip_pct)
    Xn = np.roll(W, -1, axis=1)
    pred = Aq @ W
    step = np.linalg.norm(pred - Xn, axis=0) / np.linalg.norm(Xn, axis=0)
    return {"levels": levels, "clip_pct": clip_pct,
            "step_err_max": float(step.max()),
            "step_err_mean": float(step.mean()),
            "drift_per_pass": drift_of(Aq, W)}


def write_tiles(H: np.ndarray, pitch_mm: float, k: int, out: Path) -> list[Path]:
    """Cut the height map into k x k tiles, each its own watertight STL,
    named by row and column from the origin corner (x = column, y = row).
    Tiles butt together edge to edge; the seam runs along a cell edge."""
    out.mkdir(parents=True, exist_ok=True)
    n, m = H.shape
    if n % k or m % k:
        raise ValueError(f"{n}x{m} cells do not split into {k}x{k} equal tiles")
    rows = np.array_split(np.arange(n), k)
    cols = np.array_split(np.arange(m), k)
    paths = []
    for i, ri in enumerate(rows):
        for j, cj in enumerate(cols):
            sub = H[ri[0]:ri[-1] + 1, cj[0]:cj[-1] + 1]
            V, F = heightfield_mesh(sub, pitch_mm)
            chk = check_mesh(V, F, sub, pitch_mm)
            if not (chk["watertight"] and chk["volume_ok"]):
                raise RuntimeError(f"tile r{i} c{j} failed mesh checks: {chk}")
            paths.append(write_stl(V, F, out / f"tile_r{i}_c{j}.stl",
                                   name=f"tile r{i} c{j} ({sub.shape[0]}x{sub.shape[1]})"))
    return paths


def fmt_row(rp: ReliefPlan, extra: str = "") -> str:
    pl = rp.plan
    return (f"{rp.pitch_mm:5.2f} | {rp.n:4d} | {rp.side_mm:6.1f} | {pl.N:4d} | "
            f"{pl.f:7.0f} | {pl.f / 2:6.0f} | {rp.relief_mm:4.1f} | {rp.levels:3d} | "
            f"{rp.bits_per_cell:4.2f} | {rp.capacity_bits / 8 / 1024:6.1f}{extra}")


HEADER = ("pitch |    n |  side  |    N |   f Hz | Nyq Hz | relf | lvl | bits | "
          "KiB")


def cmd_sweep(args) -> None:
    pr = printer(args.printer, bed_mm=args.bed, layer_mm=args.layer,
                 margin_mm=args.margin, max_aspect=args.max_aspect,
                 step_layers=args.step_layers)
    spec = resolve_loop(args)
    signal, T, sr = spec.signal()
    print(f"loop {spec.name} {spec.start_s}..{spec.end_s} s: T = {T:.2f} s, source {sr} Hz; "
          f"printer {pr.name}: side {pr.side_mm:.0f} mm, layer {pr.layer_mm} mm, "
          f"nozzle {pr.nozzle_mm} mm")
    side = args.tiles * pr.side_mm     # for the banner only; n = k * floor(S/p)
    print(f"f = rho n^2 / T at rho = {args.rho}; Nyq = f/2 is the highest "
          f"frequency the object can carry"
          + (f"; {args.tiles}x{args.tiles} tiles = {side:.0f} mm assembled"
             if args.tiles > 1 else ""))
    print(HEADER + (" | f_max(drift<=tol) | quantized step err | q drift"
                    if args.probe else ""))
    rows = []
    for rp in sweep(T, pr, args.pitches, rho=args.rho, base_mm=args.base,
                    tiles=args.tiles):
        extra = ""
        d = {"pitch_mm": rp.pitch_mm, "n": rp.n, "N": rp.plan.N, "tiles": args.tiles,
             "side_mm": rp.side_mm, "f_hz": rp.plan.f, "relief_mm": rp.relief_mm,
             "levels": rp.levels, "bits_per_cell": rp.bits_per_cell,
             "capacity_bits": rp.capacity_bits,
             "crisp": rp.pitch_mm >= pr.min_pitch_mm}
        if args.probe:
            r = max_feasible_rho(signal, T, rp.n, args.drift_tol)
            if r is None:
                extra = " | none (song too long)"
                d["f_max_hz"] = None
            else:
                A = materialize(r["op"])
                q = quantized_playback(A, r["W"], rp.levels, args.clip)
                extra = (f" | {r['plan'].f:7.0f} (rho {r['plan'].rho:.2f}, "
                         f"drift {r['drift']:.0e}) | {q['step_err_max']:.2e} "
                         f"| {q['drift_per_pass']:.1e}")
                d.update({"f_max_hz": r["plan"].f, "rho_max": r["plan"].rho,
                          "exact_drift": r["drift"], "quantized": q})
        d["exceeds_source_rate"] = rp.plan.f > sr
        rows.append(d)
        notes = ("" if d["crisp"] else "  (below crisp pitch)") + (
            f"  (f > source {sr} Hz: upsampling)" if d["exceeds_source_rate"] else "")
        print(fmt_row(rp, extra) + notes)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")


def cmd_build(args) -> None:
    pr = printer(args.printer, bed_mm=args.bed, layer_mm=args.layer,
                 margin_mm=args.margin, max_aspect=args.max_aspect,
                 step_layers=args.step_layers)
    spec = resolve_loop(args)
    out = Path(args.out or f"runs/relief/{spec.name}_p{args.pitch:g}")
    out.mkdir(parents=True, exist_ok=True)
    signal, T, sr = spec.signal(out_wav=out / "loop.wav")
    rp = relief_plan(T, pr, args.pitch, rho=args.rho, relief_mm=args.relief,
                     base_mm=args.base, tiles=args.tiles)
    if args.rho_max:
        r = max_feasible_rho(signal, T, rp.n, args.drift_tol)
        if r is None:
            raise SystemExit("no stable rho at this n; shorten the loop")
        rp = ReliefPlan(plan=r["plan"], pitch_mm=rp.pitch_mm, relief_mm=rp.relief_mm,
                        base_mm=rp.base_mm, step_mm=rp.step_mm)
    else:
        r = probe(signal, T, rp.n, rp.plan.rho)
    pl = r["plan"]
    print(f"T = {T:.2f} s -> n = {rp.n}, N = {pl.N}, rho = {pl.rho:.3f}, "
          f"f = {pl.f:.0f} Hz; {rp.side_mm:.1f} mm square, relief {rp.relief_mm} mm "
          f"on {rp.base_mm} mm base, {rp.levels} levels of {rp.step_mm:g} mm "
          f"({rp.bits_per_cell:.2f} bit)")
    if rp.pitch_mm < pr.min_pitch_mm:
        print(f"WARNING: pitch {rp.pitch_mm} mm < crisp pitch {pr.min_pitch_mm} mm "
              f"for a {pr.nozzle_mm} mm nozzle: squares will round")
    print(f"exact operator: Gram cond {r['gram_cond']:.1e}, drift/pass {r['drift']:.1e}")
    A = materialize(r["op"])
    q = quantized_playback(A, r["W"], rp.levels, args.clip)
    print(f"as printed ({rp.levels} levels, clip {args.clip}%): step err max "
          f"{q['step_err_max']:.2e}, drift/pass {q['drift_per_pass']:.1e}")

    H = heights(A, rp, args.clip)
    prot = max_protrusion(H)
    print(f"max protrusion above tallest neighbour {prot:.2f} mm = "
          f"{prot / rp.pitch_mm:.1f} pitches (limit {pr.max_aspect:g})"
          + ("  WARNING: free-standing columns too tall" if prot > pr.max_aspect * rp.pitch_mm else ""))
    V, F = heightfield_mesh(H, rp.pitch_mm)
    chk = check_mesh(V, F, H, rp.pitch_mm)
    stl = write_stl(V, F, out / "relief.stl", name=out.name)
    print(f"mesh: {chk['faces']} faces, watertight={chk['watertight']}, "
          f"volume {chk['volume'] / 1000:.1f} cm^3 -> {stl} "
          f"({stl.stat().st_size / 1e6:.1f} MB)")
    if args.tiles > 1:
        tiles = write_tiles(H, rp.pitch_mm, args.tiles, out / "tiles")
        print(f"{len(tiles)} tile STLs of <= {pr.side_mm:.0f} mm in {out}/tiles/")
    np.save(out / "A.npy", A)
    np.save(out / "H_mm.npy", H)
    heightmap_png(H, out / "heightmap16.png")
    hillshade(H, rp.pitch_mm, out / "hillshade.png",
              title=f"{out.name}: n={rp.n} pitch={rp.pitch_mm} mm f={pl.f:.0f} Hz")
    if rp.n <= 60:
        mesh_views(V, F, out / "mesh_views.png", title=out.name)
    meta = {
        "audio": str(spec.audio), "start_s": spec.start_s, "end_s": spec.end_s,
        "loop": spec.to_dict(), "tiles": args.tiles,
        "T_s": T, "source_hz": sr, "printer": pr.__dict__,
        "n": rp.n, "N": pl.N, "rho": pl.rho, "f_hz": pl.f,
        "pitch_mm": rp.pitch_mm, "side_mm": rp.side_mm, "relief_mm": rp.relief_mm,
        "base_mm": rp.base_mm, "step_mm": rp.step_mm, "layer_mm": pr.layer_mm,
        "levels": rp.levels, "max_protrusion_mm": prot,
        "clip_pct": args.clip, "gram_cond": r["gram_cond"],
        "exact_drift_per_pass": r["drift"], "quantized": q, "mesh": chk,
        "level_histogram": np.bincount(
            ((H - rp.base_mm) / rp.step_mm).round().astype(int).ravel(),
            minlength=rp.levels).tolist(),
    }
    (out / "relief.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}/")


DEMOS = {
    "2x2_ramp": np.array([[1.0, 2.0], [3.0, 4.0]]),
    "3x3_checker": np.array([[3.0, 1.0, 3.0], [1.0, 3.0, 1.0], [3.0, 1.0, 3.0]]),
    "3x3_flat_edge": np.array([[2.0, 2.0, 2.0], [2.0, 5.0, 2.0], [2.0, 2.0, 1.0]]),
    "4x4_random": None,      # filled from a seeded RNG
}


def cmd_demo(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pr = printer(args.printer)
    pitch = args.pitch
    rng = np.random.default_rng(3)
    cases = dict(DEMOS)
    cases["4x4_random"] = 1.0 + 0.1 * rng.integers(0, 30, (4, 4))
    if args.audio or args.loop:
        signal, T, _ = resolve_loop(args).signal()
        rp = relief_plan(T, pr, pitch, rho=0.8, side_mm=12 * pitch, relief_mm=5.0)
        assert rp.n == 12
        r = probe(signal, T, rp.n, rp.plan.rho)
        cases[f"{rp.n}x{rp.n}_song"] = heights(materialize(r["op"]), rp)
    rows, pngs = [], []
    for name, H in cases.items():
        H = np.asarray(H, dtype=np.float64)
        V, F = heightfield_mesh(H, pitch)
        chk = check_mesh(V, F, H, pitch)
        write_stl(V, F, out / f"{name}.stl", name=name)
        png = mesh_views(V, F, out / f"{name}.png",
                         title=f"{name}: pitch {pitch} mm, heights {H.min():g}..{H.max():g} mm")
        np.savetxt(out / f"{name}.heights.txt", H, fmt="%.2f")
        rows.append(f"| {name} | {H.shape[0]}x{H.shape[1]} | {chk['faces']} | "
                    f"{chk['watertight']} | {chk['nonmanifold_edges']} | "
                    f"{chk['volume']:.3f} | {chk['expected_volume']:.3f} |")
        pngs.append((name, png))
        print(f"{name}: {chk}")
    sheet(pngs, out / "demo_sheet.png", cols=2)
    (out / "README.md").write_text(f"""\
# relief demo: tiny meshes to inspect by eye and by slicer

Each case is an (n x m) height map in mm (`*.heights.txt`, row 0 is
y = 0, column 0 is x = 0) turned into a stepped solid (`*.stl`) with
cell pitch {pitch} mm, and drawn from four viewpoints (`*.png`; the
drawing is of the STL's own triangles). `demo_sheet.png` collects them.

| case | cells | faces | watertight | voxel edges | volume mm^3 | expected |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

"voxel edges" are vertical edges where two columns touch only along an
edge (diagonal neighbours both taller than the other two); four faces
meet there. The solid is closed and slicers handle it, but strict
two-manifold checkers (trimesh.is_watertight) report it.
""")
    print(f"wrote {out}/demo_sheet.png")


def test_tile(H_song: np.ndarray, rp: ReliefPlan, layer_mm: float) -> np.ndarray:
    """Calibration patterns above a corner of the real relief. Rows from
    y = 0: the song corner (cells x cells), then staircases of 1, 2 and
    4 layers per cell, checkerboards of 1 and 5 level steps, and spikes
    at full relief every third cell (the worst free-standing column)."""
    m = H_song.shape[1]
    b, step, top = rp.base_mm, rp.step_mm, rp.base_mm + rp.relief_mm
    k = np.arange(m)
    rows = [
        np.minimum(b + k * 1 * layer_mm, top),
        np.minimum(b + k * 2 * layer_mm, top),
        np.minimum(b + k * 4 * layer_mm, top),
        b + (k % 2) * step,
        b + (k % 2) * 5 * step,
        np.where(k % 3 == 1, top, b),
    ]
    return np.vstack([H_song, np.array(rows)])


TILE_ROWS = ("song corner", "stair 1 layer/cell", "stair 2 layers/cell",
             "stair 4 layers/cell", "checker 1 step", "checker 5 steps",
             "spikes to full relief")


def cmd_testtile(args) -> None:
    pr = printer(args.printer, bed_mm=args.bed, layer_mm=args.layer,
                 margin_mm=args.margin, max_aspect=args.max_aspect,
                 step_layers=args.step_layers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = resolve_loop(args)
    signal, T, _ = spec.signal()
    c = args.cells
    lines, pngs = [], []
    for pitch in args.pitches:
        rp = relief_plan(T, pr, pitch, rho=args.rho, relief_mm=args.relief,
                         base_mm=args.base)
        r = probe(signal, T, rp.n, rp.plan.rho)
        H_full = heights(materialize(r["op"]), rp, args.clip)
        H = test_tile(H_full[:c, :c], rp, pr.layer_mm)
        V, F = heightfield_mesh(H, pitch)
        chk = check_mesh(V, F, H, pitch)
        if not (chk["watertight"] and chk["volume_ok"]):
            raise RuntimeError(f"pitch {pitch}: {chk}")
        name = f"testtile_p{pitch:g}"
        write_stl(V, F, out / f"{name}.stl", name=name)
        np.savetxt(out / f"{name}.heights.txt", H, fmt="%.2f")
        pngs.append((name, mesh_views(V, F, out / f"{name}.png",
                                      title=f"{name}: {H.shape[1] * pitch:g} x "
                                            f"{H.shape[0] * pitch:g} mm")))
        lines.append(f"| {pitch:g} | {rp.n} | {rp.plan.f:.0f} | {H.shape[1] * pitch:g} x "
                     f"{H.shape[0] * pitch:g} x {H.max():g} | {rp.levels} | "
                     f"{max_protrusion(H_full):.1f} | {name}.stl |")
        print(f"{name}: {H.shape[1] * pitch:g} x {H.shape[0] * pitch:g} mm, "
              f"{chk['faces']} faces, corner of the n={rp.n} print")
    sheet(pngs, out / "testtile_sheet.png", cols=2)
    rows_desc = "\n".join(f"- rows {c + i} : {d}" if i else f"- rows 0-{c - 1}: {d}"
                          for i, d in enumerate(TILE_ROWS))
    (out / "README.md").write_text(f"""\
# test tiles: print these before the 300 mm relief

One STL per cell pitch. Each is {c} cells wide; the bottom {c} rows (y from
0) are the actual corner of the full-size print at that pitch, with the
same {rp.base_mm} mm base, {rp.relief_mm} mm relief and {pr.step_mm} mm level
step, so what you see is what the big print will look like. Above it,
from y = {c} cells upward:

{rows_desc}

Print with layer height {pr.layer_mm} mm (the levels are {pr.step_layers} layers
each), 100 % rectilinear infill, no supports, seam aligned. Then judge:
do the squares read as squares (corner rounding), which staircase step
is the smallest you can see and feel, do the spikes print clean.

| pitch mm | n of full print | f Hz | tile mm | levels | max protrusion in full print (mm) | file |
|---|---|---|---|---|---|---|
{chr(10).join(lines)}

Loop {spec.start_s} to {spec.end_s} s of {spec.audio.name} ({spec.name}); printer preset
{pr.name} ({pr.nozzle_mm} mm nozzle).
""")
    print(f"wrote {out}/README.md")


def add_parser(sub) -> None:
    p = sub.add_parser("relief", help="3D-printed height-field version of the "
                                      "song operator")
    ss = p.add_subparsers(dest="relief_cmd", required=True)

    def common(q):
        add_loop_args(q)
        q.add_argument("--printer", default="fdm04", choices=sorted(PRINTERS))
        q.add_argument("--bed", type=float, default=None, help="bed side (mm)")
        q.add_argument("--margin", type=float, default=None, help="per side (mm)")
        q.add_argument("--layer", type=float, default=None, help="layer height (mm)")
        q.add_argument("--max-aspect", type=float, default=None,
                       help="max protrusion above the tallest neighbour, in pitches")
        q.add_argument("--step-layers", type=int, default=None,
                       help="layers per height level (default 2: one-layer "
                            "steps hide in top-surface noise)")
        q.add_argument("--base", type=float, default=2.0, help="base slab (mm)")
        q.add_argument("--rho", type=float, default=0.95, help="rank fraction N/n")
        q.add_argument("--clip", type=float, default=99.5,
                       help="percentile of |A| mapped to full relief")
        q.add_argument("--drift-tol", type=float, default=1e-6,
                       help="max loop drift per pass for a stable operator")
        q.add_argument("--tiles", type=int, default=1,
                       help="k x k prints assembled into one piece (k=2 "
                            "doubles n, quadruples f)")

    q = ss.add_parser("sweep", help="pitch -> (n, f, levels) table")
    common(q)
    q.add_argument("--pitches", type=lambda s: [float(v) for v in s.split(",")],
                   default=list(DEFAULT_PITCHES))
    q.add_argument("--probe", action="store_true",
                   help="also bisect the largest stable f per pitch and "
                        "measure playback of the quantized object")
    q.add_argument("--json", help="write the table here")
    q.set_defaults(fn=cmd_sweep)

    q = ss.add_parser("build", help="STL + previews for one pitch")
    common(q)
    q.add_argument("--pitch", type=float, required=True, help="cell side (mm)")
    q.add_argument("--relief", type=float, default=None,
                   help="height range (mm); default max_aspect * pitch")
    q.add_argument("--rho-max", action="store_true",
                   help="bisect the largest stable rho instead of --rho")
    q.add_argument("--out", help="output dir (default runs/relief/<audio>_p<pitch>)")
    q.set_defaults(fn=cmd_build)

    q = ss.add_parser("testtile", help="calibration tiles per pitch, with a "
                                       "corner of the real print")
    common(q)
    q.add_argument("--pitches", type=lambda s: [float(v) for v in s.split(",")],
                   default=[1.5, 2.0, 2.5, 3.0])
    q.add_argument("--cells", type=int, default=12, help="tile width in cells")
    q.add_argument("--relief", type=float, default=None,
                   help="height range (mm); default as the full print")
    q.add_argument("--out", default="prints/testtile")
    q.set_defaults(fn=cmd_testtile)

    q = ss.add_parser("demo", help="tiny meshes for inspection")
    q.add_argument("--out", default="runs/relief/demo")
    q.add_argument("--pitch", type=float, default=2.0)
    q.add_argument("--printer", default="fdm04", choices=sorted(PRINTERS))
    add_loop_args(q, required=False)        # also a 12x12 cut of this song
    q.set_defaults(fn=cmd_demo)
