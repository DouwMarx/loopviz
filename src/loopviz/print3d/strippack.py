"""Pack a square plate of side S, bigger than the bed, as on-edge strips.

The plate is cut into cols x rows rectangular strips and every strip is
printed standing on its long edge, all strips side by side on the bed in
one job (see the hand-over note edge-strip-packing-handover.md). With
L the usable bed length, Hz the printer height, g the gap between strips,
k the largest safe slenderness h/t and t the strip thickness:

    strip length   S <= cols * L
    strip height   S <= rows * h_max(t),   h_max = min(Hz, k t)
    bed packing    cols * rows <= slots(t) = floor((L + g) / (t + g))
    body           t >= relief_mm + body_min_mm   (the hand-over omits this)

Maximising S over integers cols, rows and the thickness grid is a small
mixed-integer problem; enumeration solves it. Material follows a shell
model (walls, relief band and edges solid, the core at the infill
fraction), the time estimate is volume / (flow * duty) and is crude.

Slenderness: a strip is a cantilever clamped at the bed. Its first
natural frequency is f1 = (1.875^2 / 2 pi) sqrt(E I / (rho A)) / h^2 with
I / A = t^2 / 12, which for PLA (E 3.5 GPa, rho 1240 kg/m^3) gives
f1 ~ 270 t / h^2 Hz (t, h in metres). Keep f1 above ~100 Hz.

Hand-over check for the 340 mm bed (L 330, Hz 325, k 8, g 8): t = 10 gives
2 x 9 strips and S = 660, global 1D optimum 3 x 3 at t = 29.5 with S = 708,
both confirmed; but the 2 x 9 strips measure 330 x 73.3 (not 330 x 80),
and S exceeds 660 for every t in (27.5, 29.5], so the plateau at 2L is
not unbroken. The hand-over's 2D remark is understated: allowing two
strips end to end per bed row (strip length <= (L - g) / 2) beats the 1D
optimum on this bed, 5 x 4 strips of 161 x 201 x 25.5 mm give S = 805
versus 708 (enumerate with max_per_row=2; `plate build` places one strip
per row, so this is an option for later, not the default).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

# -- inputs -------------------------------------------------------------------

@dataclass(frozen=True)
class Bed:
    """Printer envelope, all mm. Environment overrides: LOOPVIZ_BED_MM,
    LOOPVIZ_ZMAX_MM, LOOPVIZ_MARGIN_MM, LOOPVIZ_GAP_MM."""

    size_mm: float = 340.0     # square bed side
    zmax_mm: float = 325.0
    margin_mm: float = 5.0     # brim margin per bed side, L = size - 2 margin
    gap_mm: float = 8.0        # between strips (brim + clearance)
    brim_mm: float = 5.0       # brim width, must fit inside margin

    @property
    def usable_mm(self) -> float:
        return self.size_mm - 2 * self.margin_mm

    def slots(self, thickness_mm: float) -> int:
        """Strips of this thickness that fit side by side across the bed."""
        return int((self.usable_mm + self.gap_mm) // (thickness_mm + self.gap_mm))

    def strip_len_max(self, per_row: int = 1) -> float:
        """Longest strip when `per_row` strips sit end to end in one bed row."""
        return (self.usable_mm + self.gap_mm) / per_row - self.gap_mm


_ENV = {"size_mm": "LOOPVIZ_BED_MM", "zmax_mm": "LOOPVIZ_ZMAX_MM",
        "margin_mm": "LOOPVIZ_MARGIN_MM", "gap_mm": "LOOPVIZ_GAP_MM",
        "brim_mm": "LOOPVIZ_BRIM_MM"}


def bed(**overrides) -> Bed:
    """Defaults, then environment variables, then explicit overrides."""
    env = {k: float(os.environ[v]) for k, v in _ENV.items() if v in os.environ}
    return replace(Bed(), **{**env, **{k: v for k, v in overrides.items()
                                       if v is not None}})


@dataclass(frozen=True)
class Material:
    """Shell model parameters and the crude flow-rate time model."""

    wall_lines: int = 3
    line_width_mm: float = 0.45
    infill: float = 0.15
    density_g_cm3: float = 1.24      # PLA
    flow_mm3_s: float = 10.0         # sustained extrusion
    duty: float = 0.6                # fraction of wall time spent extruding
    area_factor: float = 1.4         # relief face area / footprint, before a surface exists

    @property
    def wall_mm(self) -> float:
        return self.wall_lines * self.line_width_mm


PLA = Material()


def printed_material(front_area_mm2: float, back_area_mm2: float, edge_area_mm2: float,
                     solid_volume_mm3: float, mat: Material) -> dict:
    """Printed volume, mass and crude time of a solid with these surfaces:
    a shell of `walls` lines under every face (the slicer's perimeters
    follow the relief, it does not fill the relief band solid) and the
    rest at the infill fraction. Calibrated against PrusaSlicer on the
    660 mm plate (3 walls, 15 % gyroid: 2095 g sliced)."""
    w = mat.wall_mm
    shell = w * (front_area_mm2 + back_area_mm2 + edge_area_mm2)
    shell = min(shell, solid_volume_mm3)
    infill = max(solid_volume_mm3 - shell, 0.0) * mat.infill
    vol = shell + infill
    return {"volume_mm3": vol, "solid_mm3": shell, "infill_mm3": infill,
            "mass_kg": vol * mat.density_g_cm3 * 1e-6,
            "hours": vol / (mat.flow_mm3_s * mat.duty) / 3600.0}


def strip_material(length_mm: float, height_mm: float, thickness_mm: float,
                   relief_mm: float, mat: Material) -> dict:
    """printed_material of ONE box strip before a surface exists: relief
    face area = footprint * mat.area_factor, back = footprint, edges from
    the box, solid volume = footprint * (thickness - relief / 2) since
    the entries are zero mean."""
    face = length_mm * height_mm
    edges = 2 * (length_mm + height_mm) * thickness_mm
    solid = face * max(thickness_mm - relief_mm / 2, 0.0)
    return printed_material(face * mat.area_factor, face, edges, solid, mat)


# -- layouts ------------------------------------------------------------------

@dataclass(frozen=True)
class StripLayout:
    cols: int
    rows: int
    thickness_mm: float
    side_mm: float             # plate side S
    strip_len_mm: float        # S / cols
    strip_height_mm: float     # S / rows
    n_strips: int              # cols * rows
    slots: int                 # bed rows available: floor((L + g) / (t + g))
    per_row: int               # strips end to end per bed row (1 = hand-over)
    binding: str               # what limits S: "len" (cols * L), "kt", "zmax"
    relief_mm: float
    volume_mm3: float          # whole plate
    mass_kg: float
    hours: float               # crude: volume / (flow * duty)

    @property
    def slenderness(self) -> float:
        return self.strip_height_mm / self.thickness_mm

    @property
    def f1_hz(self) -> float:
        """First cantilever natural frequency, PLA, 270 t / h^2 (metres)."""
        return 270.0 * (self.thickness_mm / 1e3) / (self.strip_height_mm / 1e3) ** 2

    @property
    def seams(self) -> int:
        """Internal cut lines, each of length S."""
        return (self.cols - 1) + (self.rows - 1)

    @property
    def seam_len_mm(self) -> float:
        return self.seams * self.side_mm

    @property
    def kg_per_m2(self) -> float:
        return self.mass_kg / (self.side_mm / 1e3) ** 2

    @property
    def label(self) -> str:
        s = f"{self.cols}x{self.rows}"
        return s if self.per_row == 1 else f"{s} @{self.per_row}/row"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(slenderness=self.slenderness, f1_hz=self.f1_hz, seams=self.seams,
                 seam_len_mm=self.seam_len_mm, kg_per_m2=self.kg_per_m2)
        return d


def h_max(bed: Bed, thickness_mm: float, k_slender: float) -> tuple[float, str]:
    """Tallest strip and which limit set it ("kt" or "zmax")."""
    kt = k_slender * thickness_mm
    return (kt, "kt") if kt < bed.zmax_mm else (bed.zmax_mm, "zmax")


def make_layout(bed: Bed, cols: int, rows: int, thickness_mm: float,
                k_slender: float, relief_mm: float, mat: Material,
                per_row: int = 1, side_mm: float | None = None) -> StripLayout | None:
    """The layout, or None if it does not fit the bed. With `side_mm`
    the side is fixed (feasibility check), else S is the largest side."""
    slots = bed.slots(thickness_mm)
    if cols * rows > slots * per_row:
        return None
    hm, hlim = h_max(bed, thickness_mm, k_slender)
    by_len, by_h = cols * bed.strip_len_max(per_row), rows * hm
    if side_mm is None:
        side_mm = min(by_len, by_h)
    elif side_mm > min(by_len, by_h) + 1e-9:
        return None
    if side_mm <= 0:
        return None
    binding = "len" if by_len <= by_h else hlim
    length, height = side_mm / cols, side_mm / rows
    m = strip_material(length, height, thickness_mm, relief_mm, mat)
    n = cols * rows
    return StripLayout(cols, rows, thickness_mm, side_mm, length, height, n, slots,
                       per_row, binding, relief_mm, n * m["volume_mm3"],
                       n * m["mass_kg"], n * m["hours"])


def t_grid(t_min: float, t_max: float, t_step: float) -> list[float]:
    n = round((t_max - t_min) / t_step)
    return [round(t_min + i * t_step, 6) for i in range(n + 1)]


def enumerate_layouts(bed: Bed, k_slender: float = 8.0, t_min: float = 5.0,
                      t_max: float = 40.0, t_step: float = 0.5, max_cols: int = 12,
                      max_rows: int = 12, relief_mm: float = 4.0,
                      body_min_mm: float = 3.0, max_per_row: int = 1,
                      mat: Material = PLA,
                      side_mm: float | None = None) -> list[StripLayout]:
    """Every feasible layout on the thickness grid (t >= relief + body)."""
    out = []
    for t in t_grid(t_min, t_max, t_step):
        if t < relief_mm + body_min_mm - 1e-9:
            continue
        for c, r, m in product(range(1, max_cols + 1), range(1, max_rows + 1),
                               range(1, max_per_row + 1)):
            lay = make_layout(bed, c, r, t, k_slender, relief_mm, mat, m, side_mm)
            if lay is not None:
                out.append(lay)
    return out


def _rank(lay: StripLayout) -> tuple:
    return (lay.side_mm, -lay.mass_kg, -lay.n_strips)


def best_layout(layouts: list[StripLayout]) -> StripLayout:
    """Largest S; ties go to the least material, then the fewest strips."""
    return max(layouts, key=_rank)


def best_per_t(layouts: list[StripLayout]) -> list[StripLayout]:
    by_t: dict[float, StripLayout] = {}
    for lay in layouts:
        cur = by_t.get(lay.thickness_mm)
        if cur is None or _rank(lay) > _rank(cur):
            by_t[lay.thickness_mm] = lay
    return [by_t[t] for t in sorted(by_t)]


def pareto(layouts: list[StripLayout]) -> list[StripLayout]:
    """S versus material frontier of the given candidates (the plan uses
    the best layout per thickness, otherwise tiny plates dominate):
    layouts no cheaper candidate matches or beats in S. Sorted by S
    ascending, mass strictly ascending."""
    front: list[StripLayout] = []
    for lay in sorted(layouts, key=lambda x: (x.mass_kg, -x.side_mm)):
        if not front or lay.side_mm > front[-1].side_mm + 1e-9:
            front.append(lay)
    return front


def recommend(front: list[StripLayout], max_ratio: float = 5.0) -> StripLayout:
    """From the cheapest frontier layout, repeatedly jump to the largest S
    reachable at no more than `max_ratio` percent material per percent
    side; stop when no such step exists. Jumping (not stepping) ignores
    kinks like a slightly bigger plate at a worse thickness. Material
    grows at least as S^2, so a ratio of 2 is the floor for any step; the
    default 5 separates thicker strips from a bigger plate."""
    i = 0
    while True:
        nxt = [j for j in range(len(front) - 1, i, -1)
               if step_ratio(front[i], front[j]) <= max_ratio]
        if not nxt:
            return front[i]
        i = nxt[0]


def step_ratio(a: StripLayout, b: StripLayout) -> float:
    """Percent material per percent side going from a to b."""
    ds = (b.side_mm - a.side_mm) / a.side_mm
    dm = (b.mass_kg - a.mass_kg) / a.mass_kg
    return dm / ds if ds > 0 else float("inf")


# -- CLI ----------------------------------------------------------------------

HEADER = (f"{'t':>5} {'layout':<12} {'S':>6} {'strip l x h':>14} {'slots':>7} "
          f"{'h/t':>5} {'f1 Hz':>6} {'seams':>5} {'kg':>6} {'kg/m2':>6} "
          f"{'crude h':>7}  binds")


def fmt_row(lay: StripLayout, flag: str = "") -> str:
    f1 = f"{lay.f1_hz:6.0f}" + ("!" if lay.f1_hz < 100 else " ")
    return (f"{lay.thickness_mm:5.1f} {lay.label:<12} {lay.side_mm:6.1f} "
            f"{lay.strip_len_mm:6.1f} x {lay.strip_height_mm:5.1f} "
            f"{lay.n_strips:3d}/{lay.slots * lay.per_row:<3d} {lay.slenderness:5.2f} "
            f"{f1}{lay.seams:5d} {lay.mass_kg:6.2f} {lay.kg_per_m2:6.2f} "
            f"{lay.hours:7.0f}  {lay.binding}{flag}")


def plan(bed: Bed, mat: Material, k_slender: float, t_min: float, t_max: float,
         t_step: float, max_cols: int, max_rows: int, relief_mm: float,
         body_min_mm: float, max_per_row: int, max_ratio: float,
         side_mm: float | None = None) -> dict:
    """Everything `loopviz plate plan` prints, as plain data."""
    kw = {"k_slender": k_slender, "t_min": t_min, "t_max": t_max, "t_step": t_step,
          "max_cols": max_cols, "max_rows": max_rows, "relief_mm": relief_mm,
          "body_min_mm": body_min_mm, "max_per_row": max_per_row, "mat": mat}
    layouts = enumerate_layouts(bed, **kw)
    if not layouts:
        raise ValueError("no feasible layout: raise --t-max or lower --relief/--body-min")
    per_t = best_per_t(layouts)
    front = pareto(per_t)
    rec = recommend(front, max_ratio)
    out = {"bed": asdict(bed), "material": asdict(mat),
           "params": {k: v for k, v in kw.items() if k != "mat"},
           "t_infeasible": [t for t in t_grid(t_min, t_max, t_step)
                            if t < relief_mm + body_min_mm - 1e-9],
           "best_per_t": [x.to_dict() for x in per_t],
           "best": best_layout(layouts).to_dict(),
           "pareto": [x.to_dict() for x in front],
           "recommended": rec.to_dict(), "max_ratio": max_ratio}
    if side_mm is not None:
        fixed = enumerate_layouts(bed, side_mm=side_mm, **kw)
        thin: dict[tuple, StripLayout] = {}
        for lay in fixed:
            key = (lay.cols, lay.rows, lay.per_row)
            if key not in thin or lay.mass_kg < thin[key].mass_kg:
                thin[key] = lay
        out["side"] = {"side_mm": side_mm,
                       "layouts": sorted((x.to_dict() for x in fixed),
                                         key=lambda d: d["mass_kg"]),
                       "lightest_per_grid": [x.to_dict() for x in
                                             sorted(thin.values(), key=lambda x: x.mass_kg)]}
    return out


def print_plan(res: dict) -> None:
    b, p = res["bed"], res["params"]
    L = b["size_mm"] - 2 * b["margin_mm"]
    print(f"bed {b['size_mm']:g} mm, margin {b['margin_mm']:g} -> L = {L:g} mm, "
          f"zmax {b['zmax_mm']:g}, gap {b['gap_mm']:g}, k = {p['k_slender']:g}, "
          f"relief {p['relief_mm']:g} + body {p['body_min_mm']:g} -> t >= "
          f"{p['relief_mm'] + p['body_min_mm']:g} mm, strips per bed row <= "
          f"{p['max_per_row']}")
    print("S = min(cols * L, rows * min(zmax, k t)) with cols * rows <= slots; "
          "'binds' names the active limit (len = bed length: the plateau)")
    if res["t_infeasible"]:
        ti = res["t_infeasible"]
        print(f"t {ti[0]:g} .. {ti[-1]:g}: infeasible, thinner than relief + body")
    print("\nbest S per thickness")
    print(HEADER)
    best = res["best"]
    for d in res["best_per_t"]:
        lay = _from_dict(d)
        print(fmt_row(lay, "  <- max S" if abs(lay.side_mm - best["side_mm"]) < 1e-9
                      and lay.thickness_mm == best["thickness_mm"] else ""))
    print("\nPareto frontier, S versus material (step: % material per % side)")
    print(HEADER + "   step")
    front = [_from_dict(d) for d in res["pareto"]]
    for i, lay in enumerate(front):
        step = "" if i == 0 else f"{step_ratio(front[i - 1], lay):6.1f}"
        print(fmt_row(lay) + f"  {step}")
    r = _from_dict(res["recommended"])
    print(f"\nrecommended: t = {r.thickness_mm:g} mm, {r.label} strips of "
          f"{r.strip_len_mm:.1f} x {r.strip_height_mm:.1f} x {r.thickness_mm:g} mm, "
          f"S = {r.side_mm:.0f} mm, {r.mass_kg:.2f} kg, ~{r.hours:.0f} h (crude)")
    print(f"rule: from the cheapest frontier layout, jump to the largest S that "
          f"costs at most {res['max_ratio']:g} % material per % side, repeat until "
          f"no such step exists (S^2 scaling alone costs 2 % per %)")
    print("time = volume / (flow * duty): crude, slice for real numbers; "
          "f1 = 270 t / h^2 Hz for PLA, '!' marks f1 < 100 Hz")
    if "side" in res:
        s = res["side"]
        print(f"\nlayouts for S = {s['side_mm']:g} mm, thinnest (lightest) t per grid, "
              f"{len(s['layouts'])} feasible in total")
        print(HEADER)
        for d in s["lightest_per_grid"]:
            print(fmt_row(_from_dict(d)))


def _from_dict(d: dict) -> StripLayout:
    fields = StripLayout.__dataclass_fields__
    return StripLayout(**{k: d[k] for k in fields})


def cmd_plan(args) -> None:
    b = bed(size_mm=args.bed, zmax_mm=args.zmax, margin_mm=args.margin, gap_mm=args.gap)
    mat = Material(args.walls, args.line_width, args.infill, args.density,
                   args.flow, args.duty, args.area_factor)
    try:
        res = plan(b, mat, args.k, args.t_min, args.t_max, args.t_step, args.max_cols,
                   args.max_rows, args.relief, args.body_min, args.per_row,
                   args.max_ratio, args.side)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    print_plan(res)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {args.json}")


def add_plan_parser(ss) -> None:
    """Register `plan` on the `loopviz plate` sub-subparsers object."""
    q = ss.add_parser("plan", help="strip layouts: best side per thickness, "
                                   "material, Pareto frontier")
    q.add_argument("--bed", type=float, default=None, help="bed side (mm, 340)")
    q.add_argument("--zmax", type=float, default=None, help="max Z (mm, 325)")
    q.add_argument("--margin", type=float, default=None, help="per bed side (mm, 5)")
    q.add_argument("--gap", type=float, default=None, help="between strips (mm, 8)")
    q.add_argument("--k", type=float, default=8.0, help="max slenderness h/t")
    q.add_argument("--t-min", type=float, default=5.0)
    q.add_argument("--t-max", type=float, default=40.0)
    q.add_argument("--t-step", type=float, default=0.5)
    q.add_argument("--relief", type=float, default=4.0, help="peak to peak (mm)")
    q.add_argument("--body-min", type=float, default=3.0, help="under the relief (mm)")
    q.add_argument("--max-cols", type=int, default=12)
    q.add_argument("--max-rows", type=int, default=12)
    q.add_argument("--per-row", type=int, default=1,
                   help="max strips end to end per bed row (1 = hand-over model; "
                        "plate build supports 1)")
    q.add_argument("--walls", type=int, default=3, help="perimeter lines")
    q.add_argument("--line-width", type=float, default=0.45, help="mm")
    q.add_argument("--infill", type=float, default=0.15, help="core fraction")
    q.add_argument("--density", type=float, default=1.24, help="g/cm^3 (PLA)")
    q.add_argument("--flow", type=float, default=10.0, help="mm^3/s sustained")
    q.add_argument("--duty", type=float, default=0.6, help="extruding fraction")
    q.add_argument("--area-factor", type=float, default=1.4,
                   help="relief face area / footprint (1.4 measured on the 660 mm plate)")
    q.add_argument("--max-ratio", type=float, default=5.0,
                   help="recommendation: stop before a frontier step costing more "
                        "than this % material per % side")
    q.add_argument("--side", type=float, default=None,
                   help="also list layouts for this fixed plate side (mm)")
    q.add_argument("--json", help="write everything here")
    q.set_defaults(fn=cmd_plan)


def main(argv=None) -> None:
    """Standalone entry; `plan` may be omitted: python -m loopviz.print3d.strippack --bed 340."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["plan", *argv]
    ap = argparse.ArgumentParser(prog="python -m loopviz.print3d.strippack")
    add_plan_parser(ap.add_subparsers(dest="cmd", required=True))
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
