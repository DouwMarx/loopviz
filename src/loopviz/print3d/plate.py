"""`loopviz plate`: the song operator as one big smooth relief, printed as
on-edge strips in a single job.

  loopviz plate build --audio song.wav --start 324.68 --end 338.18
      --pitch 2 --side 660 --cols 2 --rows 9 --thickness 10 --relief auto
  loopviz plate build --loop loops/armed_man.json ...   (spec instead of audio flags)
  loopviz plate build ... --layout auto        (strippack picks cols/rows/t)
  loopviz plate demo --out runs/plate/demo      (tiny plates for the eye)
  loopviz plate plan ...                        (strippack, layouts table)
  loopviz plate export --build runs/plate/<name> --out export/<name>  (package)

Plate frame: x right (matrix column), y up (matrix row, row 0 at the
bottom), z toward the viewer; back face z = 0, relief in [t - relief, t].
Strips are labelled chessboard style, column letter + row number, A1 at
the bottom left. On edge: (x, y, z) -> (x, -z, y) then a translation, so
the relief faces the printer front (-Y), the back is a vertical wall and
the strip's lowest plate edge sits on the bed. Strip k (label order A1,
A2, .., B1, ..) is the k-th slot from the front.

Outputs in --out: plate.stl (uncut), bed.stl (print as is),
strips/<label>.stl, layout.json, metrics.json, assembly.md, preview_*.png,
A.npy, surface_Z.npy; with --write-cut also plate_cut.stl (strips in the
plate frame, labels engraved) and with --plate-strips plate_strips/.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from math import ceil, floor, sqrt
from pathlib import Path

import numpy as np

from ..loopspec import add_loop_args
from . import strippack
from .export import add_export_parser
from .relief import read_stl, write_stl
from .reliefviz import mesh_views
from .slab import (
    DENSITY_G_CM3,
    ON_EDGE,
    concat,
    engrave_text,
    face_normals,
    homogeneous,
    manifold_check,
    mesh_metrics,
    overhang_angles,
    slab_mesh,
    slab_mesh_tagged,
    slab_volume,
    stl_vertices,
    transform,
)
from .strippack import Bed, add_plan_parser
from .surface import MAPPINGS, METHODS, RULES, Surface, fit_relief, slope_stats, surface

# -- parameters -------------------------------------------------------------------

@dataclass(frozen=True)
class PlateParams:
    pitch_mm: float
    cols: int
    rows: int
    thickness_mm: float
    relief_mm: float
    subdiv: int = 3
    interp: str = "cubic"
    clip_pct: float = 99.5
    mapping: str = "clip"              # A -> [-1, 1]: clip / tanh / none (surface.cell_values)
    sigma_cells: float = 0.0
    orientation: str = "edge"          # "edge" (one job) or "flat" (one job per tile)
    labels: bool = True
    label_mm: float = 6.0              # cap height
    engrave_mm: float = 0.6
    label_margin_mm: float = 4.0       # label box to strip edge
    k_slender: float = 8.0             # max strip height / thickness on edge
    body_min_mm: float = 3.0           # solid under the relief trough
    fit_range: bool = True             # scale relief so the printed range fits
                                       # thickness - body_min (False: explicit relief,
                                       # only warn)

    @property
    def cap_mm(self) -> float:
        return self.thickness_mm - self.body_min_mm

    def __post_init__(self):
        if self.interp not in METHODS:
            raise ValueError(f"interp must be one of {METHODS}")
        if self.mapping not in MAPPINGS:
            raise ValueError(f"mapping must be one of {MAPPINGS}")
        if self.orientation not in ("edge", "flat"):
            raise ValueError("orientation must be 'edge' or 'flat'")
        if not self.fit_range and self.relief_mm > self.cap_mm + 1e-9:
            raise ValueError(f"thickness {self.thickness_mm} - relief {self.relief_mm} "
                             f"leaves less than {self.body_min_mm} mm of body")
        if self.cols < 1 or self.rows < 1:
            raise ValueError("cols and rows must be >= 1")


@dataclass
class Strip:
    label: str
    col: int                 # 0-based, x
    row: int                 # 0-based, y (0 = bottom)
    j0: int                  # node index range in x, inclusive
    j1: int
    i0: int                  # node index range in y, inclusive
    i1: int
    V: np.ndarray            # plate frame, labels engraved
    F: np.ndarray
    tag: np.ndarray          # 0 top (relief), 1 bottom (back), 2 sides
    T: np.ndarray            # 4x4 plate -> bed
    bottom: np.ndarray | float = 0.0

    @property
    def bed_V(self) -> np.ndarray:
        return transform(self.V, self.T)


def col_letter(c: int) -> str:
    s = ""
    c += 1
    while c:
        c, r = divmod(c - 1, 26)
        s = chr(65 + r) + s
    return s


def strip_label(col: int, row: int) -> str:
    return f"{col_letter(col)}{row + 1}"


def cut_indices(n_nodes: int, parts: int) -> list[tuple[int, int]]:
    """Inclusive node ranges of `parts` pieces; cuts snap to the nearest
    node so neighbours share the seam nodes exactly."""
    cuts = [round(c * (n_nodes - 1) / parts) for c in range(parts + 1)]
    if len(set(cuts)) != parts + 1:
        raise ValueError(f"{parts} parts need at least {parts + 1} nodes, have {n_nodes}")
    return [(cuts[c], cuts[c + 1]) for c in range(parts)]


def edge_transform(k: int, x0: float, y0: float, thickness_mm: float, bed: Bed) -> np.ndarray:
    """Plate -> bed for the k-th slot from the front: X in [margin,
    margin + l], Y in [y_k - t, y_k], Z from 0, relief facing -Y."""
    yk = bed.margin_mm + thickness_mm + k * (thickness_mm + bed.gap_mm)
    return homogeneous(ON_EDGE, (bed.margin_mm - x0, yk, -y0))


def flat_transform(x0: float, y0: float, bed: Bed) -> np.ndarray:
    """Plate -> bed for a tile printed flat on its back, one job per tile."""
    return homogeneous(np.eye(3), (bed.margin_mm - x0, bed.margin_mm - y0, 0.0))


# -- cutting -----------------------------------------------------------------------

def cut_plate(surf: Surface, P: PlateParams, bed: Bed) -> list[Strip]:
    """Strips in label order (A1, A2, .., B1, ..), meshed in the plate
    frame with labels engraved, each with its bed transform."""
    ny, nx = surf.Z.shape
    xr = cut_indices(nx, P.cols)
    yr = cut_indices(ny, P.rows)
    strips = []
    for c, (j0, j1) in enumerate(xr):
        for r, (i0, i1) in enumerate(yr):
            k = len(strips)
            x, y = surf.x[j0:j1 + 1], surf.y[i0:i1 + 1]
            top = surf.Z[i0:i1 + 1, j0:j1 + 1]
            label = strip_label(c, r)
            bottom: np.ndarray | float = 0.0
            if P.labels:
                anchor = (x[0] + P.label_margin_mm, y[-1] - P.label_margin_mm)
                try:
                    bottom = engrave_text(0.0, x, y, label, label_height(P, surf),
                                          P.engrave_mm, anchor, mirror_x=True, glyph_up=True)
                except ValueError as e:
                    raise ValueError(f"{e}; use a smaller label_mm or label_margin_mm, "
                                     f"or labels=False") from None
            V, F, tag = slab_mesh_tagged(top, bottom, x, y)
            T = (edge_transform(k, x[0], y[0], P.thickness_mm, bed) if P.orientation == "edge"
                 else flat_transform(x[0], y[0], bed))
            strips.append(Strip(label, c, r, j0, j1, i0, i1, V, F, tag, T, bottom))
    return strips


LABEL_MIN_NODES = 9       # cap height in nodes below which the font is unreadable


def label_height(P: PlateParams, surf: Surface) -> float:
    """Label cap height: label_mm, floored at LABEL_MIN_NODES fine nodes
    (one raster pixel is one node, so a coarse --subdiv coarsens labels)."""
    return max(P.label_mm, LABEL_MIN_NODES * surf.node_mm)


def bbox(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return V.min(axis=0), V.max(axis=0)


def check_layout(strips: list[Strip], P: PlateParams, bed: Bed) -> None:
    """Fail loudly if anything is off the bed or inside the margin, over
    zmax, too slender, closer than gap_mm to a neighbour, or if the brim
    does not fit the margin."""
    L = bed.usable_mm
    if bed.brim_mm > bed.margin_mm + 1e-9:
        raise RuntimeError(f"brim {bed.brim_mm} mm wider than the margin {bed.margin_mm} mm")
    boxes = []
    for s in strips:
        lo, hi = bbox(s.V)
        length, height = hi[0] - lo[0], hi[1] - lo[1]
        if length > L + 1e-9:
            raise RuntimeError(f"strip {s.label} is {length:.1f} mm long, bed holds {L:.1f}")
        if P.orientation == "edge":
            if height > bed.zmax_mm + 1e-9:
                raise RuntimeError(f"strip {s.label} is {height:.1f} mm tall, zmax {bed.zmax_mm}")
            if height / P.thickness_mm > P.k_slender + 1e-9:
                raise RuntimeError(f"strip {s.label}: h/t = {height / P.thickness_mm:.1f} "
                                   f"exceeds k = {P.k_slender}")
        elif height > L + 1e-9:
            raise RuntimeError(f"tile {s.label} is {height:.1f} mm tall, bed holds {L:.1f}")
        blo, bhi = bbox(s.bed_V)
        m = bed.margin_mm - 1e-9
        if blo[2] < -1e-9 or (blo[:2] < m).any() or (bhi[:2] > bed.size_mm - m).any() \
                or bhi[2] > bed.zmax_mm + 1e-9:
            raise RuntimeError(f"strip {s.label} leaves the bed or its {bed.margin_mm} mm "
                               f"margin: {blo} .. {bhi}")
        boxes.append((s.label, blo, bhi))
    if P.orientation == "edge":
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                la, alo, ahi = boxes[a]
                lb, blo, bhi = boxes[b]
                gap = max(blo[1] - ahi[1], alo[1] - bhi[1])
                if gap < bed.gap_mm - 1e-9:
                    raise RuntimeError(f"strips {la} and {lb} are {gap:.2f} mm apart on the "
                                       f"bed, gap {bed.gap_mm} mm required")


# -- previews ------------------------------------------------------------------------

def _views(V, F, path, title="", views=None, face_colors=None, dpi=130):
    return mesh_views(V, F, path, title=title, dpi=dpi, views=views, face_colors=face_colors)


OVERHANG_COLORS = {"ok": (0.55, 0.75, 0.45), "warn": (0.95, 0.85, 0.30),
                   "bad": (0.85, 0.20, 0.20), "none": (0.82, 0.82, 0.82)}


def overhang_colors(V: np.ndarray, F: np.ndarray, up=(0, 0, 1), warn_deg: float = 30.0,
                    limit_deg: float = 45.0) -> np.ndarray:
    """(nf, 3) face colours: grey for up-facing, vertical or floor faces,
    green under warn_deg, yellow up to limit_deg, red beyond."""
    alpha, _, _ = overhang_angles(V, F, up)
    col = np.tile(OVERHANG_COLORS["none"], (len(F), 1))
    down = ~np.isnan(alpha)
    col[down] = OVERHANG_COLORS["ok"]
    col[down & (alpha > warn_deg)] = OVERHANG_COLORS["warn"]
    col[down & (alpha > limit_deg)] = OVERHANG_COLORS["bad"]
    return col


def preview_faces(V: np.ndarray, F: np.ndarray, ratio: float = 20.0) -> np.ndarray:
    """Faces for matplotlib previews only: without the floor (every face
    lying flat on the lowest z plane: the back fan in the plate frame,
    the bed-contact edge on the bed) and without faces more than `ratio`
    times the median area, which the centroid depth sort draws through
    the relief."""
    n, area = face_normals(V, F)
    zmin = V[:, 2].min()
    floor = (n[:, 2] < -0.999) & (np.abs(V[F][:, :, 2] - zmin) < 1e-6).all(axis=1)
    return F[~floor & (area <= ratio * np.median(area))]


def decimated_bed(surf: Surface, strips: list[Strip], max_faces: int = 40000,
                  to_bed: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """All strips (flat backs, no labels) on a coarser node grid for the
    matplotlib previews; cut nodes are kept so seams stay exact."""
    stride = max(1, ceil(sqrt(2 * surf.Z.size / max_faces)))
    parts = []
    for s in strips:
        jj = np.unique(np.r_[np.arange(s.j0, s.j1, stride), s.j1])
        ii = np.unique(np.r_[np.arange(s.i0, s.i1, stride), s.i1])
        V, F, tag = slab_mesh_tagged(surf.Z[np.ix_(ii, jj)], 0.0, surf.x[jj], surf.y[ii])
        Vp = transform(V, s.T) if to_bed else V
        parts.append((Vp, preview_faces(Vp, F[tag != 1])))
    return concat(parts)


def overhang_window(surf: Surface, strips: list[Strip], width_mm: float = 60.0,
                    max_faces: int = 40000) -> tuple[np.ndarray, np.ndarray, str]:
    """Full-resolution piece of the worst strip around its steepest
    relief face (full strip height, `width_mm` wide), in the bed frame.
    Decimation would flatten the slopes, so the window is not decimated
    unless it still exceeds max_faces."""
    worst, best_a, best_f = None, -1.0, 0
    for s in strips:
        top = s.tag == 0
        a, _, _ = overhang_angles(s.bed_V, s.F[top], skip_floor=False)
        if np.nanmax(np.nan_to_num(a, nan=-1.0)) > best_a:
            worst, best_a, best_f = s, float(np.nanmax(a)), int(np.nanargmax(a))
    xc = worst.V[worst.F[worst.tag == 0][best_f]][:, 0].mean()
    half = width_mm / 2
    jj = np.flatnonzero((surf.x >= xc - half) & (surf.x <= xc + half))
    jj = jj[(jj >= worst.j0) & (jj <= worst.j1)]
    ii = np.arange(worst.i0, worst.i1 + 1)
    stride = max(1, ceil(sqrt(2 * ii.size * jj.size / max_faces)))
    jj = np.unique(np.r_[jj[::stride], jj[-1]])
    ii = np.unique(np.r_[ii[::stride], ii[-1]])
    V, F, tag = slab_mesh_tagged(surf.Z[np.ix_(ii, jj)], 0.0, surf.x[jj], surf.y[ii])
    note = (f"strip {worst.label}, x {surf.x[jj[0]]:.0f}..{surf.x[jj[-1]]:.0f} mm, "
            f"steepest {best_a:.0f} deg")
    Vb = transform(V, worst.T)
    note += f" (bed X {Vb[:, 0].min():.0f}..{Vb[:, 0].max():.0f})"
    return Vb, preview_faces(Vb, F[tag != 1]), note


def preview_plate(surf: Surface, strips: list[Strip], path: Path, title: str = "") -> Path:
    """Hillshade of the fine surface, seam lines and strip labels."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    Z = surf.Z
    step = max(1, ceil(max(Z.shape) / 1600))
    Zs = Z[::step, ::step]
    d = surf.node_mm * step
    rgb = LightSource(azdeg=315, altdeg=45).shade(Zs, cmap=plt.get_cmap("gray"),
                                                  vert_exag=1.0, dx=d, dy=d, blend_mode="soft")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, origin="lower", extent=(surf.x[0], surf.x[-1], surf.y[0], surf.y[-1]),
              interpolation="nearest")
    for s in strips:
        ax.plot([surf.x[s.j0], surf.x[s.j1], surf.x[s.j1], surf.x[s.j0], surf.x[s.j0]],
                [surf.y[s.i0], surf.y[s.i0], surf.y[s.i1], surf.y[s.i1], surf.y[s.i0]],
                color="#d62728", lw=0.8)
        ax.text(0.5 * (surf.x[s.j0] + surf.x[s.j1]), 0.5 * (surf.y[s.i0] + surf.y[s.i1]),
                s.label, color="#d62728", ha="center", va="center", fontsize=11,
                fontweight="bold")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    ax.set_title(title or "plate, front view, seams and labels")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def preview_bed(surf: Surface, strips: list[Strip], P: PlateParams, bed: Bed,
                path: Path) -> Path:
    """Top-down bed drawing (left) and an oblique view of a decimated bed
    mesh (right)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(Rectangle((0, 0), bed.size_mm, bed.size_mm, fill=False, lw=1.5))
    m = bed.margin_mm
    ax.add_patch(Rectangle((m, m), bed.size_mm - 2 * m, bed.size_mm - 2 * m, fill=False,
                           ls="--", lw=0.8, color="gray"))
    for s in strips:
        lo, hi = bbox(s.bed_V)
        ax.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], color="#4c72b0", alpha=0.6))
        ax.text(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), s.label, ha="center",
                va="center", fontsize=8, color="white")
    ax.set_xlim(-5, bed.size_mm + 5)
    ax.set_ylim(-5, bed.size_mm + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("bed X mm")
    ax.set_ylabel("bed Y mm (front at the bottom; relief faces -Y)")
    what = ("one job" if P.orientation == "edge"
            else "flat: one job per tile, all drawn at the same place")
    ax.set_title(f"{len(strips)} strips, t = {P.thickness_mm:g} mm, gap {bed.gap_mm:g}, {what}")
    fig.tight_layout()
    draw = path.with_name(path.stem + "_layout.png")
    fig.savefig(draw, dpi=140)
    plt.close(fig)
    V, F = decimated_bed(surf, strips)
    mesh = _views(V, F, path.with_name(path.stem + "_mesh.png"), title="bed.stl (decimated)",
                  views=[("oblique", 30, -60)], dpi=200)
    return pair(draw, mesh, path)


def pair(left: Path, right: Path, path: Path) -> Path:
    """Two PNGs side by side at equal height."""
    from PIL import Image

    a, b = Image.open(left).convert("RGB"), Image.open(right).convert("RGB")
    h = max(a.height, b.height)
    a = a.resize((round(a.width * h / a.height), h))
    b = b.resize((round(b.width * h / b.height), h))
    out = Image.new("RGB", (a.width + b.width, h), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (a.width, 0))
    out.save(path)
    return path


def preview_overhang(surf: Surface, strips: list[Strip], path: Path,
                     limit_deg: float = 45.0) -> Path:
    """Full-resolution window of the worst strip, faces coloured by
    overhang class (print orientation, up = +Z)."""
    V, F, note = overhang_window(surf, strips)
    return _views(V, F, path, title=f"overhang on the bed, {note}: green < 30, "
                                    f"yellow < {limit_deg:g}, red beyond",
                  views=[("front", 15, -90), ("front low", -20, -70)],
                  face_colors=overhang_colors(V, F, limit_deg=limit_deg))


# -- build -----------------------------------------------------------------------------

def strip_record(s: Strip, P: PlateParams) -> dict:
    lo, hi = bbox(s.V)
    blo, bhi = bbox(s.bed_V)
    Vb = s.bed_V
    whole = mesh_metrics(Vb, s.F)
    relief = mesh_metrics(Vb, s.F[s.tag == 0], skip_floor=False)
    _, area = face_normals(Vb, s.F)
    areas = {name: float(area[s.tag == k].sum()) for k, name in enumerate(("relief", "back", "edges"))}
    return {"label": s.label, "col": s.col, "row": s.row, "areas_mm2": areas,
            "nodes": {"i0": s.i0, "i1": s.i1, "j0": s.j0, "j1": s.j1},
            "plate_bbox": [lo.tolist(), hi.tolist()], "bed_bbox": [blo.tolist(), bhi.tolist()],
            "size_mm": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), P.thickness_mm],
            "transform": s.T.tolist(), "faces": len(s.F),
            "volume_mm3": whole["volume_mm3"], "mass_g": whole["mass_g"],
            "overhang": whole["overhang"], "overhang_relief_face": relief["overhang"]}


def _hist_sum(records: list[dict], key: str) -> dict:
    edges = records[0][key]["edges_deg"]
    area = np.sum([r[key]["area_mm2"] for r in records], axis=0)
    over = sum(r[key]["area_over_45_mm2"] for r in records)
    return {"edges_deg": edges, "area_mm2": area.tolist(), "area_over_45_mm2": over,
            "max_deg": max(r[key]["max_deg"] for r in records)}


def build_plate(A: np.ndarray, P: PlateParams, bed: Bed, out: Path,
                meta: dict | None = None, render: bool = True,
                verbose: bool = True, plate_strips: bool = False,
                write_cut: bool = False) -> dict:
    """Surface, strips, checks, every output file. Returns the metrics.
    plate_strips also writes plate_strips/<label>.stl (plate frame, with
    labels) and write_cut the single plate_cut.stl, both read by the
    visual tests; off by default, they would double the output size."""
    out = Path(out)
    (out / "strips").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    say = print if verbose else (lambda *a, **k: None)
    surf = surface(A, P.pitch_mm, P.thickness_mm, P.relief_mm, P.subdiv, P.interp,
                   P.clip_pct, P.sigma_cells, max_range_mm=P.cap_mm if P.fit_range else None,
                   mapping=P.mapping)
    ny, nx = surf.Z.shape
    slopes = slope_stats(surf)
    say(f"surface {nx} x {ny} nodes ({surf.node_mm:.3f} mm), relief {surf.relief_mm:.2f} mm "
        f"(requested {P.relief_mm:.2f}, printed range {surf.printed_range_mm:.2f}, body "
        f"{surf.Z.min():.2f}), |dh/dy| q99 "
        f"{slopes['abs_dy']['q99']:.2f} max {slopes['abs_dy']['max']:.2f}, facet over 45: "
        f"{100 * slopes['frac_facet_over_limit']:.2f} % (max {slopes['max_facet_deg']:.0f} "
        f"deg), {time.perf_counter() - t0:.1f} s")
    strips = cut_plate(surf, P, bed)
    check_layout(strips, P, bed)
    for s in strips:
        chk = manifold_check(s.V, s.F)
        vol = slab_volume(surf.Z[s.i0:s.i1 + 1, s.j0:s.j1 + 1], s.bottom,
                          surf.x[s.j0:s.j1 + 1], surf.y[s.i0:s.i1 + 1])
        if not chk["watertight"]:
            raise RuntimeError(f"strip {s.label} is not a closed 2-manifold: {chk}")
        if abs(chk["volume"] - vol) > 1e-9 * vol:
            raise RuntimeError(f"strip {s.label} volume {chk['volume']} != {vol}")
        # in-memory reassembly: inverse bed transform returns the plate vertices
        back = transform(s.bed_V, np.linalg.inv(s.T))
        if np.abs(back - s.V).max() > 1e-6:
            raise RuntimeError(f"strip {s.label}: inverse transform mismatch")
    say(f"{len(strips)} strips meshed and checked, {time.perf_counter() - t0:.1f} s")

    Vp, Fp = slab_mesh(surf.Z, 0.0, surf.x, surf.y)
    files = {"plate.stl": write_stl(Vp, Fp, out / "plate.stl", name=f"{out.name} plate")}
    del Vp, Fp
    if write_cut:
        Vc, Fc = concat([(s.V, s.F) for s in strips])
        files["plate_cut.stl"] = write_stl(Vc, Fc, out / "plate_cut.stl", name=f"{out.name} cut")
        del Vc, Fc
    for s in strips:
        files[f"strips/{s.label}.stl"] = write_stl(s.bed_V, s.F, out / "strips" / f"{s.label}.stl",
                                                   name=f"{out.name} {s.label}")
    if plate_strips:
        (out / "plate_strips").mkdir(exist_ok=True)
        for s in strips:
            write_stl(s.V, s.F, out / "plate_strips" / f"{s.label}.stl",
                      name=f"{out.name} {s.label} plate frame")
    if P.orientation == "edge":
        Vb, Fb = concat([(s.bed_V, s.F) for s in strips])
        files["bed.stl"] = write_stl(Vb, Fb, out / "bed.stl", name=f"{out.name} bed")
        del Vb, Fb
    np.save(out / "A.npy", A)
    np.save(out / "surface_Z.npy", surf.Z)
    say(f"STLs written, {time.perf_counter() - t0:.1f} s")

    records = [strip_record(s, P) for s in strips]
    total_vol = sum(r["volume_mm3"] for r in records)
    n_strips = len(strips)
    yk_max = max(r["bed_bbox"][1][1] for r in records)
    params = {**asdict(P), "relief_mm": surf.relief_mm, "relief_requested_mm": P.relief_mm,
              "label_mm": label_height(P, surf) if P.labels else None}
    layout = {
        "bed": asdict(bed), "params": params, "side_mm": float(surf.x[-1]),
        "nodes": [nx, ny], "surface_z_range": [float(surf.Z.min()), float(surf.Z.max())],
        "strips": records,
        "bed_usage": {"rows_used": n_strips if P.orientation == "edge" else 1,
                      "y_extent_mm": yk_max, "frac_of_usable_y": yk_max / bed.size_mm},
    }
    (out / "layout.json").write_text(json.dumps(layout, indent=1))
    metrics = {
        "n": int(A.shape[0]), "side_mm": float(surf.x[-1]), "nodes": [nx, ny],
        "faces_total": int(sum(r["faces"] for r in records)),
        "volume_mm3": total_vol, "mass_g": total_vol * DENSITY_G_CM3 / 1000.0,
        "relief_mm": surf.relief_mm, "thickness_mm": P.thickness_mm,
        "mapping": P.mapping, "clip_pct": P.clip_pct, "interp": P.interp, "subdiv": P.subdiv,
        "printed_range_mm": [float(surf.Z.min()), float(surf.Z.max()), surf.printed_range_mm],
        "body_min_actual_mm": float(surf.Z.min()),
        "slopes": slopes,
        "overhang_bed": _hist_sum(records, "overhang"),
        "overhang_relief_face": _hist_sum(records, "overhang_relief_face"),
        "strips": n_strips, "orientation": P.orientation,
        "files_mb": {**{k: round(v.stat().st_size / 1e6, 1) for k, v in files.items()
                        if not k.startswith("strips/")},
                     "strips/": round(sum(v.stat().st_size for k, v in files.items()
                                          if k.startswith("strips/")) / 1e6, 1)},
        "build_seconds": None,
    }
    area = metrics["overhang_bed"]
    tot_area = sum(sum(r["overhang"]["area_mm2"]) for r in records)
    metrics["overhang_bed"]["frac_over_45_of_downfacing"] = (
        area["area_over_45_mm2"] / max(tot_area, 1e-30))
    rel = metrics["overhang_relief_face"]
    rel_area = float(np.sum([mesh_metrics(s.bed_V, s.F[s.tag == 0], skip_floor=False)["area_mm2"]
                             for s in strips]))
    rel["relief_area_mm2"] = rel_area
    rel["frac_over_45_of_relief"] = rel["area_over_45_mm2"] / max(rel_area, 1e-30)
    tot = {k: sum(r["areas_mm2"][k] for r in records) for k in ("relief", "back", "edges")}
    est = strippack.printed_material(tot["relief"], tot["back"], tot["edges"], total_vol,
                                     strippack.PLA)
    metrics["material_estimate"] = {**est, "areas_mm2": tot,
                                    "note": "walls 3 x 0.45 mm under every face + 15 % infill "
                                            "of the rest; time is crude"}
    if meta:
        metrics.update(meta)
    metrics.setdefault("relief_choice", {}).update({
        "requested_mm": P.relief_mm, "nominal_mm": surf.relief_mm, "chosen_mm": surf.relief_mm,
        "printed_range_mm": surf.printed_range_mm, "scale": surf.scale,
        "mid_depth_offset_mm": surf.offset_mm, "cap_mm": P.cap_mm,
        "scaled_to_fit": P.fit_range and surf.scale < 1.0,
        "thickness_for_requested_mm": (P.relief_mm * surf.printed_range_mm / surf.relief_mm
                                       + P.body_min_mm)})

    if render:
        files["preview_plate.png"] = preview_plate(surf, strips, out / "preview_plate.png",
                                                   f"{out.name}: {nx} x {ny} nodes, "
                                                   f"relief {surf.relief_mm:.2f} mm")
        files["preview_bed.png"] = preview_bed(surf, strips, P, bed, out / "preview_bed.png")
        files["preview_overhang.png"] = preview_overhang(surf, strips,
                                                         out / "preview_overhang.png")
        say(f"previews written, {time.perf_counter() - t0:.1f} s")
    metrics["build_seconds"] = time.perf_counter() - t0
    if surf.scale < 1.0:
        say(f"relief capped by thickness, not by the overhang rule: scaled by {surf.scale:.3f} "
            f"to {surf.relief_mm:.2f} mm so the printed range ({P.interp} overshoot included) "
            f"fits {P.cap_mm:g} mm; the requested {P.relief_mm:.2f} mm needs thickness "
            f"{metrics['relief_choice']['thickness_for_requested_mm']:.1f} mm")
    if surf.Z.min() < P.body_min_mm - 1e-9:
        say(f"WARNING: printed range {surf.printed_range_mm:.2f} mm exceeds thickness - "
            f"body_min = {P.cap_mm:g}: body under the deepest trough is {surf.Z.min():.2f} mm")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "assembly.md").write_text(assembly_md(P, bed, records, metrics))
    ver = verify_build(out)
    metrics["verify"] = ver
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    say(f"reassembly check: max deviation {ver['max_dev_mm']:.2e} mm; "
        f"done in {metrics['build_seconds']:.1f} s")
    return metrics


def label_grid(records: list[dict], cols: int, rows: int) -> str:
    """Labels as the plate hangs: top row first, columns left to right."""
    grid = {(r["col"], r["row"]): r["label"] for r in records}
    lines = []
    for r in range(rows - 1, -1, -1):
        lines.append("  ".join(f"{grid[(c, r)]:>4}" for c in range(cols)))
    return "\n".join(lines)


def assembly_md(P: PlateParams, bed: Bed, records: list[dict], metrics: dict) -> str:
    side = metrics["side_mm"]
    ov = metrics["overhang_relief_face"]
    mx = ov["max_deg"]
    seams_x = sorted({r["plate_bbox"][1][0] for r in records if r["col"] < P.cols - 1})
    seams_y = sorted({r["plate_bbox"][1][1] for r in records if r["row"] < P.rows - 1})
    sizes = {tuple(round(v, 1) for v in r["size_mm"]) for r in records}
    worst = max(records, key=lambda r: r["overhang_relief_face"]["max_deg"])
    supports = ("none needed" if mx <= 45.0 else
                f"the relief exceeds 45 deg (max {mx:.0f} deg, worst strip {worst['label']}); "
                f"either accept some surface roughness there, lower --relief, or print "
                f"with supports on the relief face")
    note = ""
    if metrics["files_mb"].get("plate_cut.stl", 0) > 150:
        note = ("\nFiles are large; `--subdiv 2` halves the node count per axis (4x fewer "
                "faces) if the slicer struggles.\n")
    return f"""# Assembly: {P.cols} x {P.rows} strips, plate {side:.0f} x {side:.0f} mm

## Hung on the wall (viewer's side, relief toward you)

```
{label_grid(records, P.cols, P.rows)}
```

Columns A, B, .. run left to right; row numbers increase upward, so A1 is
the bottom-left strip and {records[-1]['label']} the top-right one. Each
strip's label is engraved {P.engrave_mm:g} mm deep into the back near the
strip's top-left corner as seen from the front, so it appears top-right
when you look at the back; the triangle points up (plate +y). Hold the
strip so the triangle points up and the text reads normally: you are then
looking at the back of a correctly oriented strip.

Strip sizes (length x height x thickness, mm): {', '.join(' x '.join(f'{v:g}' for v in s) for s in sorted(sizes))}.
Seams (plate frame): x = {', '.join(f'{v:.1f}' for v in seams_x) or 'none'}; y = {', '.join(f'{v:.1f}' for v in seams_y) or 'none'}.
Neighbouring strips share their seam nodes exactly, so edges butt together
without a gap; cut lines snap to the fine grid so strips may differ by one
fine cell ({P.pitch_mm / P.subdiv:.3f} mm).

## Print settings ({P.orientation}, {len(records)} strips, {'one job' if P.orientation == 'edge' else 'one job per tile'})

- `bed.stl` is placed and oriented; print it as is (or the single
  `strips/<label>.stl` files, which carry the same placement). Bed
  {bed.size_mm:g} mm, margin {bed.margin_mm:g} mm, gap {bed.gap_mm:g} mm,
  Z max {bed.zmax_mm:g} mm. The relief faces the printer front (-Y).
- Layer height 0.15 to 0.2 mm (the relief is formed by the XY path, not
  by layers, so layer height sets only the roughness of the top edge).
- Walls 3, infill 15 % (strips are ordinary solids; 100 % is not needed).
  Brim 5 mm around every strip for adhesion of the tall thin walls.
- Seam position: rear (the back face, which is also the labelled face).
- Supports: {supports}. Overhang of the relief face on the bed:
  {ov['area_over_45_mm2'] / 100:.0f} cm^2 over 45 deg of
  {ov['relief_area_mm2'] / 100:.0f} cm^2 ({100 * ov['frac_over_45_of_relief']:.2f} %);
  whole bed including the engraved labels, whose recess edges add small
  overhangs of their own: {metrics['overhang_bed']['area_over_45_mm2'] / 100:.0f} cm^2.
- Labels are rasterised one pixel per fine node ({P.pitch_mm / P.subdiv:.3f} mm), cap
  height floored at {LABEL_MIN_NODES} nodes; a coarser --subdiv makes them coarser.
- Adjacent {bed.brim_mm:g} mm brims merge across the {bed.gap_mm:g} mm gap into one sheet;
  cut it between the strips after printing.
- `plate.stl` and `plate_cut.stl` are reference geometry, not print files:
  plate.stl is the uncut {side:.0f} mm plate and plate_cut.stl becomes
  non-manifold once a slicer merges the touching strips' vertices.
- Elephant's foot: every strip's bottom edge (its lowest plate y, a seam
  for all but row 1) flares by 0.1 to 0.3 mm in the first layers. Enable
  the slicer's elephant foot compensation (0.2 mm) or sand the bottom edge
  flat before assembly; otherwise the seams on those edges open up.
- Slenderness h/t = {records[0]['size_mm'][1] / P.thickness_mm:.1f} (limit {P.k_slender:g}).
  Print slowly on the outer wall (40 mm/s) to keep the tops from ringing.
- Mass (solid) {metrics['mass_g'] / 1000:.2f} kg; with walls 3 / infill 15 % about
  {metrics.get('material_estimate', {}).get('mass_kg', float('nan')):.2f} kg,
  crude time {metrics.get('material_estimate', {}).get('hours', float('nan')):.0f} h.
{note}"""


# -- verification -----------------------------------------------------------------------

def verify_build(out_dir: Path, tol: float = 1e-3) -> dict:
    """Read every strips/<label>.stl, undo its bed transform and compare the
    relief vertices (z above the back, i.e. the shared fine-grid nodes;
    backs differ by the fan and the engraving) with plate.stl, both ways
    by nearest neighbour. STL is float32, so tol is 1e-3 mm, not the 1e-6
    of the in-memory check. Also re-checks the bed boxes from layout.json.
    Raises on failure."""
    from scipy.spatial import cKDTree

    out_dir = Path(out_dir)
    layout = json.loads((out_dir / "layout.json").read_text())
    zthr = 0.5 * layout["surface_z_range"][0]
    if zthr <= layout["params"]["engrave_mm"]:
        raise RuntimeError("relief trough too close to the engraving to tell them apart")
    plate = stl_vertices(out_dir / "plate.stl", tol)
    plate = plate[plate[:, 2] > zthr]
    parts = []
    for r in layout["strips"]:
        Vb = stl_vertices(out_dir / "strips" / f"{r['label']}.stl", tol)
        lo, hi = np.asarray(r["bed_bbox"])
        if np.abs(Vb.min(axis=0) - lo).max() > 1e-2 or np.abs(Vb.max(axis=0) - hi).max() > 1e-2:
            raise RuntimeError(f"{r['label']}.stl bbox differs from layout.json")
        Vp = transform(Vb, np.linalg.inv(np.asarray(r["transform"])))
        parts.append(Vp[Vp[:, 2] > zthr])
    back = np.concatenate(parts)
    d1, _ = cKDTree(plate).query(back)
    d2, _ = cKDTree(back).query(plate)
    dev = float(max(d1.max(), d2.max()))
    if dev > tol:
        raise RuntimeError(f"reassembled strips deviate from plate.stl by {dev:.2e} mm")
    bed = layout["bed"]
    boxes = [np.asarray(r["bed_bbox"]) for r in layout["strips"]]
    for b in boxes:
        if (b[0] < -1e-6).any() or b[1][0] > bed["size_mm"] + 1e-6 \
                or b[1][1] > bed["size_mm"] + 1e-6 or b[1][2] > bed["zmax_mm"] + 1e-6:
            raise RuntimeError(f"a strip leaves the bed: {b}")
    if layout["params"]["orientation"] == "edge":
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                if (boxes[a][0][:2] < boxes[b][1][:2] - 1e-6).all() and \
                        (boxes[b][0][:2] < boxes[a][1][:2] - 1e-6).all():
                    raise RuntimeError("two strips overlap on the bed")
    return {"max_dev_mm": dev, "tol_mm": tol, "strips": len(parts),
            "plate_vertices": len(plate), "strip_vertices": len(back)}


# -- demo -----------------------------------------------------------------------------------

def f_matrix(rows: int = 18, cols: int = 24) -> np.ndarray:
    """-1 background with a +1 letter F (asymmetric under rotation and
    mirroring), row 0 at the bottom."""
    A = -np.ones((rows, cols))
    A[2:16, 6:9] = 1.0          # stem
    A[13:16, 6:17] = 1.0        # top bar
    A[8:10, 6:14] = 1.0         # middle bar
    return A


def spike_matrix(n: int = 6) -> np.ndarray:
    A = -np.ones((n, n))
    A[n // 2, n // 2] = 1.0
    return A


def ridge_matrix(rows: int = 12, cols: int = 30, sigma_cells: float = 1.5) -> np.ndarray:
    i = np.arange(rows)
    v = 2.0 * np.exp(-((i - (rows - 1) / 2) ** 2) / (2 * sigma_cells ** 2)) - 1.0
    return np.repeat(v[:, None], cols, axis=1)


R_BACK = np.diag([-1.0, 1.0, -1.0])     # look at the back: rotate 180 deg about y


def render_back(V: np.ndarray, F: np.ndarray, path: Path, title: str) -> Path:
    """The back face as seen from behind (camera at -z looking +z, y up)."""
    hi = V.max(axis=0)
    Vb = transform(V, R_BACK, (hi[0], 0.0, hi[2]))
    return _views(Vb, F, path, title=title, views=[("back (from behind)", 90, -90)])


def _files(d: Path) -> dict[str, Path]:
    f = {k: d / k for k in ("plate.stl", "plate_cut.stl", "bed.stl", "layout.json",
                            "metrics.json", "assembly.md", "strips", "plate_strips")}
    f["preview_plate_png"] = d / "preview_plate.png"
    f["preview_bed_png"] = d / "preview_bed.png"
    f["preview_overhang_png"] = d / "preview_overhang.png"
    return f


def demo_cases(out: Path, render: bool = True) -> dict[str, dict[str, Path]]:
    """Small hand-made plates for the eye and the visual tests, pitch 2,
    subdiv 3: F (24 x 18 cells, raised F, uncut), F_cut (the same cut
    2 x 3 on edge with labels), spike_nearest / spike_cubic (6 x 6 single
    spike), ridge_strip (one strip on edge with a Gaussian ridge).
    Returns the files per case; every case has plate.stl, plate_cut.stl,
    bed.stl, layout.json, strips/ (print orientation) and plate_strips/
    (plate frame, labels engraved)."""
    out = Path(out)
    bed = Bed()
    cases: dict[str, dict[str, Path]] = {}
    front = [("front", 90, -90), ("oblique", 35, -60)]

    d = out / "F"
    P = PlateParams(2.0, 1, 1, 6.0, 3.0, labels=True, label_mm=6.0)
    build_plate(f_matrix(), P, bed, d, render=render, verbose=False, plate_strips=True,
                write_cut=True)
    files = _files(d)
    if render:
        V, F = read_stl(d / "plate.stl")
        files["front_png"] = _views(V, preview_faces(V, F), d / "front.png", "F plate.stl",
                                    views=front)
    cases["F"] = files

    d = out / "F_cut"
    P = PlateParams(2.0, 2, 3, 6.0, 3.0, labels=True, label_mm=6.0, label_margin_mm=2.5)
    build_plate(f_matrix(), P, bed, d, render=render, verbose=False, plate_strips=True,
                write_cut=True)
    files = _files(d)
    if render:
        V, F = read_stl(d / "plate.stl")
        files["front_png"] = _views(V, preview_faces(V, F), d / "front.png", "F_cut plate.stl",
                                    views=front)
        layout = json.loads((d / "layout.json").read_text())
        parts = []
        for r in layout["strips"]:
            Vs, Fs = read_stl(d / "strips" / f"{r['label']}.stl")
            parts.append((transform(Vs, np.linalg.inv(np.asarray(r["transform"]))), Fs))
        Vr, Fr = concat(parts)
        files["reassembled_png"] = _views(Vr, Fr, d / "reassembled.png",
                                          "strips read back and inverse-transformed",
                                          views=front)
        Vb, Fb = read_stl(d / "bed.stl")
        files["bed_png"] = _views(Vb, Fb, d / "bed_views.png", "F_cut bed.stl",
                                  views=[("oblique", 30, -60), ("top-down", 90, -90)])
        for lab in ("B2", "A3"):
            Vs, Fs = read_stl(d / "strips" / f"{lab}.stl")
            r = next(r for r in layout["strips"] if r["label"] == lab)
            Vs = transform(Vs, np.linalg.inv(np.asarray(r["transform"])))
            files[f"back_{lab}_png"] = render_back(Vs, Fs, d / f"back_{lab}.png",
                                                   f"strip {lab}, back face from behind")
    cases["F_cut"] = files

    for interp in ("nearest", "cubic"):
        d = out / f"spike_{interp}"
        P = PlateParams(2.0, 1, 1, 6.0, 3.0, interp=interp, labels=False)
        build_plate(spike_matrix(), P, bed, d, render=render, verbose=False, plate_strips=True,
                write_cut=True)
        files = _files(d)
        if render:
            V, F = read_stl(d / "plate.stl")
            files["views_png"] = _views(V, preview_faces(V, F), d / "views.png",
                                        f"6 x 6 spike, {interp}")
        cases[f"spike_{interp}"] = files

    d = out / "ridge_strip"
    P = PlateParams(2.0, 1, 1, 12.0, 8.0, labels=False)
    build_plate(ridge_matrix(), P, bed, d, render=render, verbose=False, plate_strips=True,
                write_cut=True)
    files = _files(d)
    if render:
        V, F = read_stl(d / "bed.stl")
        F = preview_faces(V, F)
        files["overhang_png"] = _views(V, F, d / "overhang.png",
                                       "ridge strip on edge: red = overhang > 45 deg",
                                       views=[("front", 15, -90), ("front low", -25, -60)],
                                       face_colors=overhang_colors(V, F))
    cases["ridge_strip"] = files
    (out / "cases.json").write_text(json.dumps(
        {c: {k: str(v) for k, v in f.items()} for c, f in cases.items()}, indent=1))
    return cases


# -- CLI --------------------------------------------------------------------------------------

def cmd_build(args) -> None:
    from ..loopspec import resolve_loop
    from ..songmatrix import materialize
    from .relief_cli import max_feasible_rho, probe

    bed = strippack.bed(size_mm=args.bed, zmax_mm=args.zmax, margin_mm=args.margin,
                        gap_mm=args.gap)
    spec = resolve_loop(args)
    out = Path(args.out or f"runs/plate/{spec.name}_p{args.pitch:g}")
    out.mkdir(parents=True, exist_ok=True)
    signal, T, sr = spec.signal(out_wav=out / "loop.wav")
    n = floor(args.side / args.pitch + 1e-9)
    side = n * args.pitch
    if args.rho_max:
        r = max_feasible_rho(signal, T, n, args.drift_tol)
        if r is None:
            raise SystemExit("no stable rho at this n; shorten the loop")
    else:
        r = probe(signal, T, n, args.rho)
    pl = r["plan"]
    print(f"T = {T:.2f} s, n = {n} ({side:g} mm at {args.pitch:g} mm), N = {pl.N}, "
          f"rho = {pl.rho:.3f}, f = {pl.f:.0f} Hz, drift/pass {r['drift']:.1e}"
          + ("  WARNING: unstable operator" if r["drift"] > args.drift_tol else ""))
    A = materialize(r["op"])
    if args.transpose:
        A = A.T                       # matrix column j -> plate y instead of row i
    skw = {"subdiv": args.subdiv, "interp": args.interp, "clip_pct": args.clip, "mapping": args.mapping,
           "sigma_cells": args.sigma}

    fits = {rule: fit_relief(A, args.pitch, args.max_overhang, args.area_quantile,
                             rule=rule, **skw) for rule in ("facet", "y")}
    print(f"relief keeping {100 * args.area_quantile:g} % of the footprint under "
          f"{args.max_overhang:g} deg: facet rule {fits['facet']:.2f} mm, y rule "
          f"{fits['y']:.2f} mm")
    if args.relief == "auto":
        relief = fits[args.rule]
        print(f"relief auto ({args.rule} rule): {relief:.2f} mm")
    else:
        relief = float(args.relief)

    if args.layout == "auto":
        lays = strippack.enumerate_layouts(bed, k_slender=args.k, t_min=args.t_min,
                                           t_max=args.t_max, t_step=args.t_step,
                                           relief_mm=relief, body_min_mm=args.body_min,
                                           side_mm=side)
        if not lays:
            raise SystemExit(f"no strip layout fits a {side:g} mm plate on this bed")
        lay = strippack.best_layout(lays)
        cols, rows, thickness = lay.cols, lay.rows, lay.thickness_mm
        print(f"layout auto: {cols} x {rows} strips, t = {thickness:g} mm "
              f"({lay.strip_len_mm:.1f} x {lay.strip_height_mm:.1f} mm, h/t "
              f"{lay.slenderness:.1f}, {lay.mass_kg:.2f} kg est.)")
    else:
        if None in (args.cols, args.rows, args.thickness):
            raise SystemExit("give --cols --rows --thickness, or --layout auto")
        cols, rows, thickness = args.cols, args.rows, args.thickness
    cap = thickness - args.body_min
    auto = args.relief == "auto"
    if not auto and relief > cap + 1e-9:
        raise SystemExit(f"relief {relief} mm leaves less than {args.body_min} mm body "
                         f"under a {thickness} mm strip")
    why = ("fit, scaled down if the printed range exceeds thickness - body_min" if auto
           else "given")
    P = PlateParams(args.pitch, cols, rows, thickness, relief, orientation=args.orientation,
                    labels=args.labels, label_mm=args.label_mm, engrave_mm=args.engrave_mm,
                    label_margin_mm=args.label_margin, k_slender=args.k,
                    body_min_mm=args.body_min, fit_range=auto, **skw)
    meta = {"audio": str(spec.audio), "start_s": spec.start_s, "end_s": spec.end_s,
            "loop": spec.to_dict(), "T_s": T,
            "source_hz": sr, "N": pl.N, "rho": pl.rho, "f_hz": pl.f, "drift": r["drift"],
            "gram_cond": r["gram_cond"], "transpose": args.transpose,
            "relief_choice": {"mode": args.relief, "rule": args.rule, "why": why,
                              "fit_facet_mm": fits["facet"], "fit_y_mm": fits["y"],
                              "max_overhang_deg": args.max_overhang,
                              "area_quantile": args.area_quantile, "chosen_mm": relief}}
    try:
        m = build_plate(A, P, bed, out, meta=meta, plate_strips=args.plate_strips,
                        write_cut=args.write_cut)
    except (ValueError, RuntimeError) as e:
        raise SystemExit(str(e)) from None
    ov = m["overhang_relief_face"]
    print(f"plate {m['side_mm']:g} mm, relief {m['relief_mm']:.2f} mm (printed range "
          f"{m['printed_range_mm'][2]:.2f}, body {m['body_min_actual_mm']:.2f}), {m['strips']} "
          f"strips of {m['faces_total'] / 1e6:.2f} M faces; solid mass {m['mass_g'] / 1000:.2f} kg; "
          f"relief face over 45 deg: {100 * ov['frac_over_45_of_relief']:.2f} % "
          f"(max {ov['max_deg']:.0f} deg); files MB {m['files_mb']}")
    print(f"wrote {out}/ (bed.stl is print-ready; see assembly.md)")


def cmd_demo(args) -> None:
    cases = demo_cases(Path(args.out), render=not args.no_render)
    for name, files in cases.items():
        print(f"{name}: {len(files)} files in {Path(args.out) / name}")
    print(f"wrote {args.out}/cases.json")


def add_parser(sub) -> None:
    p = sub.add_parser("plate", help="big smooth relief printed as on-edge strips")
    ss = p.add_subparsers(dest="plate_cmd", required=True)

    q = ss.add_parser("build", help="surface, strips, STLs, previews, assembly notes")
    add_loop_args(q)
    q.add_argument("--pitch", type=float, required=True, help="cell side (mm)")
    q.add_argument("--side", type=float, required=True, help="target plate side (mm); "
                                                            "n = floor(side / pitch)")
    q.add_argument("--cols", type=int, default=None)
    q.add_argument("--rows", type=int, default=None)
    q.add_argument("--thickness", type=float, default=None, help="strip thickness (mm)")
    q.add_argument("--layout", default=None, choices=("auto",),
                   help="auto: strippack picks cols, rows, thickness for the side")
    q.add_argument("--t-min", type=float, default=5.0, help="--layout auto thickness grid")
    q.add_argument("--t-max", type=float, default=40.0)
    q.add_argument("--t-step", type=float, default=0.5)
    q.add_argument("--relief", default="auto", help="peak to peak (mm) or 'auto'")
    q.add_argument("--max-overhang", type=float, default=45.0, help="deg, for --relief auto")
    q.add_argument("--rule", default="facet", choices=RULES,
                   help="overhang rule for --relief auto: facet (true on-edge angle, "
                        "hanging facets only) or y (|dh/dy| <= tan)")
    q.add_argument("--transpose", action="store_true",
                   help="matrix column -> plate y instead of row (default row -> y, "
                        "the smooth direction of the operator)")
    q.add_argument("--plate-strips", action="store_true",
                   help="also write plate_strips/<label>.stl in the plate frame")
    q.add_argument("--write-cut", action="store_true",
                   help="also write plate_cut.stl (all strips, plate frame, ~2x bed.stl)")
    q.add_argument("--area-quantile", type=float, default=0.99,
                   help="footprint fraction kept under --max-overhang")
    q.add_argument("--interp", default="cubic", choices=METHODS)
    q.add_argument("--subdiv", type=int, default=3, help="fine nodes per cell side")
    q.add_argument("--clip", type=float, default=99.5, help="percentile of |A| at full relief")
    q.add_argument("--mapping", default="clip", choices=MAPPINGS,
                   help="A to height: clip at --clip (default), tanh soft knee at --clip, "
                        "or none (A / max|A|, spikes take the range)")
    q.add_argument("--sigma", type=float, default=0.0, help="Gaussian pre-smoothing (cells)")
    q.add_argument("--rho", type=float, default=0.95, help="rank fraction N/n")
    q.add_argument("--rho-max", action="store_true", help="bisect the largest stable rho")
    q.add_argument("--drift-tol", type=float, default=1e-6)
    q.add_argument("--bed", type=float, default=None, help="bed side (mm, 340)")
    q.add_argument("--zmax", type=float, default=None, help="max Z (mm, 325)")
    q.add_argument("--margin", type=float, default=None, help="brim margin (mm, 5)")
    q.add_argument("--gap", type=float, default=None, help="between strips (mm, 8)")
    q.add_argument("--k", type=float, default=8.0, help="max slenderness h/t")
    q.add_argument("--body-min", type=float, default=3.0, help="solid under the relief (mm)")
    q.add_argument("--labels", action=_BoolOpt, default=True,
                   help="--labels / --no-labels: engrave strip labels on the back")
    q.add_argument("--label-mm", type=float, default=6.0, help="label cap height (mm)")
    q.add_argument("--engrave-mm", type=float, default=0.6)
    q.add_argument("--label-margin", type=float, default=4.0, help="label to strip edge (mm)")
    q.add_argument("--orientation", default="edge", choices=("edge", "flat"))
    q.add_argument("--out", help="output dir (default runs/plate/<audio>_p<pitch>)")
    q.set_defaults(fn=cmd_build)

    q = ss.add_parser("demo", help="tiny hand-made plates for inspection")
    q.add_argument("--out", default="runs/plate/demo")
    q.add_argument("--no-render", action="store_true", help="skip the PNGs")
    q.set_defaults(fn=cmd_demo)

    add_plan_parser(ss)
    add_export_parser(ss)


class _BoolOpt(argparse.Action):
    """--flag / --no-flag."""

    def __init__(self, option_strings, dest, default=True, help=None, **kw):
        opts = list(option_strings) + [o.replace("--", "--no-", 1) for o in option_strings]
        super().__init__(opts, dest, nargs=0, default=default, help=help)

    def __call__(self, parser, ns, values, option_string=None):
        setattr(ns, self.dest, not option_string.startswith("--no-"))
