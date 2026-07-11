"""Print-ready files from the paper-sweep pool: send straight to the printer.

For each paper size, the best candidate per track under the fitted
preference weights becomes one PDF whose page is EXACTLY the paper
(portrait) with the matrix centered at its true pixel pitch: one matrix
entry = one dot, never resampled. A PNG master with embedded dpi sits
next to each PDF.

The pitch is display geometry only - the operator never depends on it -
so any candidate can also be printed at a coarser pitch on a bigger
sheet, as long as the matrix still fits. `--grid A3 --sheet A1 --pitch
0.5` prints the A3-grid picks at 0.5 mm/entry on A1 pages (n~1147 ->
~574 mm square) into runs/prints/A1-0.5mm/.

Print instructions (also written to each README.md): print at 100% /
"Actual Size" - any fit-to-page scaling destroys the entry-per-dot
exactness the artwork is about.

Run: .venv/bin/python scripts/make_prints.py
       [--weights runs/fitted_weights.json] [--per-paper-top 0]
       [--grid A3 --sheet A1 --pitch 0.5]
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.bt import scorer_from_weights
from playlistviz.pool import candidate_dir, iter_candidate_files, slug

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "prints"
Image.MAX_IMAGE_PIXELS = None
MARGIN_MM = 5.0          # matches the sweep's printable-area assumption

# A-series full sheets, portrait, mm
PAPER_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0), "A2": (420.0, 594.0),
            "A1": (594.0, 841.0), "A0": (841.0, 1189.0)}


def make_print(d: dict, out_dir: Path, rank: int,
               pitch: float | None = None, sheet: str | None = None) -> Path:
    pitch = pitch or d["pitch_mm"]
    sheet = sheet or d["paper"]
    dpi = 25.4 / pitch                       # 0.25 mm -> exactly 101.6
    pw_mm, ph_mm = PAPER_MM[sheet]
    assert d["n"] * pitch + 2 * MARGIN_MM <= min(pw_mm, ph_mm), \
        f"{d['id']}: {d['n']} entries at {pitch} mm do not fit {sheet}"
    page = Image.new("L", (round(pw_mm / pitch), round(ph_mm / pitch)), 255)
    art = Image.open(candidate_dir(ROOT / "runs", d["id"])
                     / "presentation.png").convert("L")
    assert art.size == (d["n"], d["n"]), f"{d['id']}: png size != n"
    page.paste(art, ((page.width - art.width) // 2,
                     (page.height - art.height) // 2))
    base = f"{rank:02d}_{slug(d.get('title') or d['id'])}_{d['id']}"
    pdf = out_dir / f"{base}.pdf"
    page.save(pdf, "PDF", resolution=dpi)
    png = out_dir / f"{base}.png"
    page.save(png, dpi=(dpi, dpi))
    return pdf


def write_readme(path: Path, index: list[str], pitch_desc: str,
                 weights_name: str) -> None:
    path.write_text(f"""\
# prints: send straight to the printer

One PDF per (paper, track): page is exactly the paper size (portrait),
matrix centered, one matrix entry = one {pitch_desc}, never resampled.
The PNG next to each PDF is the same page with dpi metadata, for shops
that prefer raster.

PRINT AT 100% / "ACTUAL SIZE". Any fit-to-page or shrink-to-margins
option resamples the matrix and destroys the entry-per-dot exactness -
the print would no longer be the operator. Sanity check on paper: the
matrix square must measure exactly its listed side length.

Ranked per paper by the fitted preference model ({weights_name}).

| paper | rank | title | id | n | f (Hz) | rho | clip | print size | file |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(index)}
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/fitted_weights.json")
    ap.add_argument("--per-paper-top", type=int, default=0,
                    help="also keep only the global top-K per paper "
                         "(0 = best per track, all tracks)")
    ap.add_argument("--grid", choices=sorted(PAPER_MM),
                    help="only candidates sized for this paper grid")
    ap.add_argument("--sheet", choices=sorted(PAPER_MM),
                    help="print on this sheet instead of the grid's "
                         "native paper (requires --pitch)")
    ap.add_argument("--pitch", type=float,
                    help="dot pitch in mm (default: the candidate's own)")
    args = ap.parse_args()
    if args.sheet and not (args.grid and args.pitch):
        raise SystemExit("--sheet requires --grid and --pitch")
    score = scorer_from_weights(json.loads((ROOT / args.weights).read_text()))

    cands = []
    for cj in iter_candidate_files(ROOT / "runs"):
        d = json.loads(cj.read_text())
        if "paper" not in d:
            continue                          # pre-paper-era candidate
        if d.get("display") == "inverted":
            continue    # no preference data on inversion yet (exp_invert)
        d["_score"] = score(d["phi"])
        cands.append(d)
    if not cands:
        raise SystemExit("no paper-sweep candidates yet")
    cands.sort(key=lambda d: d["_score"])

    grids = [args.grid] if args.grid else sorted({d["paper"] for d in cands})
    index = []
    for paper in grids:
        pool = [d for d in cands if d["paper"] == paper]
        seen, picks = set(), []
        for d in pool:
            if d["song"] not in seen:
                seen.add(d["song"])
                picks.append(d)
        if args.per_paper_top:
            picks = picks[:args.per_paper_top]
        out_name = f"{args.sheet}-{args.pitch:g}mm" if args.sheet else paper
        out_dir = OUT / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for rank, d in enumerate(picks, 1):
            pdf = make_print(d, out_dir, rank,
                             pitch=args.pitch, sheet=args.sheet)
            side = d["n"] * (args.pitch or d["pitch_mm"])
            index.append(
                f"| {out_name} | {rank} | {d.get('title','')[:36]} "
                f"| {d['id']} | {d['n']} | {d['f_hz']:.0f} | {d['rho']:.2f} "
                f"| {d['clip_pct']:g} | {side:.0f} x {side:.0f} mm "
                f"| {pdf.relative_to(ROOT)} |")
            print(f"{out_name} #{rank}: {d['id']} -> {pdf.name}")

    if args.sheet:
        pitch_desc = f"{args.pitch:g} mm dot ({25.4/args.pitch:g} dpi)"
        write_readme(OUT / f"{args.sheet}-{args.pitch:g}mm" / "README.md",
                     index, pitch_desc, Path(args.weights).name)
    else:
        write_readme(OUT / "README.md", index,
                     "0.25 mm dot (101.6 dpi native)", Path(args.weights).name)
    print(f"wrote {len(index)} prints")


if __name__ == "__main__":
    main()
