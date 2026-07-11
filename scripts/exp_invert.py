"""Inverted ink map: black becomes white for the top candidate per song.

The operator is untouched - inversion flips only the value -> ink ramp
(255 - pixel on the stored presentation, which is lossless), the same
kind of display freedom as the diverging palettes. For the best
candidate per (song, paper) under the fitted preference model this
writes a sibling candidate <id>_inv into the same song/paper folder with
metrics recomputed on the inverted image, so `playlistviz compare` will
pair original vs inverted within each song block and the BT loop can
learn whether inversion is liked. Prints ignore inverted candidates
until that data exists (the model has never seen an inverted image, so
its scores there are extrapolation).

Idempotent: existing _inv siblings are skipped; re-run after a sweep to
cover new songs.

Run: .venv/bin/python scripts/exp_invert.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from playlistviz.bt import scorer_from_weights
from playlistviz.loss import equal_weights, loss_vector, scalar_loss
from playlistviz.metrics import METRICS, features
from playlistviz.pool import candidate_dir, iter_candidate_files
from playlistviz.sheet import make_sheet

ROOT = Path(__file__).parent.parent
OUT = ROOT / "runs" / "exp_invert"
Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    score = scorer_from_weights(
        json.loads((ROOT / "runs" / "fitted_weights.json").read_text()))
    cands = []
    for cj in iter_candidate_files(ROOT / "runs"):
        d = json.loads(cj.read_text())
        if "paper" not in d or d.get("display") == "inverted":
            continue
        d["_score"] = score(d["phi"])
        cands.append(d)
    cands.sort(key=lambda d: d["_score"])

    tops = {}
    for d in cands:
        tops.setdefault((d["paper"], d["song"]), d)

    OUT.mkdir(parents=True, exist_ok=True)
    tiles, rows, made, skipped = [], [], 0, 0
    for (paper, song), d in sorted(tops.items()):
        parent_dir = candidate_dir(ROOT / "runs", d["id"])
        inv_id = d["id"] + "_inv"
        inv_dir = parent_dir.parent / inv_id
        if (inv_dir / "candidate.json").exists():
            skipped += 1
            continue
        u8 = np.asarray(Image.open(parent_dir / "presentation.png"))
        inv_u8 = 255 - u8          # exact in uint8; a float round-trip
        img = u8 / 255.0           # through save_png truncation is not
        inv = inv_u8 / 255.0
        phi = features(inv)
        inv_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(inv_u8, mode="L").save(inv_dir / "presentation.png")
        out = dict(d)
        out.pop("_score")
        out.update({
            "id": inv_id, "display": "inverted", "inverted_from": d["id"],
            "phi": {m.name: float(v) for m, v in zip(METRICS, phi)},
            "loss_vector": [float(v) for v in loss_vector(phi)],
            "loss_eq": float(scalar_loss(phi, equal_weights())),
        })
        (inv_dir / "candidate.json").write_text(json.dumps(out, indent=2))
        made += 1
        dsk = phi[[m.name for m in METRICS].index("lum_skewness")]
        rows.append(f"| {d.get('title','')[:32]} | {paper} | {d['id']} "
                    f"| {d['phi']['lum_skewness']:+.2f} -> {dsk:+.2f} |")
        if paper == "A2":  # one sheet row per song at the biggest paper
            tiles.append((f"{(d.get('title') or d['id'])[:30]}", img))
            tiles.append(("inverted", inv))
        print(f"{paper} {d.get('title', d['id'])[:40]}: {inv_id}")

    if tiles:
        make_sheet(tiles, OUT / "invert_pairs.png", tile_size=440, cols=6)
    (OUT / "README.md").write_text(f"""\
# exp_invert: black becomes white

For the best candidate per (song, paper) under the fitted preference
model, a sibling candidate `<id>_inv` with the ink ramp inverted
(255 - pixel, lossless; the operator is untouched) now sits in the same
song/paper folder. `playlistviz compare` pairs candidates within a song
block, so originals and inversions will meet head-to-head and the BT fit
learns whether inversion is preferred. Prints exclude inverted
candidates until such data exists.

{made} inversions written, {skipped} already existed.
`invert_pairs.png`: original | inverted pairs for the A2 tops.

Largest predictable metric change is luminance skewness (sign flips):

| song | paper | parent id | lum_skewness orig -> inv |
|---|---|---|---|
{chr(10).join(rows)}
""")
    print(f"\n{made} inversions written ({skipped} existed) "
          f"-> {OUT}/README.md")


if __name__ == "__main__":
    main()
