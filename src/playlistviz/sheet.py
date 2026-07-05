"""Labeled contact sheets for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_sheet(tiles: list[tuple[str, np.ndarray]], path: Path,
               tile_size: int = 400, cols: int | None = None,
               readme: str | None = None) -> None:
    """Save a grid of labeled grayscale tiles.

    tiles: (label, 2D float image in [0,1]) pairs, drawn with a caption bar
    under each tile. Optionally writes a README.md next to the sheet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(tiles)
    cols = cols or min(n, 3)
    rows = (n + cols - 1) // cols
    pad, caption = 8, 22
    cell_w, cell_h = tile_size + pad, tile_size + caption + pad

    sheet = Image.new("L", (cols * cell_w + pad, rows * cell_h + pad), 24)
    draw = ImageDraw.Draw(sheet)
    for k, (label, img) in enumerate(tiles):
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "L")
        im = im.resize((tile_size, tile_size), Image.LANCZOS)
        x = pad + (k % cols) * cell_w
        y = pad + (k // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + tile_size + 4), label, fill=230)
    sheet.save(path)

    if readme is not None:
        (path.parent / "README.md").write_text(readme)
