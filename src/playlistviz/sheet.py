"""Labeled contact sheets for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_sheet(tiles: list[tuple[str, np.ndarray]], path: Path,
               tile_size: int = 400, cols: int | None = None,
               readme: str | None = None) -> None:
    """Save a grid of labeled tiles.

    tiles: (label, float image in [0,1]) pairs - 2D grayscale or (h, w, 3)
    RGB - drawn with a caption bar under each tile. Optionally writes a
    README.md next to the sheet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(tiles)
    cols = cols or min(n, 3)
    rows = (n + cols - 1) // cols
    pad, caption = 8, 22
    cell_w, cell_h = tile_size + pad, tile_size + caption + pad

    color = any(img.ndim == 3 for _, img in tiles)
    mode = "RGB" if color else "L"
    bg = (24, 24, 24) if color else 24
    fg = (230, 230, 230) if color else 230
    sheet = Image.new(mode, (cols * cell_w + pad, rows * cell_h + pad), bg)
    draw = ImageDraw.Draw(sheet)
    for k, (label, img) in enumerate(tiles):
        arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(arr, "RGB" if img.ndim == 3 else "L")
        if color and im.mode != "RGB":
            im = im.convert("RGB")
        im = im.resize((tile_size, tile_size), Image.LANCZOS)
        x = pad + (k % cols) * cell_w
        y = pad + (k // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.text((x + 2, y + tile_size + 4), label, fill=fg)
    sheet.save(path)

    if readme is not None:
        (path.parent / "README.md").write_text(readme)
