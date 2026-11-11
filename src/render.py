"""Draw a level grid as coloured squares (PNG via matplotlib) and as a camera-pan GIF (Pillow).

These are pictures of the tile grid, not screenshots of the game and not gameplay.

    python -m src.render path/to/level.txt            # writes level.png and level_camera_pan.gif
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image, ImageDraw

# tile -> (colour, label). Colours are our own choice, not the game's art.
TILE_STYLE = {
    "-": ("#a8d8ff", "empty"),
    "X": ("#7a4a1f", "ground"),
    "S": ("#d2691e", "breakable brick"),
    "?": ("#ffc400", "question block (full)"),
    "Q": ("#b8a07a", "question block (empty)"),
    "E": ("#e02020", "enemy"),
    "<": ("#0b6e1f", "pipe top-left"),
    ">": ("#0b6e1f", "pipe top-right"),
    "[": ("#2fb34a", "pipe left"),
    "]": ("#2fb34a", "pipe right"),
    "o": ("#fff36b", "coin"),
    "B": ("#202020", "cannon top"),
    "b": ("#5a5a5a", "cannon bottom"),
}
TILES = list(TILE_STYLE)
UNKNOWN_COLOUR = "#ff00ff"


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def level_to_index(rows):
    idx = {t: i for i, t in enumerate(TILES)}
    return np.array([[idx.get(ch, len(TILES)) for ch in row] for row in rows])


def save_level_png(rows, path, title="Level", tile_inches=0.12):
    grid = level_to_index(rows)
    h, w = grid.shape
    cmap = ListedColormap([TILE_STYLE[t][0] for t in TILES] + [UNKNOWN_COLOUR])
    fig_w = max(6, w * tile_inches)
    fig, ax = plt.subplots(figsize=(fig_w, h * tile_inches + 1.6))
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(TILES) + 0.5, interpolation="nearest", aspect="equal")
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3, alpha=0.5)
    ax.tick_params(which="both", left=False, labelleft=False, bottom=True, length=0)
    ax.set_xticks(range(0, w, 20))
    ax.set_xlabel("column")
    ax.set_title(title)
    # Legend: only tiles that appear; the two halves of a pipe share one entry.
    entries = {}
    for t in TILES:
        if any(t in r for r in rows):
            colour, label = TILE_STYLE[t]
            if t in "<>":
                label, t = "pipe top", "<>"
            elif t in "[]":
                label, t = "pipe body", "[]"
            entries[label] = (colour, t)
    handles = [Patch(facecolor=c, edgecolor="#555", label=f"{lab}  '{t}'") for lab, (c, t) in entries.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=min(len(handles), 7), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def level_to_image(rows, px=12):
    """Pillow image of the grid, px pixels per tile, with a thin outline on every non-empty tile."""
    h, w = len(rows), len(rows[0])
    img = Image.new("RGB", (w * px, h * px), _rgb(TILE_STYLE["-"][0]))
    d = ImageDraw.Draw(img)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "-":
                continue
            colour = _rgb(TILE_STYLE.get(ch, (UNKNOWN_COLOUR,))[0])
            d.rectangle([c * px, r * px, (c + 1) * px - 1, (r + 1) * px - 1], fill=colour, outline=(40, 40, 40))
    return img


def save_camera_pan_gif(rows, path, view_cols=32, step=2, px=12, ms_per_frame=80):
    """Slide a fixed-size camera window from left to right across the level and save as a GIF.

    This is a CAMERA PAN over a static picture. Nothing moves, nobody plays.
    """
    full = level_to_image(rows, px)
    w_cols = len(rows[0])
    view_cols = min(view_cols, w_cols)
    banner = 16
    frames = []
    for start in list(range(0, w_cols - view_cols + 1, step)) or [0]:
        crop = full.crop((start * px, 0, (start + view_cols) * px, full.height))
        frame = Image.new("RGB", (crop.width, crop.height + banner), (255, 255, 255))
        frame.paste(crop, (0, banner))
        ImageDraw.Draw(frame).text((4, 2), f"camera pan (not gameplay) - columns {start}-{start + view_cols - 1}",
                                   fill=(0, 0, 0))
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms_per_frame, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("level_txt")
    args = p.parse_args()
    path = Path(args.level_txt)
    rows = [r for r in path.read_text(encoding="utf-8").splitlines() if r]
    save_level_png(rows, path.with_suffix(".png"), path.stem)
    save_camera_pan_gif(rows, path.with_name(path.stem + "_camera_pan.gif"))
    print(f"saved {path.with_suffix('.png')} and {path.stem}_camera_pan.gif")


if __name__ == "__main__":
    main()
