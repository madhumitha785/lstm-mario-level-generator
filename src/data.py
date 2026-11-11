"""Loading levels, the tile legend, the column tokenizer and training windows.

A level file is a grid of characters, one row per line (14 rows for SMB).
We read it column by column: each column (14 characters, top to bottom) is one token.
"""
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch

UNK = "<unk>"
EMPTY = "-"


# ---------------------------------------------------------------- legend & files
def load_legend(path):
    """Return {tile_char: [tags]} read from smb.json."""
    with open(path, "r", encoding="utf-8") as f:
        legend = json.load(f)["tiles"]
    if EMPTY not in legend:
        raise ValueError(f"Legend {path} has no empty tile {EMPTY!r}")
    return legend


def solid_tiles(legend):
    """Characters tagged 'solid' in the legend (ground, bricks, blocks, pipes, cannons)."""
    return {ch for ch, tags in legend.items() if "solid" in tags}


def parse_level(text, legend, name="<level>", expected_rows=14, row_mismatch="pad"):
    """Turn level text into a list of equal-length row strings.

    Fails loudly on characters not in the legend or ragged rows. If the row count is not
    `expected_rows`: 'pad' adds empty sky rows on top (or drops top rows if too tall),
    'skip' returns None. Either way a warning is printed.
    """
    rows = [r.rstrip("\r") for r in text.splitlines() if r.strip("\r") != ""]
    if not rows:
        raise ValueError(f"{name}: file is empty")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{name}: rows have different widths {sorted(widths)}")
    for r_i, row in enumerate(rows):
        for c_i, ch in enumerate(row):
            if ch not in legend:
                raise ValueError(f"{name}: character {ch!r} at row {r_i}, column {c_i} "
                                 f"is not in the legend {sorted(legend)}")
    if len(rows) != expected_rows:
        if row_mismatch == "skip":
            warnings.warn(f"{name}: has {len(rows)} rows, expected {expected_rows} -> SKIPPED")
            return None
        if row_mismatch != "pad":
            raise ValueError(f"row_mismatch must be 'pad' or 'skip', got {row_mismatch!r}")
        warnings.warn(f"{name}: has {len(rows)} rows, expected {expected_rows} -> padded/cropped at the top")
        width = len(rows[0])
        if len(rows) < expected_rows:
            rows = [EMPTY * width] * (expected_rows - len(rows)) + rows
        else:
            rows = rows[len(rows) - expected_rows:]
    return rows


def load_levels(cfg):
    """Load every mario-*.txt level. Returns (legend, {name: rows})."""
    root = Path(cfg["data"]["root"])
    legend_path = root / cfg["data"]["legend_file"]
    levels_dir = root / cfg["data"]["levels_dir"]
    if not legend_path.exists() or not levels_dir.exists():
        raise FileNotFoundError(f"Dataset not found under {root}. Run: python scripts/download_data.py")
    legend = load_legend(legend_path)
    levels = {}
    for path in sorted(levels_dir.glob("mario-*.txt")):
        rows = parse_level(path.read_text(encoding="utf-8"), legend, path.name,
                           cfg["data"]["expected_rows"], cfg["data"]["row_mismatch"])
        if rows is not None:
            levels[path.stem] = rows
    if not levels:
        raise RuntimeError(f"No levels loaded from {levels_dir}")
    return legend, levels


# ---------------------------------------------------------------- rows <-> columns
def rows_to_columns(rows):
    """['ab', 'cd'] -> ['ac', 'bd']  (each column read top to bottom)."""
    return ["".join(col) for col in zip(*rows)]


def columns_to_rows(columns):
    return ["".join(row) for row in zip(*columns)]


def level_to_text(rows):
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------- tokenizer
class ColumnVocab:
    """Maps each distinct column string to an integer id. Id 0 is the unknown token."""

    def __init__(self, columns=()):
        self.itos = [UNK] + sorted(set(columns))
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk_id = 0

    def __len__(self):
        return len(self.itos)

    def encode(self, columns):
        return [self.stoi.get(c, self.unk_id) for c in columns]

    def decode(self, ids, height=14):
        # The unknown token has no shape; we draw it as an all-empty column.
        # (Generation masks it out, so this only matters for encoded validation data.)
        return [EMPTY * height if i == self.unk_id else self.itos[i] for i in ids]

    def to_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        v = cls()
        v.itos = list(d["itos"])
        v.stoi = {s: i for i, s in enumerate(v.itos)}
        return v


# ---------------------------------------------------------------- split & windows
def split_levels(names, n_val, seed):
    """Hold out whole levels (not windows) so overlapping windows cannot leak."""
    names = sorted(names)
    val = sorted(random.Random(seed).sample(names, n_val))
    train = [n for n in names if n not in val]
    return train, val


def make_windows(ids, window):
    """Sliding windows: X[i] = ids[i:i+window], y[i] = ids[i+window]."""
    ids = np.asarray(ids, dtype=np.int64)
    n = len(ids) - window
    if n <= 0:
        return np.zeros((0, window), np.int64), np.zeros((0,), np.int64)
    X = np.stack([ids[i:i + window] for i in range(n)])
    y = ids[window:]
    return X, y


def build_dataset(level_rows, vocab, window):
    """Windows from several levels, never crossing a level boundary."""
    Xs, ys = [], []
    for rows in level_rows:
        X, y = make_windows(vocab.encode(rows_to_columns(rows)), window)
        Xs.append(X)
        ys.append(y)
    X = torch.from_numpy(np.concatenate(Xs))
    y = torch.from_numpy(np.concatenate(ys))
    return torch.utils.data.TensorDataset(X, y)
