"""Unit tests. Run with:  python -m pytest -q

Tests that need the real dataset are skipped if data/ has not been downloaded.
"""
import json
from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.data import (ColumnVocab, build_dataset, columns_to_rows, make_windows, parse_level,
                      rows_to_columns, split_levels)
from src.evaluate import copy_rate, ground_fraction, ngram_set, pipe_errors, widest_gap
from src.generate import fix_ground, generate_ids, sample_next
from src.model import LevelLSTM

ROOT = Path(__file__).resolve().parents[1]
LEGEND = {"X": ["solid", "ground"], "-": ["passable", "empty"], "E": ["enemy"],
          "<": ["solid", "pipe"], ">": ["solid", "pipe"], "[": ["solid", "pipe"], "]": ["solid", "pipe"]}
TINY = ["------",
        "--<>--",
        "--[]E-",
        "XXXX-X"]


# ---------------------------------------------------------------- tile validity
def test_unknown_character_fails_loudly():
    with pytest.raises(ValueError, match="not in the legend"):
        parse_level("--Z-\nXXXX\n", LEGEND, "bad.txt", expected_rows=2)


def test_ragged_rows_fail():
    with pytest.raises(ValueError, match="different widths"):
        parse_level("---\nXXXX\n", LEGEND, "ragged.txt", expected_rows=2)


def test_row_mismatch_pad_and_skip():
    with pytest.warns(UserWarning):
        rows = parse_level("XXXX\n", LEGEND, "short.txt", expected_rows=3, row_mismatch="pad")
    assert rows == ["----", "----", "XXXX"]
    with pytest.warns(UserWarning):
        assert parse_level("XXXX\n", LEGEND, "short.txt", expected_rows=3, row_mismatch="skip") is None


# ---------------------------------------------------------------- tokenizer
def test_columns_round_trip():
    assert columns_to_rows(rows_to_columns(TINY)) == TINY


def test_vocab_round_trip_and_unk():
    cols = rows_to_columns(TINY)
    vocab = ColumnVocab(cols)
    assert vocab.itos[0] == "<unk>"
    assert len(vocab) == len(set(cols)) + 1
    assert vocab.decode(vocab.encode(cols), height=4) == cols
    assert vocab.encode(["EEEE"]) == [vocab.unk_id]
    assert ColumnVocab.from_dict(json.loads(json.dumps(vocab.to_dict()))).itos == vocab.itos


# ---------------------------------------------------------------- shapes
def test_windows_shapes():
    X, y = make_windows(list(range(20)), 16)
    assert X.shape == (4, 16) and y.shape == (4,)
    assert X[0, -1] == 15 and y[0] == 16


def test_dataset_does_not_cross_levels():
    a, b = ["XXXXX"], ["-----"]  # two 1-row levels with 5 columns each
    vocab = ColumnVocab(rows_to_columns(a) + rows_to_columns(b))
    ds = build_dataset([a, b], vocab, window=3)
    assert len(ds) == 2 + 2


def test_model_output_shape():
    m = LevelLSTM(vocab_size=50, embed_dim=8, hidden_size=16, num_layers=2, dropout=0.3)
    assert m(torch.randint(0, 50, (4, 16))).shape == (4, 50)


def test_split_is_by_level_and_seeded():
    names = [f"l{i}" for i in range(15)]
    t1, v1 = split_levels(names, 2, 42)
    t2, v2 = split_levels(names, 2, 42)
    assert v1 == v2 and len(v1) == 2 and not set(v1) & set(t1) and len(t1) == 13


# ---------------------------------------------------------------- sampling & generation
def test_sampling_never_returns_unk_and_top1_is_argmax():
    logits = torch.tensor([10.0, 1.0, 3.0, 2.0])  # unk (id 0) has the highest score
    g = torch.Generator().manual_seed(0)
    assert all(sample_next(logits, 1.0, 0, 0, g) != 0 for _ in range(50))
    assert all(sample_next(logits, 1.5, 1, 0, g) == 2 for _ in range(20))
    with pytest.raises(ValueError):
        sample_next(logits, 0.0)


def test_generate_ids_length():
    m = LevelLSTM(vocab_size=10, embed_dim=4, hidden_size=8)
    ids = generate_ids(m, [1] * 16, length=25, window=16, generator=torch.Generator().manual_seed(0))
    assert len(ids) == 25 and all(0 < i < 10 for i in ids)


# ---------------------------------------------------------------- metrics & post-processing
def test_metrics_on_tiny_level():
    assert widest_gap(TINY, LEGEND) == 1
    assert ground_fraction(TINY, LEGEND) == 5 / 6
    assert pipe_errors(TINY) == []
    broken = ["--<---", "--[]--", "XXXXXX"]
    assert len(pipe_errors(broken)) >= 2   # '<' without '>' and ']' without pipe above


def test_fix_ground():
    rows = ["--------", "XX-----X"]
    fixed, n = fix_ground(rows, LEGEND, max_gap=4)
    assert n == 1 and fixed[-1] == "XXXXXXXX" and rows[-1] == "XX-----X"
    same, n = fix_ground(rows, LEGEND, max_gap=5)
    assert n == 0 and same == rows


def test_copy_rate():
    train = ngram_set(TINY, 2)
    assert copy_rate(TINY, train, 2) == 1.0
    assert copy_rate(["EEEEEE"] * 4, train, 2) == 0.0


# ---------------------------------------------------------------- real data (skipped if absent)
def test_real_levels_load_if_downloaded():
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg["data"]["root"] = str(ROOT / cfg["data"]["root"])
    if not Path(cfg["data"]["root"]).exists():
        pytest.skip("dataset not downloaded")
    from src.data import load_levels
    legend, levels = load_levels(cfg)
    assert len(levels) == 15
    assert all(len(rows) == 14 for rows in levels.values())
    assert all(pipe_errors(rows) == [] for rows in levels.values())
