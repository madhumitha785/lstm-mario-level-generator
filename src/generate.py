"""Generate new levels one column at a time, then save them as .txt, .png and a camera-pan .gif.

    python -m src.generate
    python -m src.generate --set generate.temperature=0.7 --set generate.top_k=10
    python -m src.generate --set generate.fix_ground=true     # optional post-processing
"""
import json
import random
from pathlib import Path

import torch

from src.config import base_parser, get_device, load_config, set_seed
from src.data import ColumnVocab, columns_to_rows, level_to_text, load_levels, rows_to_columns, solid_tiles
from src.model import build_model


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = ColumnVocab.from_dict(ckpt["vocab"])
    model = build_model(ckpt["config"], len(vocab)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, vocab, ckpt


def sample_next(logits, temperature=1.0, top_k=0, unk_id=0, generator=None):
    """Pick one id from a (vocab,) logit vector.

    1. divide logits by the temperature (low T -> sharper, high T -> flatter)
    2. never sample the unknown token
    3. optionally keep only the k highest-scoring columns
    4. softmax -> probabilities -> random draw
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits.float() / temperature
    logits[unk_id] = -float("inf")
    if top_k and top_k > 0:
        k = min(top_k, logits.numel() - 1)
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs.cpu(), 1, generator=generator).item())


@torch.no_grad()
def generate_ids(model, seed_ids, length, window, temperature=1.0, top_k=0, unk_id=0, generator=None):
    """Autoregressive loop: predict, sample, append, slide the window. Returns only the NEW ids."""
    device = next(model.parameters()).device
    context = list(seed_ids[-window:])
    new = []
    for _ in range(length):
        x = torch.tensor([context[-window:]], dtype=torch.long, device=device)
        nxt = sample_next(model(x)[0], temperature, top_k, unk_id, generator)
        new.append(nxt)
        context.append(nxt)
    return new


def fix_ground(rows, legend, max_gap):
    """OPTIONAL post-processing: fill bottom-row gaps wider than `max_gap` with ground 'X'.

    Returns (new_rows, number_of_gaps_filled). The original rows are not modified.
    """
    solid = solid_tiles(legend)
    bottom = list(rows[-1])
    filled, c = 0, 0
    while c < len(bottom):
        if bottom[c] in solid:
            c += 1
            continue
        start = c
        while c < len(bottom) and bottom[c] not in solid:
            c += 1
        if c - start > max_gap:
            bottom[start:c] = ["X"] * (c - start)
            filled += 1
    return rows[:-1] + ["".join(bottom)], filled


def pick_seed(levels, train_names, window, rng):
    """A random window of real columns from a random TRAINING level."""
    name = rng.choice(train_names)
    cols = rows_to_columns(levels[name])
    start = rng.randrange(0, len(cols) - window + 1)
    return name, start, cols[start:start + window]


def generate_levels(model, vocab, levels, train_names, n, length, window, temperature, top_k, seed):
    """Generate n levels. Returns list of dicts with rows + where the seed came from."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    height = len(next(iter(levels.values())))
    out = []
    for _ in range(n):
        name, start, seed_cols = pick_seed(levels, train_names, window, rng)
        ids = generate_ids(model, vocab.encode(seed_cols), length, window, temperature, top_k,
                           vocab.unk_id, g)
        out.append({"rows": columns_to_rows(vocab.decode(ids, height)),
                    "seed_level": name, "seed_start": start})
    return out


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    gc = cfg["generate"]
    device = get_device()
    model, vocab, ckpt = load_checkpoint(cfg["train"]["checkpoint"], device)
    legend, levels = load_levels(cfg)
    window = ckpt["config"]["train"]["window"]

    from src.render import save_camera_pan_gif, save_level_png  # imported here to keep tests light

    out_dir = Path(gc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"T{str(gc['temperature']).replace('.', 'p')}_k{gc['top_k']}"
    results = generate_levels(model, vocab, levels, ckpt["train_levels"], gc["n_levels"], gc["length"],
                              window, gc["temperature"], gc["top_k"], cfg["seed"])
    meta = []
    for i, r in enumerate(results):
        rows = r["rows"]
        name = f"level_{tag}_{i:02d}"
        info = {"name": name, "seed_level": r["seed_level"], "seed_start": r["seed_start"],
                "temperature": gc["temperature"], "top_k": gc["top_k"], "fix_ground": gc["fix_ground"]}
        if gc["fix_ground"]:
            rows, n_fixed = fix_ground(rows, legend, gc["max_gap"])
            name += "_fixground"
            info.update(name=name, gaps_filled=n_fixed, max_gap=gc["max_gap"])
        (out_dir / f"{name}.txt").write_text(level_to_text(rows), encoding="utf-8")
        title = f"Generated level (T={gc['temperature']}, top-k={gc['top_k']})"
        if gc["fix_ground"]:
            title += f" + ground fix ({info['gaps_filled']} gaps filled)"
        save_level_png(rows, out_dir / f"{name}.png", title)
        save_camera_pan_gif(rows, out_dir / f"{name}_camera_pan.gif")
        meta.append(info)
        print(f"saved {out_dir / name}.txt/.png/_camera_pan.gif  (seeded from {r['seed_level']} "
              f"columns {r['seed_start']}-{r['seed_start'] + window - 1})")
    (out_dir / f"meta_{tag}{'_fixground' if gc['fix_ground'] else ''}.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
