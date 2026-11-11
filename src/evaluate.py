"""Compare generated levels with real ones using simple, countable tile statistics.

    python -m src.evaluate
    python -m src.evaluate --set evaluate.n_levels=10     # quicker

Writes outputs/eval/{summary.md, summary.json, per_level.csv, distributions.png, temperature.png}
and a few sample images to outputs/samples/.
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import base_parser, get_device, load_config, set_seed
from src.data import level_to_text, load_levels, rows_to_columns, solid_tiles
from src.generate import fix_ground, generate_levels, load_checkpoint

METRICS = {
    "widest_gap": "widest gap in ground row (tiles)",
    "ground_fraction": "ground fraction (bottom row solid)",
    "enemies_per_100": "enemies per 100 columns",
    "qblocks_per_100": "question blocks per 100 columns",
    "pipes_valid": "pipe validity (share of levels valid)",
    "copy_rate": "copy rate: 8-col windows found in training",
}


# ---------------------------------------------------------------- per-level metrics
def widest_gap(rows, legend):
    """Longest run of non-solid tiles in the bottom row."""
    solid = solid_tiles(legend)
    best = run = 0
    for ch in rows[-1]:
        run = 0 if ch in solid else run + 1
        best = max(best, run)
    return best


def ground_fraction(rows, legend):
    solid = solid_tiles(legend)
    return sum(ch in solid for ch in rows[-1]) / len(rows[-1])


def per_100_columns(rows, chars):
    return 100 * sum(row.count(ch) for row in rows for ch in chars) / len(rows[0])


def pipe_errors(rows):
    """List of pipe rule violations. Rules (all 15 real levels pass them):
    - '<' must be directly followed by '>' and every '>' directly preceded by '<' (same for '[' ']')
    - a pipe body '[' / ']' must have '<' or '[' / '>' or ']' directly above it (bodies line up)
    - a pipe top-left '<' or body-left '[' must stand on '[' or ground 'X' (unless on the last row)
    """
    errs = []
    h, w = len(rows), len(rows[0])
    for r in range(h):
        for c in range(w):
            t = rows[r][c]
            right = rows[r][c + 1] if c + 1 < w else None
            left = rows[r][c - 1] if c > 0 else None
            above = rows[r - 1][c] if r > 0 else None
            below = rows[r + 1][c] if r + 1 < h else None
            if t == "<" and right != ">":
                errs.append((r, c, "'<' not followed by '>'"))
            if t == ">" and left != "<":
                errs.append((r, c, "'>' not preceded by '<'"))
            if t == "[" and right != "]":
                errs.append((r, c, "'[' not followed by ']'"))
            if t == "]" and left != "[":
                errs.append((r, c, "']' not preceded by '['"))
            if t == "[" and above not in ("<", "["):
                errs.append((r, c, "'[' without pipe above"))
            if t == "]" and above not in (">", "]"):
                errs.append((r, c, "']' without pipe above"))
            if t in "<[" and below is not None and below not in ("[", "X"):
                errs.append((r, c, f"'{t}' not standing on pipe or ground"))
    return errs


def ngram_set(rows, n):
    cols = rows_to_columns(rows)
    return {tuple(cols[i:i + n]) for i in range(len(cols) - n + 1)}


def copy_rate(rows, train_ngrams, n):
    """Share of n-column windows in `rows` that appear exactly somewhere in the training levels."""
    cols = rows_to_columns(rows)
    grams = [tuple(cols[i:i + n]) for i in range(len(cols) - n + 1)]
    return sum(g in train_ngrams for g in grams) / len(grams) if grams else float("nan")


def level_metrics(rows, legend, train_ngrams, n):
    n_pipe_errors = len(pipe_errors(rows))
    return {
        "widest_gap": widest_gap(rows, legend),
        "ground_fraction": ground_fraction(rows, legend),
        "enemies_per_100": per_100_columns(rows, "E"),
        "qblocks_per_100": per_100_columns(rows, "?Q"),
        "pipes_valid": int(n_pipe_errors == 0),
        "pipe_errors": n_pipe_errors,
        "copy_rate": copy_rate(rows, train_ngrams, n),
        "width": len(rows[0]),
    }


def summarise(records, key):
    vals = [r[key] for r in records if not (isinstance(r[key], float) and np.isnan(r[key]))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals), "n": len(vals)}


# ---------------------------------------------------------------- charts
REAL_COLOUR = "#52514e"
T_COLOURS = ["#8fbcef", "#2a78d6", "#0d3f80"]   # one blue hue, light -> dark = low -> high temperature


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_distributions(groups, path):
    """groups: list of (label, colour, records). One panel per metric."""
    keys = list(METRICS)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    rng = np.random.default_rng(0)
    for ax, key in zip(axes.flat, keys):
        labels = [g[0] for g in groups]
        if key == "pipes_valid":
            vals = [100 * statistics.fmean(r[key] for r in g[2]) for g in groups]
            ax.bar(range(len(groups)), vals, color=[g[1] for g in groups], width=0.6)
            for i, v in enumerate(vals):
                ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
            ax.set_ylim(0, 112)
            ax.set_ylabel("% of levels with valid pipes")
        else:
            data = [[r[key] for r in g[2] if not np.isnan(r[key])] for g in groups]
            if key == "copy_rate":  # training levels are 1.0 by definition: compare with held-out levels only
                data[0] = [r[key] for r in groups[0][2] if r["split"] == "val"]
                labels[0] = "real held-out (2)"
            ax.boxplot(data, positions=range(len(groups)), widths=0.5, showfliers=False,
                       medianprops={"color": "black"}, whiskerprops={"color": "#888"},
                       capprops={"color": "#888"}, boxprops={"color": "#888"})
            for i, (d, g) in enumerate(zip(data, groups)):
                x = i + rng.uniform(-0.15, 0.15, len(d))
                ax.scatter(x, d, s=12, color=g[1], alpha=0.7, edgecolors="none", zorder=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(METRICS[key], fontsize=10)
        _style(ax)
    fig.suptitle("Generated levels (raw model output) vs real levels", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_temperature(temps, raw, fixed, real, path):
    """Mean +/- std of each metric vs temperature, raw and with the optional ground fix."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, METRICS):
        series = [(raw, "raw model output", "#2a78d6", "o")]
        if key in ("widest_gap", "ground_fraction"):  # the ground fix only changes the bottom row
            series.append((fixed, "with ground fix (post-processed)", "#eb6834", "s"))
        for recs, label, colour, marker in series:
            m = np.array([summarise(recs[t], key)["mean"] for t in temps])
            s = np.array([summarise(recs[t], key)["std"] for t in temps])
            if key == "pipes_valid":
                m, s = 100 * m, 0 * s
            ax.plot(temps, m, marker=marker, ms=8, lw=2, color=colour, label=label)
            ax.fill_between(temps, np.maximum(m - s, 0), m + s, color=colour, alpha=0.12)
        if key == "copy_rate":
            ref = [r for r in real if r["split"] == "val"]
            ref_label = "real held-out levels (mean)"
        else:
            ref, ref_label = real, "real levels (mean)"
        rv = summarise(ref, key)["mean"] * (100 if key == "pipes_valid" else 1)
        ax.axhline(rv, color=REAL_COLOUR, ls="--", lw=1.5, label=ref_label)
        if key in ("widest_gap", "copy_rate"):
            ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(temps)
        ax.set_xlabel("sampling temperature")
        ax.set_title(METRICS[key] + (" (%)" if key == "pipes_valid" else ""), fontsize=10)
        _style(ax)
    fig.suptitle("Effect of temperature (line = mean, band = +/- 1 std over generated levels)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    ec, gc = cfg["evaluate"], cfg["generate"]
    n = ec["novelty_window"]
    device = get_device()
    model, vocab, ckpt = load_checkpoint(cfg["train"]["checkpoint"], device)
    legend, levels = load_levels(cfg)
    train_names, val_names = ckpt["train_levels"], ckpt["val_levels"]
    window = ckpt["config"]["train"]["window"]

    train_ngrams = set().union(*(ngram_set(levels[nm], n) for nm in train_names))
    real = []
    for nm, rows in levels.items():
        rec = level_metrics(rows, legend, train_ngrams, n)
        rec.update(source="real", level=nm, split="val" if nm in val_names else "train")
        real.append(rec)
    real_val = [r for r in real if r["split"] == "val"]

    out = Path(ec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    samples = Path("outputs/samples")
    samples.mkdir(parents=True, exist_ok=True)
    from src.render import save_camera_pan_gif, save_level_png

    temps = ec["temperatures"]
    raw, fixed, gaps_filled = {}, {}, {}
    for ti, t in enumerate(temps):
        print(f"Generating {ec['n_levels']} levels at T={t} ...")
        levs = generate_levels(model, vocab, levels, train_names, ec["n_levels"], gc["length"], window,
                               t, gc["top_k"], cfg["seed"] + 1000 * (ti + 1))
        raw[t], fixed[t], gaps_filled[t] = [], [], []
        for i, lv in enumerate(levs):
            r = level_metrics(lv["rows"], legend, train_ngrams, n)
            r.update(source=f"T={t}", level=i, split="generated")
            raw[t].append(r)
            frows, nf = fix_ground(lv["rows"], legend, gc["max_gap"])
            f = level_metrics(frows, legend, train_ngrams, n)
            f.update(source=f"T={t} +fix", level=i, split="generated")
            fixed[t].append(f)
            gaps_filled[t].append(nf)
            if i < 2:  # keep a couple of samples per temperature
                stem = f"T{str(t).replace('.', 'p')}_{i:02d}"   # e.g. T0p5_00 (no dot -> no suffix confusion)
                (samples / f"{stem}.txt").write_text(level_to_text(lv["rows"]), encoding="utf-8")
                save_level_png(lv["rows"], samples / f"{stem}.png",
                               f"Generated, T={t}, raw model output (seeded from {lv['seed_level']})")
                if i == 0:
                    save_level_png(frows, samples / f"{stem}_fixground.png",
                                   f"Same level with optional ground fix ({nf} gaps > {gc['max_gap']} filled)")
                if t == 1.0 and i == 0:
                    save_camera_pan_gif(lv["rows"], samples / f"{stem}_camera_pan.gif")

    # A real level for side-by-side comparison
    save_level_png(levels["mario-1-1"], samples / "real_mario-1-1.png", "Real level 1-1 (from the dataset)")

    # ---- summary
    keys = list(METRICS) + ["pipe_errors"]
    real_train = [r for r in real if r["split"] == "train"]
    table = {"real (all 15)": {k: summarise(real, k) for k in keys},
             "real train (13)": {k: summarise(real_train, k) for k in keys},
             f"real val {val_names}": {k: summarise(real_val, k) for k in keys}}
    for t in temps:
        table[f"T={t} raw"] = {k: summarise(raw[t], k) for k in keys}
        table[f"T={t} +ground fix"] = {k: summarise(fixed[t], k) for k in keys}
    lines = ["| group | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for g, row in table.items():
        cells = []
        for k in keys:
            m, s = row[k]["mean"], row[k]["std"]
            cells.append(f"{100 * m:.0f}%" if k == "pipes_valid" else f"{m:.2f} ± {s:.2f}")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    md += (f"\n\nGenerated levels: {ec['n_levels']} per temperature, {gc['length']} columns each, top-k={gc['top_k']}."
           f"\nReal levels are {min(r['width'] for r in real)}-{max(r['width'] for r in real)} columns wide, "
           f"so 'widest gap' is not perfectly comparable (longer levels have more chances for a wide gap)."
           f"\nGround fix: gaps wider than {gc['max_gap']} filled; mean gaps filled per level: "
           + ", ".join(f"T={t}: {statistics.fmean(gaps_filled[t]):.2f}" for t in temps)
           + f"\nCopy-rate reference: the 2 held-out real levels have {summarise(real_val, 'copy_rate')['mean']:.2f}"
           f" of their {n}-column windows in the training levels.\n")
    (out / "summary.md").write_text(md, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"table": table, "gaps_filled": {str(t): gaps_filled[t] for t in temps},
                                                  "val_levels": val_names}, indent=2))
    all_recs = real + [r for t in temps for r in raw[t] + fixed[t]]
    with open(out / "per_level.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "level", "split"] + keys + ["width"])
        w.writeheader()
        for r in all_recs:
            w.writerow({k: r[k] for k in w.fieldnames})

    groups = [("real (15)", REAL_COLOUR, real)] + [(f"T={t}", c, raw[t]) for t, c in zip(temps, T_COLOURS)]
    plot_distributions(groups, out / "distributions.png")
    plot_temperature(temps, raw, fixed, real, out / "temperature.png")
    print(md)
    print(f"Saved charts and tables to {out}/ and samples to {samples}/")


if __name__ == "__main__":
    main()
