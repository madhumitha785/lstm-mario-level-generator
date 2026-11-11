"""Train the LSTM on sliding windows of columns, with early stopping on validation loss.

    python -m src.train
    python -m src.train --set train.max_epochs=5     # quick smoke run
"""
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from src.config import base_parser, get_device, load_config, set_seed
from src.data import ColumnVocab, build_dataset, load_levels, rows_to_columns, split_levels
from src.model import build_model


def evaluate_loss(model, loader, device, unk_id=0):
    """Validation metrics.

    Returns (loss_known, acc_known, loss_all):
      loss_known / acc_known - only windows whose target column exists in the training vocabulary.
                               This is what early stopping uses.
      loss_all               - every window, including targets that are <unk>. Reported for honesty.
    Why: a target that never appears in training (<unk>) cannot be predicted by any model, and on
    this tiny dataset ~24% of validation targets are <unk> (level 1-2 is underground with a ceiling).
    Those windows dominate the loss and made early stopping pick an almost untrained model.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    tot_known = tot_all = 0.0
    correct = n_known = n_all = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            losses = loss_fn(logits, y)
            known = y != unk_id
            tot_all += losses.sum().item()
            n_all += len(y)
            tot_known += losses[known].sum().item()
            correct += (logits.argmax(1)[known] == y[known]).sum().item()
            n_known += known.sum().item()
    return tot_known / max(n_known, 1), correct / max(n_known, 1), tot_all / max(n_all, 1)


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = get_device()
    print(f"Device: {device}")

    legend, levels = load_levels(cfg)
    train_names, val_names = split_levels(list(levels), cfg["data"]["n_val_levels"], cfg["seed"])
    print(f"Levels: {len(levels)} total | train {len(train_names)} | val {val_names}")

    # Vocabulary from TRAINING levels only; val columns never seen in training become <unk>.
    train_cols = [c for n in train_names for c in rows_to_columns(levels[n])]
    vocab = ColumnVocab(train_cols)
    val_cols = [c for n in val_names for c in rows_to_columns(levels[n])]
    n_unk = sum(c not in vocab.stoi for c in val_cols)
    all_unique = len({c for rows in levels.values() for c in rows_to_columns(rows)})
    print(f"Vocabulary size: {len(vocab)} ({len(vocab) - 1} unique training columns + <unk>); "
          f"unique columns in all 15 levels: {all_unique}")
    print(f"Validation columns not in the vocabulary (-> <unk>): {n_unk}/{len(val_cols)}")

    W = cfg["train"]["window"]
    train_ds = build_dataset([levels[n] for n in train_names], vocab, W)
    val_ds = build_dataset([levels[n] for n in val_names], vocab, W)
    print(f"Windows: train {len(train_ds)} | val {len(val_ds)} (window={W})")

    # Simple reference point: always predict the most frequent training column.
    most_common = Counter(train_ds.tensors[1].tolist()).most_common(1)[0][0]
    yv = val_ds.tensors[1]
    baseline_acc = (yv[yv != vocab.unk_id] == most_common).float().mean().item()
    print(f"Baseline (always predict most common column) val accuracy on known targets: {baseline_acc:.3f}")

    g = torch.Generator().manual_seed(cfg["seed"])
    bs = cfg["train"]["batch_size"]
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=bs)

    model = build_model(cfg, len(vocab)).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = nn.CrossEntropyLoss()

    ckpt_path = Path(cfg["train"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best, bad_epochs, history = math.inf, 0, []
    t0 = time.time()
    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(y)
            n += len(y)
        train_loss = running / n
        val_loss, val_acc, val_loss_all = evaluate_loss(model, val_dl, device, vocab.unk_id)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc,
                        "val_loss_incl_unk": val_loss_all})
        improved = val_loss < best
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val (known targets) {val_loss:.4f} "
              f"acc {val_acc:.3f} | val incl. <unk> {val_loss_all:.4f}{'  *saved' if improved else ''}")
        if improved:
            best, bad_epochs = val_loss, 0
            torch.save({"model": model.state_dict(), "vocab": vocab.to_dict(), "config": cfg,
                        "train_levels": train_names, "val_levels": val_names,
                        "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["train"]["patience"]:
                print(f"Early stopping: no val improvement for {bad_epochs} epochs.")
                break
    print(f"Done in {time.time() - t0:.1f}s. Best val loss {best:.4f} -> {ckpt_path}")

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    summary = {"history": history, "best_val_loss": best, "vocab_size": len(vocab),
               "val_unk_columns": n_unk, "val_columns": len(val_cols), "baseline_val_acc": baseline_acc,
               "train_windows": len(train_ds), "val_windows": len(val_ds),
               "train_levels": train_names, "val_levels": val_names, "device": str(device)}
    (out / "train_history.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ep = [h["epoch"] for h in history]
    ax.plot(ep, [h["train_loss"] for h in history], lw=2, color="#2a78d6", label="train")
    ax.plot(ep, [h["val_loss"] for h in history], lw=2, color="#eb6834", label="validation (known targets, used for early stopping)")
    ax.plot(ep, [h["val_loss_incl_unk"] for h in history], lw=1.5, ls="--", color="#eb6834",
            label="validation incl. <unk> targets")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training curve")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "train_curve.png", dpi=120)
    print(f"Saved {out / 'train_curve.png'} and {out / 'train_history.json'}")


if __name__ == "__main__":
    main()
