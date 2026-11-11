"""A tiny local web app: generate levels in the browser with the trained model.

    python -m src.app                 # then open http://localhost:8000
    python -m src.app --port 8080

Uses only the Python standard library web server (no Flask needed). It loads the checkpoint
once, and every click on "Generate" runs the same sampling loop as src/generate.py.
"""
import json
import mimetypes
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from src.config import base_parser, get_device, load_config
from src.data import columns_to_rows, level_to_text, load_levels, rows_to_columns
from src.evaluate import level_metrics, ngram_set, summarise
from src.generate import fix_ground, generate_ids, load_checkpoint
from src.render import TILE_STYLE

STATIC = Path(__file__).parent / "static"
OUTPUTS = Path("outputs").resolve()


class State:
    """Everything loaded once at start-up."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = get_device()
        self.model, self.vocab, self.ckpt = load_checkpoint(cfg["train"]["checkpoint"], self.device)
        self.legend, self.levels = load_levels(cfg)
        self.window = self.ckpt["config"]["train"]["window"]
        self.train_names = self.ckpt["train_levels"]
        self.n = cfg["evaluate"]["novelty_window"]
        self.train_ngrams = set().union(*(ngram_set(self.levels[nm], self.n) for nm in self.train_names))
        real = [level_metrics(r, self.legend, self.train_ngrams, self.n) for r in self.levels.values()]
        val = [level_metrics(self.levels[nm], self.legend, self.train_ngrams, self.n)
               for nm in self.ckpt["val_levels"]]
        keys = ["widest_gap", "ground_fraction", "enemies_per_100", "qblocks_per_100", "pipes_valid"]
        self.real_means = {k: summarise(real, k)["mean"] for k in keys}
        self.real_means["copy_rate"] = summarise(val, "copy_rate")["mean"]  # held-out levels only
        self.lock = threading.Lock()  # one generation at a time

    def info(self):
        return {"train_levels": self.train_names, "val_levels": self.ckpt["val_levels"],
                "epoch": self.ckpt["epoch"], "val_loss": self.ckpt["val_loss"], "val_acc": self.ckpt["val_acc"],
                "vocab_size": len(self.vocab), "window": self.window, "device": str(self.device),
                "real_means": self.real_means,
                "tiles": {t: {"colour": c, "label": lab} for t, (c, lab) in TILE_STYLE.items()}}

    def generate(self, temperature, top_k, length, seed, seed_level, apply_fix, max_gap):
        rng = random.Random(seed)
        name = seed_level if seed_level in self.train_names else rng.choice(self.train_names)
        cols = rows_to_columns(self.levels[name])
        start = rng.randrange(0, len(cols) - self.window + 1)
        seed_cols = cols[start:start + self.window]
        with self.lock, torch.no_grad():
            ids = generate_ids(self.model, self.vocab.encode(seed_cols), length, self.window, temperature,
                               top_k, self.vocab.unk_id, torch.Generator().manual_seed(seed))
        rows = columns_to_rows(self.vocab.decode(ids, len(seed_cols[0])))
        gaps_filled = 0
        if apply_fix:
            rows, gaps_filled = fix_ground(rows, self.legend, max_gap)
        m = level_metrics(rows, self.legend, self.train_ngrams, self.n)
        return {"rows": rows, "seed_rows": columns_to_rows(seed_cols), "seed_level": name,
                "seed_start": start, "seed": seed, "metrics": m, "gaps_filled": gaps_filled,
                "text": level_to_text(rows)}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _file(self, path):
            if not path.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), ctype)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._file(STATIC / "index.html")
                if url.path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                if url.path == "/api/info":
                    return self._json(state.info())
                if url.path == "/api/generate":
                    temperature = float(q.get("temperature", 1.0))
                    top_k = int(q.get("top_k", 0))
                    length = int(q.get("length", 200))
                    if not (0.05 <= temperature <= 5 and 0 <= top_k <= 1000 and 16 <= length <= 1000):
                        return self._json({"error": "temperature 0.05-5, top_k 0-1000, length 16-1000"}, 400)
                    seed = int(q["seed"]) if q.get("seed", "") != "" else random.randrange(1_000_000)
                    return self._json(state.generate(temperature, top_k, length, seed, q.get("seed_level"),
                                                     q.get("fix_ground") == "1", int(q.get("max_gap", 4))))
                if url.path.startswith("/outputs/"):
                    target = (OUTPUTS / url.path[len("/outputs/"):]).resolve()
                    if OUTPUTS in target.parents:  # no escaping the outputs folder
                        return self._file(target)
                return self._send(404, b"not found", "text/plain")
            except ValueError as e:
                return self._json({"error": str(e)}, 400)

        def log_message(self, fmt, *args):  # quieter console
            print("  " + fmt % args)

    return Handler


def main():
    p = base_parser(__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = only this computer can open it")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    state = State(cfg)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Model loaded (epoch {state.ckpt['epoch']}, {state.device}). Open http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
