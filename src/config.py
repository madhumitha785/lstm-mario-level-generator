"""Load config.yaml and apply command line overrides such as --set train.lr=0.0005."""
import argparse
import random

import numpy as np
import torch
import yaml


def load_config(path="config.yaml", overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node:
                raise KeyError(f"Unknown config section {p!r} in {key!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key {key!r}")
        node[parts[-1]] = yaml.safe_load(value)  # "0.5" -> 0.5, "true" -> True, "[1,2]" -> list
    return cfg


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set train.lr=0.0005 (repeatable)")
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
