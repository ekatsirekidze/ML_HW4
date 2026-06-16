"""
Run a hyperparameter grid for one architecture.
Each combination = separate W&B run (MLflow-style: one run per experiment).
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml


def parse_list(value):
    if isinstance(value, list):
        return value
    return [value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch_key", type=str, required=True,
                        help="Key in configs/experiments.yaml e.g. arch_v3_cnn_medium")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--project", type=str, default="fer-expression-recognition")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)

    if args.arch_key not in all_cfg:
        raise KeyError(f"{args.arch_key} not in {args.config}")

    cfg = all_cfg[args.arch_key]
    model_name = cfg["model_name"]

    grid_keys = ["learning_rate", "batch_size", "dropout", "weight_decay", "label_smoothing"]
    grid = {}
    for k in grid_keys:
        if k in cfg:
            grid[k] = parse_list(cfg[k])

    if not grid:
        grid = {"learning_rate": [0.001], "batch_size": [64], "dropout": [0.3], "weight_decay": [1e-4]}

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    print(f"Running {len(combos)} experiments for {model_name} ({args.arch_key})")

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        run_name = f"{model_name}_" + "_".join(f"{k}{params[k]}" for k in keys)

        cmd = [
            sys.executable,
            "train.py",
            "--train_csv", args.train_csv,
            "--model", model_name,
            "--epochs", str(cfg.get("epochs", 35)),
            "--lr", str(params.get("learning_rate", 0.001)),
            "--batch_size", str(params.get("batch_size", 64)),
            "--dropout", str(params.get("dropout", 0.3)),
            "--weight_decay", str(params.get("weight_decay", 1e-4)),
            "--project", args.project,
            "--group", model_name,
            "--run_name", run_name,
        ]

        if "channels" in cfg:
            cmd.extend(["--channels", ",".join(str(c) for c in cfg["channels"])])
        if "fc_dim" in cfg:
            cmd.extend(["--fc_dim", str(cfg["fc_dim"])])
        if "hidden_dims" in cfg:
            hd = cfg["hidden_dims"]
            if isinstance(hd, list):
                cmd.extend(["--hidden_dims", ",".join(str(x) for x in hd)])
        if cfg.get("use_augmentation"):
            cmd.append("--use_augmentation")
        else:
            cmd.append("--no_augmentation")
        if "label_smoothing" in params:
            cmd.extend(["--label_smoothing", str(params["label_smoothing"])])
        if args.entity:
            cmd.extend(["--entity", args.entity])

        print(f"\n[{i}/{len(combos)}] {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))


if __name__ == "__main__":
    main()
