"""
Generate Kaggle submission.csv from a trained checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import create_test_loader
from src.models.architectures import build_model
from src.utils.seed import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default="test.csv")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", {})
    model_name = config.get("architecture", "cnn_best")

    model_kwargs = {k: v for k, v in config.items() if k in ("dropout", "channels", "fc_dim", "hidden_dims")}
    model = build_model(model_name, **model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    device = get_device()
    model = model.to(device)
    model.eval()

    loader = create_test_loader(args.test_csv, batch_size=args.batch_size)
    preds = []
    with torch.no_grad():
        for images in tqdm(loader, desc="predict"):
            images = images.to(device)
            logits = model(images)
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    sub = pd.read_csv(args.test_csv)
    if len(preds) != len(sub):
        raise ValueError(f"Prediction count {len(preds)} != test rows {len(sub)}")

    submission = pd.DataFrame({"emotion": preds})
    submission.to_csv(args.output, index=False)
    print(f"Saved {args.output} with {len(submission)} predictions")


if __name__ == "__main__":
    main()
