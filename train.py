"""
Main training script — one W&B run per architecture + hyperparameter combo.
Usage:
  python train.py --model cnn_medium --lr 0.001 --batch_size 64 --dropout 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.checks.model_checks import run_model_checks
from src.constants import EMOTION_LABELS, NUM_CLASSES
from src.data.dataset import create_dataloaders
from src.models.architectures import build_model
from src.utils.metrics import diagnose_fit, evaluate_epoch, get_classification_report_dict
from src.utils.seed import count_parameters, get_device, set_seed
from src.utils.wandb_logger import (
    init_wandb_run,
    log_confusion_matrix,
    log_epoch_metrics,
    log_model_checks,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FER model with W&B logging")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--model", type=str, required=True,
                        choices=["mlp_underfit", "cnn_small", "cnn_medium", "cnn_deep_overfit", "cnn_best"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dims", type=str, default="128", help="For MLP, comma-separated")
    parser.add_argument("--channels", type=str, default="", help="For CNN, comma-separated e.g. 32,64,128")
    parser.add_argument("--fc_dim", type=int, default=256)
    parser.add_argument("--project", type=str, default="fer-expression-recognition")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None, help="W&B group = architecture family")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def build_model_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {"dropout": args.dropout}
    if args.model == "mlp_underfit":
        kwargs["hidden_dims"] = [int(x) for x in args.hidden_dims.split(",")]
    else:
        if args.channels:
            kwargs["channels"] = [int(x) for x in args.channels.split(",")]
        kwargs["fc_dim"] = args.fc_dim
        if args.model == "cnn_best":
            pass  # always uses BN
        elif args.model in ("cnn_small", "cnn_medium"):
            kwargs["use_bn"] = False
    return kwargs


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "macro_f1": 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    use_aug = True
    if args.no_augmentation:
        use_aug = False
    elif args.use_augmentation:
        use_aug = True
    elif args.model in ("mlp_underfit", "cnn_deep_overfit"):
        use_aug = False

    train_loader, val_loader, train_df, val_df = create_dataloaders(
        args.train_csv,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_augmentation=use_aug,
        num_workers=args.num_workers,
    )

    model_kwargs = build_model_kwargs(args)
    model = build_model(args.model, **model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    run_name = args.run_name or f"{args.model}_lr{args.lr}_bs{args.batch_size}_do{args.dropout}"
    group = args.group or args.model

    config = {
        "architecture": args.model,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "use_augmentation": use_aug,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_parameters": count_parameters(model),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "device": str(device),
        **model_kwargs,
    }

    init_wandb_run(
        project=args.project,
        run_name=run_name,
        config=config,
        group=group,
        tags=[args.model, "fer2013", "pytorch"],
        entity=args.entity,
    )

    # Forward / backward checks on one batch (lecture requirement)
    sample_images, sample_labels = next(iter(train_loader))
    sample_images = sample_images[:8].to(device)
    sample_labels = sample_labels[:8].to(device)
    fwd, bwd = run_model_checks(model, sample_images, sample_labels, criterion, NUM_CLASSES)
    log_model_checks(fwd, bwd)
    print("Forward check:", fwd)
    print("Backward check OK:", bwd["backward_pass_ok"], "| grad norm:", bwd["total_grad_norm"])

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_out = evaluate_epoch(model, val_loader, criterion, device)
        val_metrics = {k: v for k, v in val_out.items() if k not in ("predictions", "targets")}

        train_eval = evaluate_epoch(model, train_loader, criterion, device)
        train_metrics["macro_f1"] = train_eval["macro_f1"]

        scheduler.step(val_metrics["loss"])
        fit = diagnose_fit(train_metrics, val_metrics)
        log_epoch_metrics(epoch, train_metrics, val_metrics, optimizer.param_groups[0]["lr"], fit)

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "fit": fit,
            }
        )

        print(
            f"Epoch {epoch:03d} | train acc {train_metrics['accuracy']:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} | gap {train_metrics['accuracy'] - val_metrics['accuracy']:.4f} | {fit}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            ckpt_path = os.path.join(args.output_dir, f"{run_name}_best.pt")
            save_checkpoint(
                model,
                ckpt_path,
                extra={"epoch": epoch, "val_accuracy": best_val_acc, "config": config},
            )

    final_val = evaluate_epoch(model, val_loader, criterion, device)
    report = get_classification_report_dict(
        final_val["targets"], final_val["predictions"], EMOTION_LABELS
    )

    cm_path = os.path.join(args.output_dir, f"{run_name}_cm.png")
    log_confusion_matrix(final_val["targets"], final_val["predictions"], cm_path)

    summary = {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "final_val_accuracy": final_val["accuracy"],
        "final_train_val_gap": history[-1]["train"]["accuracy"] - history[-1]["val"]["accuracy"],
        "final_fit_diagnosis": history[-1]["fit"],
        "forward_check_ok": fwd["forward_pass_ok"],
        "backward_check_ok": bwd["backward_pass_ok"],
    }

    import wandb

    wandb.run.summary.update(summary)
    wandb.log({"final/classification_report": report["classification_report"]})

    results_path = os.path.join(args.output_dir, f"{run_name}_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "history": history, "report": report}, f, indent=2)
    wandb.save(results_path)

    wandb.finish()
    print("Done. Best val accuracy:", best_val_acc)


if __name__ == "__main__":
    main()
