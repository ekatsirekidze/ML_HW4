from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

from src.constants import EMOTION_LABELS


def init_wandb_run(
    project: str,
    run_name: str,
    config: Dict[str, Any],
    group: Optional[str] = None,
    tags: Optional[list] = None,
    entity: Optional[str] = None,
) -> None:
    wandb.init(
        project=project,
        name=run_name,
        config=config,
        group=group,
        tags=tags or [],
        entity=entity,
        reinit=True,
    )


def log_epoch_metrics(
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    learning_rate: float,
    fit_diagnosis: str,
) -> None:
    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/accuracy": train_metrics["accuracy"],
            "train/macro_f1": train_metrics["macro_f1"],
            "val/loss": val_metrics["loss"],
            "val/accuracy": val_metrics["accuracy"],
            "val/macro_f1": val_metrics["macro_f1"],
            "train_val_gap/accuracy": train_metrics["accuracy"] - val_metrics["accuracy"],
            "train_val_gap/loss": val_metrics["loss"] - train_metrics["loss"],
            "learning_rate": learning_rate,
            "fit_diagnosis": fit_diagnosis,
        }
    )


def log_model_checks(forward_results: Dict, backward_results: Dict) -> None:
    wandb.log(
        {
            "checks/forward_pass_ok": int(forward_results["forward_pass_ok"]),
            "checks/backward_pass_ok": int(backward_results["backward_pass_ok"]),
            "checks/total_grad_norm": backward_results["total_grad_norm"],
            "checks/loss_value": backward_results["loss_value"],
            "checks/vanishing_grad": int(backward_results["vanishing_grad_warning"]),
            "checks/exploding_grad": int(backward_results["exploding_grad_warning"]),
        }
    )
    wandb.log(
        {
            "checks/forward": forward_results,
            "checks/backward_summary": {
                k: backward_results[k]
                for k in [
                    "loss_finite",
                    "all_params_have_grad",
                    "total_grad_norm",
                    "vanishing_grad_warning",
                    "exploding_grad_warning",
                    "backward_pass_ok",
                ]
            },
        }
    )


def log_confusion_matrix(targets, preds, save_path: str) -> None:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[EMOTION_LABELS[i] for i in range(7)],
        yticklabels=[EMOTION_LABELS[i] for i in range(7)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    wandb.log({"val/confusion_matrix": wandb.Image(save_path)})
    plt.close()


def save_checkpoint(model, path: str, extra: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    wandb.save(path)
