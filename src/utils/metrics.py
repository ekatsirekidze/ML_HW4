from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "targets": all_targets,
    }


def diagnose_fit(train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> str:
    """Classify underfitting vs overfitting from train/val gap."""
    train_acc = train_metrics["accuracy"]
    val_acc = val_metrics["accuracy"]
    gap = train_acc - val_acc

    if train_acc < 0.55 and val_acc < 0.55:
        return "underfit"
    if gap > 0.12 and train_acc > 0.70:
        return "overfit"
    if gap > 0.08 and train_acc > 0.65:
        return "mild_overfit"
    return "good_fit"


def get_classification_report_dict(targets, preds, label_names: Dict[int, str]) -> Dict:
    report = classification_report(
        targets,
        preds,
        target_names=[label_names[i] for i in sorted(label_names)],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(targets, preds)
    return {"classification_report": report, "confusion_matrix": cm.tolist()}
