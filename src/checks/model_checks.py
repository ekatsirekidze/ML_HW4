"""
Forward and backward checks for neural network sanity verification.
These are the lecture-style diagnostics your instructor expects.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def forward_check(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_classes: int = 7,
) -> Dict[str, object]:
    """
    Forward pass sanity check:
    - output shape matches (batch, num_classes)
    - no NaN / Inf in logits
    - probabilities sum to ~1 after softmax
    """
    model.eval()
    with torch.no_grad():
        logits = model(sample_input)
        probs = torch.softmax(logits, dim=1)

    checks = {
        "output_shape_ok": tuple(logits.shape) == (sample_input.size(0), num_classes),
        "no_nan": not torch.isnan(logits).any().item(),
        "no_inf": not torch.isinf(logits).any().item(),
        "prob_sum_close_to_1": torch.allclose(probs.sum(dim=1), torch.ones(sample_input.size(0)), atol=1e-4),
        "logits_min": float(logits.min().item()),
        "logits_max": float(logits.max().item()),
        "logits_mean": float(logits.mean().item()),
    }
    checks["forward_pass_ok"] = all(
        checks[k] for k in ["output_shape_ok", "no_nan", "no_inf", "prob_sum_close_to_1"]
    )
    return checks


def backward_check(
    model: nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    criterion: nn.Module,
) -> Dict[str, object]:
    """
    Backward pass sanity check:
    - loss is finite
    - gradients exist for trainable parameters
    - gradient norms per layer (detect vanishing/exploding)
    """
    model.train()
    model.zero_grad(set_to_none=True)

    logits = model(sample_input)
    loss = criterion(logits, sample_target)
    loss.backward()

    grad_norms: Dict[str, float] = {}
    missing_grads: List[str] = []
    zero_grads: List[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing_grads.append(name)
            continue
        norm = param.grad.data.norm(2).item()
        grad_norms[name] = norm
        if norm == 0.0:
            zero_grads.append(name)

    total_norm = 0.0
    if grad_norms:
        total_norm = sum(v ** 2 for v in grad_norms.values()) ** 0.5

    checks = {
        "loss_finite": torch.isfinite(loss).item(),
        "loss_value": float(loss.item()),
        "all_params_have_grad": len(missing_grads) == 0,
        "missing_grad_params": missing_grads,
        "zero_grad_params": zero_grads,
        "total_grad_norm": total_norm,
        "layer_grad_norms": grad_norms,
        "vanishing_grad_warning": total_norm < 1e-7,
        "exploding_grad_warning": total_norm > 1e3,
    }
    checks["backward_pass_ok"] = (
        checks["loss_finite"]
        and checks["all_params_have_grad"]
        and not checks["vanishing_grad_warning"]
        and not checks["exploding_grad_warning"]
    )
    return checks


def run_model_checks(
    model: nn.Module,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    criterion: nn.Module,
    num_classes: int = 7,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    forward_results = forward_check(model, sample_input, num_classes=num_classes)
    backward_results = backward_check(model, sample_input, sample_target, criterion)
    return forward_results, backward_results
