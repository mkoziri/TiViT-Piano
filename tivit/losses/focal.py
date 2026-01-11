"""Binary focal loss helpers.

Purpose:
    - Provide a focal BCE variant for class imbalance.
    - Expose a registry-friendly builder for configs.
Key Functions/Classes:
    - focal_loss: focal BCE on logits with alpha/gamma.
    - build_loss: closure that binds default kwargs.
CLI Arguments:
    (none)
Usage:
    loss_fn = build_loss(alpha=0.25, gamma=2.0)
    loss = loss_fn(logits, targets)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid focal loss used for sparse positive classes."""
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy(prob, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    focal = (alpha * (1 - p_t) ** gamma) * ce
    if reduction == "mean":
        return focal.mean()
    if reduction == "sum":
        return focal.sum()
    return focal


def build_loss(**kwargs):
    """Return a closure that applies focal loss with the provided defaults."""
    def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(logits, target, **kwargs)

    return _loss


__all__ = ["focal_loss", "build_loss"]
