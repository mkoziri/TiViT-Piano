"""Binary cross entropy loss helpers.

Purpose:
    - Wrap BCE-with-logits for reuse across heads.
    - Expose a registry-friendly builder for configs.
Key Functions/Classes:
    - bce_loss: thin wrapper around torch BCE-with-logits.
    - build_loss: closure that binds default kwargs.
CLI Arguments:
    (none)
Usage:
    loss_fn = build_loss(reduction="mean")
    loss = loss_fn(logits, targets)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, target: torch.Tensor, *, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> torch.Tensor:
    """Binary cross-entropy with logits wrapper kept for registry compatibility."""
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction=reduction)


def build_loss(**kwargs):
    """Return a closure that applies BCE with the provided keyword defaults."""
    def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return bce_loss(logits, target, **kwargs)

    return _loss


__all__ = ["bce_loss", "build_loss"]
