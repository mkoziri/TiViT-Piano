"""Binary cross entropy wrapper."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, target: torch.Tensor, *, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction=reduction)


def build_loss(**kwargs):
    def _loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return bce_loss(logits, target, **kwargs)

    return _loss


__all__ = ["bce_loss", "build_loss"]

