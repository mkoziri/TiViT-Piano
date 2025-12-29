"""Platt scaling utilities."""

from __future__ import annotations

import torch


def platt_scale(logits: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return torch.sigmoid(a * logits + b)


__all__ = ["platt_scale"]

