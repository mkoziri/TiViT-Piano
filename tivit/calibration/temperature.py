"""Temperature scaling utilities."""

from __future__ import annotations

import torch


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(float(temperature), 1e-5)
    return logits / temp


__all__ = ["temperature_scale"]

