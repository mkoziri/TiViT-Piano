"""Temporal harmony smoothness prior placeholder."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import Prior


class ChordSmoothnessPrior(Prior):
    def __init__(self, strength: float = 0.0) -> None:
        self.strength = float(strength)

    def apply_to_logits(self, logits: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        # Lightweight smoothing: average with a detached copy to avoid extra memory.
        if not logits:
            return logits
        smoothed = {}
        for name, tensor in logits.items():
            if torch.is_tensor(tensor) and tensor.ndim >= 3 and self.strength > 0.0:
                # smooth along time axis without allocating full conv kernels
                avg = torch.stack(
                    [
                        tensor,
                        torch.roll(tensor, shifts=1, dims=-2),
                        torch.roll(tensor, shifts=-1, dims=-2),
                    ],
                    dim=0,
                ).mean(0)
                smoothed[name] = tensor.lerp(avg, self.strength)
            else:
                smoothed[name] = tensor
        return smoothed


def build_prior(cfg: Mapping[str, Any] | None = None) -> Prior:
    cfg = cfg or {}
    return ChordSmoothnessPrior(strength=float(cfg.get("strength", 0.0)))


__all__ = ["ChordSmoothnessPrior", "build_prior"]

