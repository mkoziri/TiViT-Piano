"""Hand gating prior placeholder."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .base import Prior


class HandGatingPrior(Prior):
    def __init__(self, mode: str = "loss_reweight", strength: float = 1.0) -> None:
        self.mode = mode
        self.strength = float(strength)

    def apply_to_logits(self, logits: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        # No-op placeholder that preserves structure; actual gating lives in training loop.
        return logits


def build_prior(cfg: Mapping[str, Any] | None = None) -> Prior:
    cfg = cfg or {}
    return HandGatingPrior(mode=str(cfg.get("mode", "loss_reweight")), strength=float(cfg.get("strength", 1.0)))


__all__ = ["HandGatingPrior", "build_prior"]

