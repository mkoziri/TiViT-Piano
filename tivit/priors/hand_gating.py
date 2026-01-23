"""Hand-gating prior helpers for training.

Purpose:
    - Provide a training-time hand-gating prior interface and reach-weight helper.
    - Keep decode-time gating in ``tivit.postproc.hand_gate_runtime``.
Key Functions/Classes:
    - HandGatingPrior: Placeholder prior used by the registry.
    - build_prior: Factory for training-time hand gating config.
    - build_reach_weights: Compute loss reweighting masks from hand reach labels.
CLI Arguments:
    (none)
Usage:
    from tivit.priors.hand_gating import build_reach_weights
    weights = build_reach_weights(hand_reach, hand_reach_valid, cfg, device=device, dtype=roll.dtype)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from .base import Prior


class HandGatingPrior(Prior):
    def __init__(self, mode: str = "loss_reweight", strength: float = 1.0) -> None:
        self.mode = mode
        self.strength = float(strength)

    def apply_to_logits(self, logits: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        # No-op placeholder that preserves structure; actual gating lives in training loop.
        return logits


def build_reach_weights(
    hand_reach: torch.Tensor | None,
    hand_reach_valid: torch.Tensor | None,
    cfg: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Return loss reweighting mask for hand reach gating, or None if disabled."""
    if not torch.is_tensor(hand_reach) or not torch.is_tensor(hand_reach_valid):
        return None
    gating_mode = str(cfg.get("mode", "off")).lower()
    if gating_mode != "loss_reweight":
        return None
    try:
        strength = float(cfg.get("strength", 1.0))
    except (TypeError, ValueError):
        strength = 1.0

    hr = hand_reach.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    hr_valid = hand_reach_valid.to(device=device, dtype=dtype)
    if hr_valid.dim() == 2:
        hr_valid = hr_valid.unsqueeze(-1).expand_as(hr)
    if hr_valid.shape != hr.shape:
        return None
    neg_weight = 1.0 + strength * (1.0 - hr)
    return torch.where(hr_valid > 0.5, neg_weight, torch.ones_like(hr))


def build_prior(cfg: Mapping[str, Any] | None = None) -> Prior:
    cfg = cfg or {}
    return HandGatingPrior(mode=str(cfg.get("mode", "loss_reweight")), strength=float(cfg.get("strength", 1.0)))


__all__ = ["HandGatingPrior", "build_prior", "build_reach_weights"]
