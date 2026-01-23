"""Prior interface and shared training-time helpers.

Purpose:
    - Define the minimal prior interface used by training-time components.
    - Provide shared helpers for training-time prior regularization.
Key Functions/Classes:
    - Prior: Base class for prior adapters.
    - null_prior: Return a no-op prior instance.
    - compute_prior_mean_regularizer: Optional onset/offset prior-mean regularizer.
CLI Arguments:
    (none)
Usage:
    from tivit.priors.base import compute_prior_mean_regularizer
    reg = compute_prior_mean_regularizer(logits, cfg, default_mean=0.12)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


class Prior:
    """Minimal interface for priors that adjust targets or logits."""

    def apply_to_targets(self, targets: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return targets

    def apply_to_logits(self, logits: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return logits


class _NullPrior(Prior):
    pass


def null_prior(*_: Any, **__: Any) -> Prior:
    return _NullPrior()


def compute_prior_mean_regularizer(
    logits: torch.Tensor,
    cfg: Mapping[str, Any],
    *,
    default_mean: float = 0.12,
) -> Optional[torch.Tensor]:
    """Compute a prior-mean regularizer term for a logit tensor."""
    prior_w = float(cfg.get("prior_weight", 0.0))
    if prior_w <= 0.0:
        return None
    prior_mean = float(cfg.get("prior_mean", default_mean))
    return prior_w * (torch.sigmoid(logits).mean() - prior_mean).abs()


__all__ = ["Prior", "null_prior", "compute_prior_mean_regularizer"]
