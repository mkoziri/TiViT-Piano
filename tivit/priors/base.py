"""Prior interface."""

from __future__ import annotations

from typing import Any, Mapping

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


__all__ = ["Prior", "null_prior"]

