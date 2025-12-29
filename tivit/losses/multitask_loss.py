"""Per-head loss composition helper."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import torch

from .bce import bce_loss
from .focal import focal_loss


def _select_loss_fn(head_cfg: Mapping[str, Any]):
    loss_name = str(head_cfg.get("loss", "bce")).lower()
    if loss_name == "focal":
        return lambda logits, target: focal_loss(
            logits,
            target,
            alpha=float(head_cfg.get("focal_alpha", 0.25)),
            gamma=float(head_cfg.get("focal_gamma", 2.0)),
        )
    return lambda logits, target: bce_loss(
        logits,
        target,
        pos_weight=None,
    )


def build_loss(head_weights: Mapping[str, float] | None = None):
    weights = {str(k): float(v) for k, v in (head_weights or {}).items()}

    def _loss(preds: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], cfg: Mapping[str, Any] | None = None):
        total = torch.tensor(0.0, device=next(iter(preds.values())).device if preds else "cpu")
        by_head: MutableMapping[str, torch.Tensor] = {}
        head_cfg = cfg or {}
        for name, logits in preds.items():
            if name not in targets:
                continue
            loss_fn = _select_loss_fn(head_cfg.get(name, {}) if isinstance(head_cfg, Mapping) else {})
            val = loss_fn(logits, targets[name])
            weighted = val * weights.get(name, 1.0)
            by_head[name] = weighted
            total = total + weighted
        by_head["total"] = total
        return total, by_head

    return _loss


__all__ = ["build_loss"]

