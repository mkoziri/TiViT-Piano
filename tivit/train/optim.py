"""Optimizer and LR scheduler builders for TiViT-Piano training."""

# Purpose:
# - Build AdamW param groups with head-specific LR/WD multipliers.
# - Keep setup minimal to stay memory efficient for long clips.
#
# Key Functions/Classes:
# - build_optimizer: construct AdamW with grouped LR/weight decay.
#
# CLI Arguments:
# - (none; import-only utilities).
#
# Usage:
# - from tivit.train.optim import build_optimizer

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch


def _collect_params(module) -> Sequence[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad] if module is not None else []


def build_optimizer(model, cfg: Mapping[str, Any]):
    """
    Build AdamW with head LR/WD multipliers (keeps defaults compact for memory).
    """
    training_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    optim_cfg = cfg.get("optim", {}) if isinstance(cfg, Mapping) else {}
    lr = float(training_cfg.get("learning_rate", optim_cfg.get("learning_rate", 3e-4)))
    weight_decay = float(training_cfg.get("weight_decay", optim_cfg.get("weight_decay", 0.01)))
    head_lr_mult = float(optim_cfg.get("head_lr_multiplier", 1.0))
    offset_wd_mult = float(optim_cfg.get("offset_weight_decay_multiplier", 1.0))

    head_pitch = getattr(model, "head_pitch", None)
    head_onset = getattr(model, "head_onset", None)
    head_offset = getattr(model, "head_offset", None)
    head_hand = getattr(model, "head_hand", None)
    head_clef = getattr(model, "head_clef", None)

    param_groups: list[dict[str, Any]] = []
    seen: set[int] = set()

    def _add_group(params, *, lr_mult: float = 1.0, wd_mult: float = 1.0) -> None:
        filtered = [p for p in params if p.requires_grad and id(p) not in seen]
        if not filtered:
            return
        seen.update(id(p) for p in filtered)
        param_groups.append(
            {
                "params": filtered,
                "lr": lr * lr_mult,
                "weight_decay": weight_decay * wd_mult,
            }
        )

    # Head groups with LR multiplier and optional offset WD multiplier.
    _add_group(_collect_params(head_pitch), lr_mult=head_lr_mult)
    _add_group(_collect_params(head_onset), lr_mult=head_lr_mult)
    _add_group(_collect_params(head_offset), lr_mult=head_lr_mult, wd_mult=offset_wd_mult)
    _add_group(_collect_params(head_hand), lr_mult=head_lr_mult)
    _add_group(_collect_params(head_clef), lr_mult=head_lr_mult)

    # Backbone / remaining parameters.
    remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    _add_group(remaining)

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)


__all__ = ["build_optimizer"]
