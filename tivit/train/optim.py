"""Optimizer + LR sched builders."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch


def build_optimizer(model, cfg: Mapping[str, Any]):
    training_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    optim_cfg = cfg.get("optim", {}) if isinstance(cfg, Mapping) else {}
    lr = float(training_cfg.get("learning_rate", optim_cfg.get("learning_rate", 3e-4)))
    weight_decay = float(training_cfg.get("weight_decay", optim_cfg.get("weight_decay", 0.01)))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


__all__ = ["build_optimizer"]

