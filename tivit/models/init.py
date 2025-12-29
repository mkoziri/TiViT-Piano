"""Weight loading / freezing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def load_weights(model, checkpoint: str | Path, *, strict: bool = True, map_location: Optional[str] = "cpu"):
    state = torch.load(Path(checkpoint), map_location=map_location)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)
    return model


def freeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


__all__ = ["load_weights", "freeze_module"]

