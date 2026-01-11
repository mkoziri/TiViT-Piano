"""Weight loading and freezing helpers.

Purpose:
    - Provide lightweight utilities for restoring checkpoints and freezing
      model submodules during finetuning or evaluation.
Key Functions/Classes:
    - ``load_weights``: Load a checkpoint (optionally with ``state_dict`` key)
      and apply it to a model instance.
    - ``freeze_module``: Disable gradients for all parameters in a module.
CLI Arguments:
    (none)
Usage:
    from tivit.models.init import load_weights, freeze_module
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch


def load_weights(model, checkpoint: str | Path, *, strict: bool = True, map_location: Optional[str] = "cpu"):
    """Load weights from ``checkpoint`` into ``model`` with optional strictness."""
    state = torch.load(Path(checkpoint), map_location=map_location)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)
    return model


def freeze_module(module) -> None:
    """Mark every parameter in ``module`` as frozen (no gradients)."""
    for param in module.parameters():
        param.requires_grad_(False)


__all__ = ["load_weights", "freeze_module"]
