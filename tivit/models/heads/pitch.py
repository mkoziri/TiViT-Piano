"""Pitch head builder.

Purpose:
    - Supply a pitch-activity head that mirrors the legacy multi-layer setup while living in the new registry-driven layout.
Key Functions/Classes:
    - ``build_head``: Construct a feedforward head for per-pitch activity.
CLI Arguments:
    (none)
Usage:
    from tivit.models.heads.pitch import build_head
"""

from __future__ import annotations

from typing import Iterable

from tivit.models.heads.common import MultiLayerHead


def build_head(d_model: int = 192, out_dim: int = 88, hidden_dims: Iterable[int] | None = None, dropout: float = 0.1) -> MultiLayerHead:
    """Create the pitch activity head MLP with optional hidden dims."""
    hidden = tuple(int(h) for h in (hidden_dims or (512,)))
    return MultiLayerHead(d_model, out_dim, hidden_dims=hidden, dropout=dropout)


__all__ = ["build_head"]
