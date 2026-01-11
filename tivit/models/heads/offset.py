"""Offset head builder.

Purpose:
    - Expose the offset prediction head via the new layout while reusing the legacy ``MultiLayerHead`` implementation.
Key Functions/Classes:
    - ``build_head``: Construct a feedforward head for offset logits with configurable hidden layers and dropout.
CLI Arguments:
    (none)
Usage:
    from tivit.models.heads.offset import build_head
"""

from __future__ import annotations

from typing import Iterable

from tivit.models.heads.common import MultiLayerHead


def build_head(d_model: int = 192, out_dim: int = 88, hidden_dims: Iterable[int] | None = None, dropout: float = 0.1) -> MultiLayerHead:
    """Create the offset head MLP with optional hidden dims and dropout."""
    hidden = tuple(int(h) for h in (hidden_dims or (512,)))
    return MultiLayerHead(d_model, out_dim, hidden_dims=hidden, dropout=dropout)


__all__ = ["build_head"]
