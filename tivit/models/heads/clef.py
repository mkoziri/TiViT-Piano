"""Clef prediction head builder.

Purpose:
    - Provide a clef classification head that mirrors the legacy ``MultiLayerHead`` setup while fitting the new registry layout.
Key Functions/Classes:
    - ``build_head``: Construct a clef head with configurable class count, hidden layers, and dropout.
CLI Arguments:
    (none)
Usage:
    from tivit.models.heads.clef import build_head
"""

from __future__ import annotations

from typing import Iterable

from tivit.models.heads.common import MultiLayerHead


def build_head(d_model: int = 192, out_dim: int = 3, hidden_dims: Iterable[int] | None = None, dropout: float = 0.1) -> MultiLayerHead:
    """Create the clef head MLP with optional hidden dims and dropout."""
    hidden = tuple(int(h) for h in (hidden_dims or (256,)))
    return MultiLayerHead(d_model, out_dim, hidden_dims=hidden, dropout=dropout)


__all__ = ["build_head"]
