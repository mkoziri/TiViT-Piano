"""Onset head builder.

Purpose:
    - Provide the default onset classification head matching the legacy ``MultiLayerHead`` configuration.
Key Functions/Classes:
    - ``build_head``: Construct a feedforward head for onset logits with optional hidden sizes and dropout.
CLI Arguments:
    (none)
Usage:
    from tivit.models.heads.onset import build_head
"""

from __future__ import annotations

from typing import Iterable, Sequence

from tivit.models.heads.common import MultiLayerHead


def build_head(d_model: int = 192, out_dim: int = 88, hidden_dims: Iterable[int] | None = None, dropout: float = 0.1) -> MultiLayerHead:
    """Create the onset head MLP with configurable width/depth and dropout."""
    hidden = tuple(int(h) for h in (hidden_dims or (512,)))
    return MultiLayerHead(d_model, out_dim, hidden_dims=hidden, dropout=dropout)


__all__ = ["build_head"]
