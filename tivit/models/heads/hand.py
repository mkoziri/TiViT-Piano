"""Hand presence head builder.

Purpose:
    - Provide left/right hand detection logits consistent with the legacy multitask head while fitting the new registry structure.
Key Functions/Classes:
    - ``build_head``: Construct a two-class (or configurable) hand presence head with optional hidden layers.
CLI Arguments:
    (none)
Usage:
    from tivit.models.heads.hand import build_head
"""

from __future__ import annotations

from typing import Iterable

from tivit.models.heads.common import MultiLayerHead


def build_head(d_model: int = 192, out_dim: int = 2, hidden_dims: Iterable[int] | None = None, dropout: float = 0.1) -> MultiLayerHead:
    """Create the hand-presence head MLP with configurable width/depth."""
    hidden = tuple(int(h) for h in (hidden_dims or (256,)))
    return MultiLayerHead(d_model, out_dim, hidden_dims=hidden, dropout=dropout)


__all__ = ["build_head"]
