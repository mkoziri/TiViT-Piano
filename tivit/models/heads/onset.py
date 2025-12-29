"""Onset head builder."""

from __future__ import annotations

from typing import Iterable, Sequence

from src.models.tivit_piano import MultiLayerHead


def build_head(d_model: int = 192, out_dim: int = 88, hidden_dims: Iterable[int] | None = None, dropout: float = 0.1) -> MultiLayerHead:
    hidden = tuple(int(h) for h in (hidden_dims or (512,)))
    return MultiLayerHead(d_model, out_dim, hidden_dims=hidden, dropout=dropout)


__all__ = ["build_head"]

