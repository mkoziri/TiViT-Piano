"""Shared head layers for TiViT models (GCR).

Purpose:
    - Provide a local copy of the lightweight MLP head used by all tasks while avoiding legacy imports.
Key Functions/Classes:
    - ``MultiLayerHead``: LayerNorm + MLP stack with optional dropout.
CLI Arguments:
    (none)
Usage:
    from tivit.models.heads.common import MultiLayerHead
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class MultiLayerHead(nn.Module):
    """Light-weight MLP head with normalization, hidden layers, and dropout."""

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (512,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(d_model)]

        in_dim = d_model
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


__all__ = ["MultiLayerHead"]
