"""Backbone entrypoints.

Purpose:
    - Re-export backbone builders under concise names for registry and factory consumption.
Key Functions/Classes:
    - ``build_vivit`` / ``build_vit_small``: Construct ViViT or ViT-S tile backbones.
CLI Arguments:
    (none)
Usage:
    from tivit.models.backbones import build_vivit, build_vit_small
"""

from .vivit import build_model as build_vivit
from .vit_small import build_model as build_vit_small

__all__ = ["build_vivit", "build_vit_small"]
