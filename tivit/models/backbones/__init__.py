"""Backbone entrypoints."""

from .vivit import build_model as build_vivit
from .vit_small import build_model as build_vit_small

__all__ = ["build_vivit", "build_vit_small"]
