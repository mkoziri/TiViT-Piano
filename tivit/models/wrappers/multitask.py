"""Backbone+head composition wrapper.

Purpose:
    - Keep a stable entrypoint for constructing the multitask model from the new implementation.
Key Functions/Classes:
    - ``build_model``: Forward the provided config to the new factory so backbone+head composition stays consistent.
CLI Arguments:
    (none)
Usage:
    from tivit.models.wrappers.multitask import build_model
"""

from __future__ import annotations

from typing import Any, Mapping

from tivit.models.factory import build_model as _build_model


def build_model(cfg: Mapping[str, Any]):
    """Construct the multitask model via the new factory."""

    return _build_model(cfg)


__all__ = ["build_model"]
