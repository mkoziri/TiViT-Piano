"""Backbone+head composition wrapper."""

from __future__ import annotations

from typing import Any, Mapping

from src.models.factory import build_model as _legacy_build_model


def build_model(cfg: Mapping[str, Any]):
    """
    Delegate to the legacy factory.

    Keeping a single implementation avoids behavioural drift while enabling the
    new registry/namespace layout.
    """

    return _legacy_build_model(cfg)


__all__ = ["build_model"]

