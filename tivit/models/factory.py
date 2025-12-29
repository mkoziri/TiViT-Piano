"""Model factory that honors the registry-backed layout.

This wrapper keeps compatibility with the legacy :mod:`src.models.factory`
while allowing experiments to resolve backbones via the new registry.
"""

from __future__ import annotations

from typing import Any, Mapping

from tivit.core.registry import MODELS, register_default_components

register_default_components()


def build_model(cfg: Mapping[str, Any]):
    """Build a model using the registered backbone indicated by ``cfg``."""

    model_cfg = cfg.get("model", {}) if isinstance(cfg, Mapping) else {}
    backend = str(model_cfg.get("backend", "vivit") or "vivit").lower()
    builder = MODELS.get("vivit" if backend == "vivit" else backend)
    return builder(cfg)


__all__ = ["build_model"]

