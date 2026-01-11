"""TiViT model factory shim and registry bridge.

Purpose:
    - Provide a registry-backed entrypoint that mirrors the prior behaviour while using the new implementations.
    - Keep a single source of truth for model construction while the new layout stabilizes.
Key Functions/Classes:
    - ``build_model``: Resolve the configured backbone name and delegate to the registered builder (defaults to ViViT) with a legacy-compatible config.
CLI Arguments:
    (none)
Usage:
    Import ``build_model`` from this module or ``tivit.models`` to construct a model for training/evaluation scripts.
"""

from __future__ import annotations

from typing import Any, Mapping

from tivit.core.registry import MODELS, register_default_components

register_default_components()


def build_model(cfg: Mapping[str, Any]):
    """Build the requested backbone via the registry (defaults to ViViT)."""

    model_cfg = cfg.get("model", {}) if isinstance(cfg, Mapping) else {}
    backend = str(model_cfg.get("backend", "vivit") or "vivit").lower()
    builder = MODELS.get("vivit" if backend == "vivit" else backend)
    return builder(cfg)


__all__ = ["build_model"]
