"""ViViT-style backbone wrapper."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.models.factory import build_model as _legacy_build_model


def build_model(cfg: Mapping[str, Any]):
    cfg_copy = deepcopy(cfg)
    model_cfg = cfg_copy.setdefault("model", {})
    model_cfg.setdefault("backend", "vivit")
    return _legacy_build_model(cfg_copy)


__all__ = ["build_model"]

