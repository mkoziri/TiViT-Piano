"""Dataloader builders wired through the shared registry.

The goal is to reuse existing dataset implementations while exposing a stable
entrypoint under ``tivit.data``. We avoid extra allocations by keeping the
wrapper minimal and deferring heavy work to the underlying dataset modules.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from tivit.core.registry import DATASETS, register_default_components

register_default_components()


def make_dataloader(
    cfg: Mapping[str, Any],
    split: str,
    drop_last: bool = False,
    *,
    seed: Optional[int] = None,
):
    """Build a dataloader for ``split`` using the registered dataset."""

    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    name = str(dataset_cfg.get("name", "omaps")).lower()
    builder = DATASETS.get(name)
    return builder(cfg, split=split, drop_last=drop_last, seed=seed)


__all__ = ["make_dataloader"]
