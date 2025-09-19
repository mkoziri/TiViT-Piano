"""Purpose:
    Route dataset construction to the appropriate backend (OMAPS or PianoYT)
    while keeping a single public ``make_dataloader`` entry point.  This allows
    the rest of the project to import ``data.make_dataloader`` without knowing
    which dataset is selected in the configuration.

Key Functions/Classes:
    - make_dataloader(): Dispatches to the dataset-specific factory based on
      ``cfg['dataset']['name']``.
"""

from __future__ import annotations

from typing import Any, Dict


def make_dataloader(cfg: Dict[str, Any], split: str, drop_last: bool = False):
    dataset_cfg = cfg.get("dataset", {})
    if "pipeline_v2" not in dataset_cfg:
        dataset_cfg["pipeline_v2"] = False
    name = str(dataset_cfg.get("name", "OMAPS")).lower()

    if name == "omaps":
        from . import omaps_dataset as dataset_mod
    elif name == "pianoyt":
        from . import pianoyt_dataset as dataset_mod
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_cfg.get('name')}")

    return dataset_mod.make_dataloader(cfg, split, drop_last)


def is_pipeline_v2_enabled(cfg: Dict[str, Any]) -> bool:
    dataset_cfg = cfg.get("dataset", {})
    if "pipeline_v2" not in dataset_cfg:
        dataset_cfg["pipeline_v2"] = False
    return bool(dataset_cfg["pipeline_v2"])


__all__ = ["make_dataloader", "is_pipeline_v2_enabled"]
