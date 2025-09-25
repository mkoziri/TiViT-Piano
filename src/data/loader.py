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

_PIPELINE_WARNING_EMITTED = False
_PIPELINE_KEY = "pipeline_" + "v2"


def _consume_pipeline_flag(dataset_cfg: Dict[str, Any]) -> None:
    global _PIPELINE_WARNING_EMITTED

    if _PIPELINE_KEY not in dataset_cfg:
        return

    value = dataset_cfg.pop(_PIPELINE_KEY)
    if value in (False, "false", "False", 0):
        raise ValueError("pipeline_v1 has been removed")

    if not _PIPELINE_WARNING_EMITTED:
        key_path = f"dataset.{_PIPELINE_KEY}"
        print(f"{key_path} is ignored; v2 is permanent.")
        _PIPELINE_WARNING_EMITTED = True


def make_dataloader(cfg: Dict[str, Any], split: str, drop_last: bool = False):
    dataset_cfg = cfg.get("dataset", {})
    _consume_pipeline_flag(dataset_cfg)
    name = str(dataset_cfg.get("name", "OMAPS")).lower()

    if name == "omaps":
        from . import omaps_dataset as dataset_mod
    elif name == "pianoyt":
        from . import pianoyt_dataset as dataset_mod
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_cfg.get('name')}")

    return dataset_mod.make_dataloader(cfg, split, drop_last)


__all__ = ["make_dataloader"]
