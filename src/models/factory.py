"""Purpose:
    Provide a thin factory that instantiates :class:`TiViTPiano` from nested
    configuration dictionaries used across training and evaluation scripts.

Key Functions/Classes:
    - build_model(): Reads transformer/head parameters from ``cfg`` and returns
      a configured TiViT-Piano instance.

CLI:
    Not applicable; this module is imported by scripts such as
    :mod:`scripts.train` and :mod:`scripts.eval_thresholds`.
"""

from typing import Any, Mapping
from .tivit_piano import TiViTPiano

def _get(d: Mapping[str, Any], key: str, default):
    v = d.get(key, default)
    return v if v is not None else default

def build_model(cfg: Mapping[str, Any]):
    if "model" not in cfg or "transformer" not in cfg["model"]:
        raise ValueError("Config must have model.transformer section")

    mcfg = cfg["model"]
    tcfg = mcfg["transformer"]

    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    tiling_cfg = cfg.get("tiling", {}) if isinstance(cfg, Mapping) else {}
    
    return TiViTPiano(
        tiles=_get(mcfg, "tiles", 3),
        input_channels=_get(mcfg, "input_channels", 3),
        patch_size=_get(tcfg, "input_patch_size", 16),
        tube_size=_get(tcfg, "tube_size", 2),
        d_model=_get(tcfg, "d_model", 768),
        nhead=_get(tcfg, "num_heads", 8),
        depth_temporal=_get(tcfg, "depth_temporal", 2),
        depth_spatial=_get(tcfg, "depth_spatial", 2),
        depth_global=_get(tcfg, "depth_global", 1),
        global_tokens=_get(tcfg, "global_tokens", 2),
        mlp_ratio=_get(tcfg, "mlp_ratio", 4.0),
        dropout=_get(tcfg, "dropout", 0.1),
        head_mode=cfg["model"].get("head_mode", "clip"),
        tiling_cfg=tiling_cfg,
    )

