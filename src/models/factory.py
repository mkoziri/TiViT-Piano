"""Purpose:
    Provide a thin factory that instantiates TiViT-Piano backends from nested
    configuration dictionaries used across training and evaluation scripts.

Key Functions/Classes:
    - build_model(): Reads ``cfg`` and returns the requested backend so higher
      level code stays agnostic to the underlying architecture.

CLI:
    Not applicable; this module is imported by scripts such as
    :mod:`scripts.train` and :mod:`scripts.eval_thresholds`.
"""

import logging
from typing import Any, Mapping, Sequence

from .tivit_piano import TiViTPiano
from .vits_tile import ViTSTilePiano

LOGGER = logging.getLogger(__name__)


def _get(d: Mapping[str, Any], key: str, default):
    v = d.get(key, default)
    return v if v is not None else default


def _backend_name(mcfg: Mapping[str, Any]) -> str:
    raw = mcfg.get("backend", "vivit")
    label = str(raw).strip().lower() if raw is not None else "vivit"
    return label or "vivit"


def build_model(cfg: Mapping[str, Any]):
    if "model" not in cfg:
        raise ValueError("Config must have a model section")

    mcfg = cfg["model"]
    backend = _backend_name(mcfg)
    LOGGER.info("[model] backend=%s", backend)

    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    tiling_cfg = cfg.get("tiling", {}) if isinstance(cfg, Mapping) else {}
    tiles = int(_get(dataset_cfg, "tiles", 3))
    input_channels = int(_get(dataset_cfg, "channels", 3))

    if backend == "vivit":
        if "transformer" not in mcfg:
            raise ValueError("model.transformer is required for the ViViT backend")
        tcfg = mcfg["transformer"]
        return TiViTPiano(
            tiles=tiles,
            input_channels=input_channels,
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
            head_mode=mcfg.get("head_mode", "clip"),
            tiling_cfg=tiling_cfg,
        )

    if backend == "vits_tile":
        vcfg = mcfg.get("vits_tile", {}) or {}
        tcfg = mcfg.get("transformer", {}) or {}
        input_hw = vcfg.get("input_hw", (145, 342))
        if not isinstance(input_hw, Sequence):
            raise ValueError("model.vits_tile.input_hw must be a 2-element sequence")
        if len(input_hw) < 2:
            raise ValueError("model.vits_tile.input_hw must provide height and width")
        dropout_default = _get(tcfg, "dropout", 0.1)
        return ViTSTilePiano(
            tiles=tiles,
            input_channels=input_channels,
            head_mode=mcfg.get("head_mode", "frame"),
            tiling_cfg=tiling_cfg,
            backbone_name=str(vcfg.get("backbone_name", "vit_small_patch16_224")),
            embed_dim=int(vcfg.get("embed_dim", 384)),
            pretrained=bool(vcfg.get("pretrained", True)),
            freeze_backbone=bool(vcfg.get("freeze_backbone", True)),
            input_hw=input_hw,
            temporal_layers=int(vcfg.get("temporal_layers", 2)),
            temporal_heads=int(vcfg.get("temporal_heads", 4)),
            global_mixing_layers=int(vcfg.get("global_mixing_layers", 1)),
            dropout=float(vcfg.get("dropout", dropout_default)),
        )

    raise ValueError(f"Unsupported model backend '{backend}'")
