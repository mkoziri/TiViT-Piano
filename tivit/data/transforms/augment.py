"""
Purpose:
    Apply lightweight global augmentations (resize jitter, random crop, color
    jitter) with deterministic RNG support for the new data pipeline.

Key Functions:
    - apply_global_augment(): Executes configured augmentations (resize jitter,
      random crop, color jitter) with optional seeded RNG.

CLI Arguments:
    (none)

Usage:
    frames = apply_global_augment(frames, cfg.get(\"global_aug\"), rng=random.Random(seed))
"""

from __future__ import annotations

import random
from typing import Mapping, Optional, Sequence, cast

import torch
import torch.nn.functional as F


def _resolve_hw(cfg: Mapping[str, object] | None, key: str, fallback: Sequence[int]) -> Optional[Sequence[int]]:
    if not isinstance(cfg, Mapping):
        return None
    value = cfg.get(key)
    if isinstance(value, Sequence) and len(value) >= 2:
        return [int(value[0]), int(value[1])]
    return fallback


def _get_float(cfg: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = cfg.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _color_jitter(frames: torch.Tensor, cj_cfg: Mapping[str, object], rng: random.Random) -> torch.Tensor:
    """Apply simple brightness/contrast jitter using seeded RNG."""
    out = frames
    brightness = _get_float(cj_cfg, "brightness", 0.0)
    contrast = _get_float(cj_cfg, "contrast", 0.0)
    if brightness > 0.0:
        factor = 1.0 + (rng.random() * 2.0 - 1.0) * brightness
        out = out * factor
    if contrast > 0.0:
        mean = out.mean(dim=(-2, -1), keepdim=True)
        factor = 1.0 + (rng.random() * 2.0 - 1.0) * contrast
        out = (out - mean) * factor + mean
    return out.clamp(0.0, 1.0)


def apply_global_augment(frames: torch.Tensor, aug_cfg: Mapping[str, object] | None = None, *, rng: Optional[random.Random] = None) -> torch.Tensor:
    """Apply resize jitter, random crop, and light color jitter if configured."""

    if not isinstance(aug_cfg, Mapping):
        return frames
    if not aug_cfg.get("enabled", False):
        return frames

    local_rng = rng or random.Random()
    resize_jitter = _resolve_hw(aug_cfg, "resize_jitter", frames.shape[-2:])
    crop_hw = _resolve_hw(aug_cfg, "random_crop_hw", frames.shape[-2:])
    color_jitter_cfg = aug_cfg.get("color_jitter")
    color_cfg: Mapping[str, object] = cast(Mapping[str, object], color_jitter_cfg) if isinstance(color_jitter_cfg, Mapping) else {}

    out = frames
    if resize_jitter and tuple(resize_jitter) != tuple(frames.shape[-2:]):
        out = F.interpolate(out, size=tuple(resize_jitter), mode="bilinear", align_corners=False)
    if crop_hw and len(crop_hw) >= 2:
        h, w = out.shape[-2:]
        crop_h, crop_w = int(crop_hw[0]), int(crop_hw[1])
        if crop_h < h and crop_w < w:
            y0 = local_rng.randint(0, h - crop_h)
            x0 = local_rng.randint(0, w - crop_w)
            out = out[..., y0 : y0 + crop_h, x0 : x0 + crop_w]
    if color_cfg:
        out = _color_jitter(out, color_cfg, local_rng)
    return out.clamp(0.0, 1.0)


__all__ = ["apply_global_augment"]
