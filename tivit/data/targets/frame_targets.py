"""
Purpose:
    Wrap frame target configuration, soft targets, and cache-aware building
    for the new data layout. Keeps targets lag-agnostic by consuming aligned
    events and delegating to the shared frame target logic.

Key Functions/Classes:
    - build_frame_target_spec(): Normalize config into FrameTargetSpec.
    - build_soft_target_cfg(): Resolve optional soft targets.
    - build_frame_targets(): Construct/cache targets for a clip.
    - resolve_lag_ms: Utility re-export for lag handling.

CLI Arguments:
    (none)

Usage:
    spec = build_frame_target_spec(cfg, frames=96, stride=2, fps=30.0, canonical_hw=(180,1536))
    result = build_frame_targets(labels=events, lag_result=None, spec=spec, cache=cache, split=\"train\", video_id=\"vid\", clip_start=0.0)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import torch

from utils.frame_target_cache import FrameTargetCache
from utils.frame_targets import (
    FrameTargetResult,
    FrameTargetSpec,
    SoftTargetConfig,
    prepare_frame_targets,
    resolve_frame_target_spec,
    resolve_soft_target_config,
    resolve_lag_ms,
)
from utils.av_sync import AVLagResult


def build_frame_target_spec(frame_cfg: Mapping[str, Any], *, frames: int, stride: int, fps: float, canonical_hw: Sequence[int]) -> Optional[FrameTargetSpec]:
    """Normalise config into FrameTargetSpec or return None when disabled."""

    return resolve_frame_target_spec(frame_cfg, frames=frames, stride=stride, fps=fps, canonical_hw=canonical_hw)


def build_soft_target_cfg(frame_cfg: Mapping[str, Any]) -> Optional[SoftTargetConfig]:
    """Return soft-target configuration when enabled."""

    return resolve_soft_target_config(frame_cfg)


def build_frame_targets(
    *,
    labels: Optional[torch.Tensor],
    lag_result: Optional[AVLagResult],
    spec: FrameTargetSpec,
    cache: FrameTargetCache,
    split: str,
    video_id: str,
    clip_start: float,
    soft_targets: Optional[SoftTargetConfig] = None,
) -> FrameTargetResult:
    """Load or construct frame targets for a clip using shared pipeline."""

    return prepare_frame_targets(
        labels=labels,
        lag_result=lag_result,
        spec=spec,
        cache=cache,
        split=split,
        video_id=video_id,
        clip_start=clip_start,
        soft_targets=soft_targets,
    )


__all__ = [
    "FrameTargetResult",
    "FrameTargetSpec",
    "SoftTargetConfig",
    "build_frame_target_spec",
    "build_soft_target_cfg",
    "build_frame_targets",
    "resolve_lag_ms",
]
