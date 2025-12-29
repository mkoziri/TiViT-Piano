"""
Purpose:
    Provide simple tensor normalization helpers for the new data layout.

Key Functions:
    - normalize(): In-place channel-wise normalization with configurable mean/std.

CLI Arguments:
    (none)

Usage:
    frames = normalize(frames, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
"""

from __future__ import annotations

import torch
from typing import Sequence


def normalize(frames: torch.Tensor, mean: Sequence[float] = (0.5, 0.5, 0.5), std: Sequence[float] = (0.5, 0.5, 0.5)) -> torch.Tensor:
    """Normalize frames in-place."""

    if frames.ndim < 4:
        return frames
    mean_t = torch.as_tensor(mean, device=frames.device, dtype=frames.dtype).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, device=frames.device, dtype=frames.dtype).view(1, -1, 1, 1)
    frames.sub_(mean_t).div_(std_t.clamp(min=1e-6))
    return frames


__all__ = ["normalize"]
