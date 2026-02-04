"""Imbalance helpers shared across preprocessing and loss weighting."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch


def sanitize_ratio(values: np.ndarray) -> np.ndarray:
    """Replace non-finite ratios with the max finite value."""
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    max_val = arr[finite].max()
    return np.where(finite, arr, max_val)


def map_ratio_to_band(
    ratio: torch.Tensor | np.ndarray | Sequence[float],
    band: Tuple[float, float],
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map per-class neg/pos ratios into a configured band."""
    ratio_t = torch.as_tensor(ratio, dtype=torch.float32)
    low, high = band
    if high < low:
        low, high = high, low
    span = ratio_t.max() - ratio_t.min()
    if span <= eps:
        mapped = torch.full_like(ratio_t, 0.5 * (low + high))
    else:
        mapped = (ratio_t - ratio_t.min()) / span.clamp_min(eps)
        mapped = low + mapped * (high - low)
    return mapped.clamp(min=min(low, high), max=max(low, high))
