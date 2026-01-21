"""Hand reach mask utilities.

Purpose:
    - Compute per-frame reach probabilities over keys from hand landmarks.
    - Provide soft weighting masks for hand-gated losses.

Key Functions/Classes:
    - HandReachResult: Container for reach/valid/coverage outputs.
    - compute_hand_reach(): Generate a soft reach mask for each frame.

CLI Arguments:
    (none)

Usage:
    reach = compute_hand_reach(canonical, key_centers_norm=centers)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .coordinate_transforms import CanonicalLandmarks


@dataclass
class HandReachResult:
    reach: torch.Tensor
    valid: torch.Tensor
    coverage: float
    metadata: Dict[str, float]


def compute_hand_reach(
    canonical: CanonicalLandmarks,
    *,
    key_centers_norm: torch.Tensor,
    radius: float = 0.12,
    dilate: float = 0.02,
    min_points: int = 4,
) -> HandReachResult:
    """Derive a soft reach mask over keys for each frame."""

    T = canonical.xy_norm.shape[0]
    P = key_centers_norm.numel()
    reach = torch.ones((T, P), dtype=torch.float32, device=key_centers_norm.device)
    valid = torch.zeros((T,), dtype=torch.bool, device=key_centers_norm.device)

    expanded_radius = max(float(radius) + float(dilate), 1e-3)
    key_centers = key_centers_norm.to(device=canonical.xy_norm.device, dtype=torch.float32)

    for t in range(T):
        hand_x = []
        for h in range(2):
            mask = canonical.mask[t, h]
            if mask.sum() < int(min_points):
                continue
            x_vals = canonical.xy_norm[t, h, mask, 0]
            if x_vals.numel() == 0:
                continue
            hand_x.append(torch.median(x_vals))
        if not hand_x:
            continue
        valid[t] = True
        best = torch.zeros((len(hand_x), P), device=key_centers.device)
        for idx, hx in enumerate(hand_x):
            dist = (key_centers - hx).abs()
            best[idx] = torch.clamp(1.0 - dist / expanded_radius, min=0.0, max=1.0)
        reach_frame = best.max(dim=0).values if hand_x else torch.ones((P,), device=key_centers.device)
        reach[t] = reach_frame

    coverage = float(valid.float().mean().item()) if T > 0 else 0.0
    meta = {
        "radius": float(radius),
        "dilate": float(dilate),
        "min_points": float(min_points),
        "coverage": coverage,
    }
    return HandReachResult(reach=reach, valid=valid, coverage=coverage, metadata=meta)


__all__ = ["HandReachResult", "compute_hand_reach"]
