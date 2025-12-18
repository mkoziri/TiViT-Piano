"""Hand reach mask utilities.

Compute per-frame reach probabilities over keys based on canonicalised hand
positions. Outputs remain soft so downstream losses/priors can use them as
weights instead of hard suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .coordinate_transforms import CanonicalLandmarks


@dataclass
class HandReachResult:
    reach: torch.Tensor        # (T, P) float in [0,1], 1 means reachable
    valid: torch.Tensor        # (T,) bool; False -> reach is all ones (disabled)
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
    """
    Derive a soft reach mask over keys for each frame.

    Args:
        canonical: Canonicalised landmarks/mask for a clip.
        key_centers_norm: (P,) normalized [0,1] key centers in canonical space.
        radius: Base reach radius (normalized units).
        dilate: Extra margin to avoid over-suppressing.
        min_points: Minimum valid keypoints to trust a hand position.

    Returns:
        HandReachResult with ``reach`` in [0,1], ``valid`` mask per frame, and
        coverage fraction (frames with valid hand evidence).
    """

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
