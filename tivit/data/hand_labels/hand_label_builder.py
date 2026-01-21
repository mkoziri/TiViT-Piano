"""Build supervision signals from canonicalised hand landmarks.

Purpose:
    - Assign left/right hand labels to onset events using landmark geometry.
    - Extract key-center positions from registration metadata for supervision.

Key Functions/Classes:
    - EventHandLabelConfig: Configuration for event-level labeling.
    - build_event_hand_labels(): Produce labels and validity masks.
    - key_centers_from_geometry(): Extract normalized key centers.

CLI Arguments:
    (none)

Usage:
    labels = build_event_hand_labels(onsets_sec=onsets, key_indices=keys, key_centers_norm=centers, frame_times=t, canonical=canon, config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .coordinate_transforms import CanonicalLandmarks


@dataclass
class EventHandLabelConfig:
    """Configuration for assigning left/right to onset events."""

    time_tolerance: float
    max_dx: float = 0.12
    min_points: int = 4
    unknown_class: int = 2


@dataclass
class EventHandLabels:
    """Event-level hand targets plus validity mask."""

    labels: torch.Tensor
    mask: torch.Tensor
    coverage: float
    metadata: Dict[str, Any]


def key_centers_from_geometry(
    geometry_meta: Optional[Dict[str, Any]],
    *,
    fallback_keys: int = 88,
) -> Optional[torch.Tensor]:
    """Extract normalized key-center positions from registration geometry metadata."""

    if not isinstance(geometry_meta, dict):
        return None
    key_bounds = geometry_meta.get("key_bounds_px")
    target_hw = geometry_meta.get("target_hw")
    if not isinstance(key_bounds, (list, tuple)) or not isinstance(target_hw, (list, tuple)):
        return None
    if len(target_hw) < 2 or len(key_bounds) == 0:
        return None
    try:
        width = float(target_hw[1])
    except (TypeError, ValueError):
        return None
    if width <= 1.0:
        return None

    centers: list[float] = []
    for bound in key_bounds:
        if not isinstance(bound, (list, tuple)) or len(bound) < 2:
            continue
        try:
            left, right = float(bound[0]), float(bound[1])
        except (TypeError, ValueError):
            continue
        centers.append(0.5 * (left + right) / max(width - 1.0, 1e-6))
    if not centers:
        return torch.linspace(0.0, 1.0, fallback_keys, dtype=torch.float32)
    return torch.tensor(centers, dtype=torch.float32)


def _hand_x_positions(
    canon: CanonicalLandmarks,
    *,
    min_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute representative x position per hand/frame in normalized coords."""

    T = canon.xy_norm.shape[0]
    hand_x = torch.full((T, 2), float("nan"), dtype=torch.float32, device=canon.xy_norm.device)
    hand_valid = torch.zeros((T, 2), dtype=torch.bool, device=canon.xy_norm.device)

    for t in range(T):
        for h in range(2):
            valid_pts = canon.mask[t, h]
            if valid_pts.sum() < int(min_points):
                continue
            x_vals = canon.xy_norm[t, h, valid_pts, 0]
            if x_vals.numel() == 0:
                continue
            hand_x[t, h] = torch.median(x_vals)
            hand_valid[t, h] = True

    return hand_x, hand_valid


def build_event_hand_labels(
    *,
    onsets_sec: torch.Tensor,
    key_indices: torch.Tensor,
    key_centers_norm: torch.Tensor,
    frame_times: torch.Tensor,
    canonical: CanonicalLandmarks,
    config: EventHandLabelConfig,
) -> EventHandLabels:
    """Assign left/right labels to onset events based on canonical landmarks."""

    if onsets_sec.numel() == 0 or key_indices.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.long, device=onsets_sec.device)
        return EventHandLabels(
            labels=empty,
            mask=empty.to(dtype=torch.bool),
            coverage=0.0,
            metadata={"reason": "no_events"},
        )

    if onsets_sec.shape != key_indices.shape:
        raise ValueError(
            f"onsets_sec shape {onsets_sec.shape} does not match key_indices {key_indices.shape}"
        )

    hand_x, hand_valid = _hand_x_positions(canonical, min_points=config.min_points)
    N = onsets_sec.shape[0]
    labels = torch.full((N,), int(config.unknown_class), dtype=torch.long, device=onsets_sec.device)
    mask = torch.zeros((N,), dtype=torch.bool, device=onsets_sec.device)

    for idx in range(N):
        onset = float(onsets_sec[idx].item())
        key_idx = int(key_indices[idx].item())
        if key_idx < 0 or key_idx >= key_centers_norm.numel():
            continue
        key_x = float(key_centers_norm[key_idx].item())

        delta = torch.abs(frame_times - onset)
        frame_idx = int(delta.argmin().item())
        gap = float(delta[frame_idx].item())
        if gap > float(config.time_tolerance):
            continue

        if frame_idx >= hand_x.shape[0]:
            continue

        left_ok = bool(hand_valid[frame_idx, 0].item())
        right_ok = bool(hand_valid[frame_idx, 1].item())
        if not (left_ok or right_ok):
            continue

        best_hand = None
        best_dx = None
        if left_ok:
            dx_left = abs(float(hand_x[frame_idx, 0].item()) - key_x)
            best_hand = 0
            best_dx = dx_left
        if right_ok:
            dx_right = abs(float(hand_x[frame_idx, 1].item()) - key_x)
            if best_dx is None or dx_right < best_dx:
                best_hand = 1
                best_dx = dx_right

        if best_dx is None or best_hand is None:
            continue
        if best_dx > float(config.max_dx):
            continue

        labels[idx] = int(best_hand)
        mask[idx] = True

    coverage = float(mask.float().mean().item()) if mask.numel() > 0 else 0.0
    meta = {
        "coverage": coverage,
        "valid_events": int(mask.sum().item()),
        "total_events": int(mask.numel()),
        "config": {
            "time_tolerance": float(config.time_tolerance),
            "max_dx": float(config.max_dx),
            "min_points": int(config.min_points),
        },
    }
    return EventHandLabels(labels=labels, mask=mask, coverage=coverage, metadata=meta)


__all__ = ["EventHandLabelConfig", "EventHandLabels", "build_event_hand_labels", "key_centers_from_geometry"]
