"""Coordinate transforms for hand landmarks.

Purpose:
    - Map aligned hand landmarks into canonical keyboard space.
    - Normalize coordinates and masks for downstream supervision utilities.

Key Functions/Classes:
    - CanonicalLandmarks: Container for canonicalized points/masks.
    - map_landmarks_to_canonical(): Apply crop/registration to landmarks.

CLI Arguments:
    (none)

Usage:
    canonical = map_landmarks_to_canonical(aligned, registration=payload, source_hw=(H, W))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .pianovam_loader import AlignedHandLandmarks

RegistrationPayload = Mapping[str, Any]


@dataclass
class CanonicalLandmarks:
    """Canonicalised hand landmarks aligned to TiViT's keyboard space."""

    xy: torch.Tensor
    xy_norm: torch.Tensor
    mask: torch.Tensor
    metadata: Dict[str, Any]


def _parse_crop_meta(meta: Optional[Union[Sequence[float], Mapping[str, Any]]]) -> Optional[Tuple[float, float, float, float]]:
    if meta is None:
        return None
    if isinstance(meta, Mapping):
        keys = (
            ("min_y", "max_y", "min_x", "max_x"),
            ("top", "bottom", "left", "right"),
            ("y0", "y1", "x0", "x1"),
        )
        vals = None
        for candidate in keys:
            if all(k in meta for k in candidate):
                vals = [meta[k] for k in candidate]  # type: ignore[index]
                break
        if vals is None and "crop" in meta:
            cval = meta["crop"]
            if isinstance(cval, (list, tuple)) and len(cval) >= 4:
                vals = list(cval[:4])
        if vals is None:
            return None
    elif isinstance(meta, Sequence) and not isinstance(meta, (str, bytes)):
        if len(meta) < 4:
            return None
        vals = list(meta[:4])
    else:
        return None
    try:
        min_y, max_y, min_x, max_x = (float(v) for v in vals[:4])
    except (TypeError, ValueError):
        return None
    return (min_y, max_y, min_x, max_x)


def _resolve_crop_bounds(
    source_hw: Sequence[float],
    crop_meta: Optional[Union[Sequence[float], Mapping[str, Any]]],
) -> Tuple[float, float, float, float]:
    if len(source_hw) < 2:
        return (0.0, 0.0, 0.0, 0.0)

    h_src = float(source_hw[0])
    w_src = float(source_hw[1])

    parsed = _parse_crop_meta(crop_meta)
    if parsed is None:
        return (0.0, h_src, 0.0, w_src)

    min_y, max_y, min_x, max_x = parsed
    is_normalized = all(0.0 <= v <= 1.0 for v in (min_y, max_y, min_x, max_x))
    if is_normalized:
        min_y *= h_src
        max_y *= h_src
        min_x *= w_src
        max_x *= w_src

    y0 = max(0.0, min(min_y, max_y))
    y1 = min(max(min_y, max_y), h_src)
    x0 = max(0.0, min(min_x, max_x))
    x1 = min(max(min_x, max_x), w_src)

    if y1 <= y0:
        y1 = min(h_src, y0 + max(1.0, h_src * 0.01))
    if x1 <= x0:
        x1 = min(w_src, x0 + max(1.0, w_src * 0.01))

    return (y0, y1, x0, x1)


def _safe_homography_from_payload(registration: RegistrationPayload) -> Optional[torch.Tensor]:
    H_raw = registration.get("homography") if isinstance(registration, Mapping) else None
    if H_raw is None:
        return None
    try:
        H = torch.as_tensor(H_raw, dtype=torch.float32).reshape(3, 3)
    except Exception:
        return None
    if H.numel() != 9:
        return None
    return H


def _resolve_hw_from_registration(registration: RegistrationPayload, key: str) -> Optional[Tuple[int, int]]:
    val = registration.get(key) if isinstance(registration, Mapping) else None
    if not isinstance(val, (list, tuple)):
        return None
    if len(val) < 2:
        return None
    try:
        h, w = int(val[0]), int(val[1])
    except (TypeError, ValueError):
        return None
    if h <= 0 or w <= 0:
        return None
    return (h, w)


def _apply_x_warp(x: torch.Tensor, warp_ctrl: Optional[Union[torch.Tensor, np.ndarray]]) -> torch.Tensor:
    if warp_ctrl is None:
        return x
    ctrl = torch.as_tensor(warp_ctrl, dtype=torch.float32)
    if ctrl.numel() < 4 or ctrl.shape[-1] != 2:
        return x
    pre = ctrl[:, 0].cpu().numpy()
    post = ctrl[:, 1].cpu().numpy()
    order = np.argsort(pre)
    pre_sorted = pre[order]
    post_sorted = post[order]
    x_np = x.detach().cpu().numpy()
    warped = np.interp(x_np, pre_sorted, post_sorted, left=post_sorted[0], right=post_sorted[-1])
    return torch.from_numpy(warped).to(x)


def map_landmarks_to_canonical(
    aligned: AlignedHandLandmarks,
    *,
    registration: Optional[RegistrationPayload],
    source_hw: Sequence[int],
    crop_meta: Optional[Union[Sequence[float], Mapping[str, Any]]] = None,
    clamp: bool = True,
) -> CanonicalLandmarks:
    """Project clip-aligned landmarks into canonical keyboard coordinates."""

    base_mask = aligned.mask.clone()

    H = _safe_homography_from_payload(registration or {})
    canonical_hw = _resolve_hw_from_registration(registration or {}, "target_hw")
    reg_source_hw = _resolve_hw_from_registration(registration or {}, "source_hw")

    if H is None or canonical_hw is None or reg_source_hw is None:
        zeros = torch.zeros_like(aligned.landmarks)
        return CanonicalLandmarks(
            xy=zeros,
            xy_norm=zeros[..., :2],
            mask=torch.zeros_like(aligned.mask),
            metadata={"reason": "missing_registration"},
        )

    # Translate points into the cropped source coordinate system expected by H.
    y0, y1, x0, x1 = _resolve_crop_bounds(source_hw, crop_meta)
    crop_h = max(y1 - y0, 1.0)
    crop_w = max(x1 - x0, 1.0)
    scale_y = float(reg_source_hw[0]) / max(crop_h, 1e-6)
    scale_x = float(reg_source_hw[1]) / max(crop_w, 1e-6)

    pts = aligned.landmarks.clone()
    pts[..., 0] = (pts[..., 0] - float(x0)) * scale_x
    pts[..., 1] = (pts[..., 1] - float(y0)) * scale_y

    # Flatten for transform.
    flat_mask = base_mask.reshape(-1)
    flat_pts = pts.reshape(-1, 3)
    valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        zeros = torch.zeros_like(aligned.landmarks)
        return CanonicalLandmarks(
            xy=zeros,
            xy_norm=zeros[..., :2],
            mask=torch.zeros_like(aligned.mask),
            metadata={"reason": "no_valid_points"},
        )

    coords = flat_pts[valid_idx, :2]
    conf = flat_pts[valid_idx, 2:3]

    ones = torch.ones((coords.shape[0], 1), dtype=torch.float32, device=coords.device)
    homog = torch.cat([coords, ones], dim=1)
    H_t = H.to(coords.device, coords.dtype)
    mapped = homog @ H_t.transpose(0, 1)
    denom = mapped[:, 2].clamp(min=1e-6)
    mapped_xy = mapped[:, :2] / denom.unsqueeze(-1)

    if registration and isinstance(registration, Mapping):
        warp_ctrl = registration.get("x_warp_ctrl")
        if warp_ctrl is not None:
            mapped_xy[:, 0] = _apply_x_warp(mapped_xy[:, 0], warp_ctrl)

    if clamp:
        mapped_xy[:, 0] = mapped_xy[:, 0].clamp(0.0, float(canonical_hw[1] - 1))
        mapped_xy[:, 1] = mapped_xy[:, 1].clamp(0.0, float(canonical_hw[0] - 1))

    out_pts = torch.zeros_like(flat_pts)
    out_pts[valid_idx, :2] = mapped_xy
    out_pts[valid_idx, 2:3] = conf
    out_pts = out_pts.reshape_as(pts)

    mask = base_mask.clone()
    if clamp:
        in_bounds = (mapped_xy[:, 0] >= 0.0) & (mapped_xy[:, 0] <= float(canonical_hw[1] - 1))
        in_bounds &= (mapped_xy[:, 1] >= 0.0) & (mapped_xy[:, 1] <= float(canonical_hw[0] - 1))
        mask_flat = mask.reshape(-1)
        mask_flat[valid_idx] &= in_bounds
        mask = mask_flat.reshape_as(mask)

    xy_norm = out_pts[..., :2].clone()
    xy_norm[..., 0] = xy_norm[..., 0] / max(float(canonical_hw[1] - 1), 1e-6)
    xy_norm[..., 1] = xy_norm[..., 1] / max(float(canonical_hw[0] - 1), 1e-6)

    meta = {
        "applied_crop": crop_meta is not None,
        "crop_h": float(crop_h),
        "crop_w": float(crop_w),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
    }

    return CanonicalLandmarks(
        xy=out_pts,
        xy_norm=xy_norm,
        mask=mask,
        metadata=meta,
    )


__all__ = ["CanonicalLandmarks", "map_landmarks_to_canonical"]
