"""
Registration refinement utilities for per-video keyboard rectification.

This module estimates a refined homography that maps cropped piano keyboard
frames to the canonical resolution (e.g. 145x800).  The refinement is driven by
edge detections aggregated across a handful of frames sampled throughout each
video.  Results are cached on disk so repeated runs avoid recomputation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .identifiers import canonical_video_id

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_HW_LOGGED: Set[Tuple[int, int]] = set()
_CACHE_SUMMARY_LOGGED: Set[Tuple[Tuple[int, int], Path]] = set()

_MIDI_LOW = 21
_MIDI_HIGH = 108
_WHITE_PITCHES = {0, 2, 4, 5, 7, 9, 11}
_KEY_COUNT = _MIDI_HIGH - _MIDI_LOW + 1
_DEFAULT_META_TILES = 3


def _midi_is_white(midi: int) -> bool:
    return (midi % 12) in _WHITE_PITCHES


def _key_bounds_from_white_edges(edges: np.ndarray, width: float) -> Optional[List[List[float]]]:
    if edges.ndim != 1 or edges.size < 2:
        return None
    sorted_edges = np.sort(edges.astype(np.float32, copy=True))
    white_segments = sorted_edges.size - 1
    if white_segments < 52:
        return None
    width = float(width)
    bounds: List[List[float]] = []
    white_idx = 0
    for key_idx in range(_KEY_COUNT):
        midi = _MIDI_LOW + key_idx
        if _midi_is_white(midi):
            if white_idx + 1 >= sorted_edges.size:
                return None
            left = float(np.clip(sorted_edges[white_idx], 0.0, width))
            right = float(np.clip(sorted_edges[white_idx + 1], 0.0, width))
            if right < left:
                left, right = right, left
            bounds.append([left, right])
            white_idx += 1
        else:
            left_white = max(white_idx - 1, 0)
            right_white = min(white_idx, sorted_edges.size - 2)
            left_center = 0.5 * (sorted_edges[left_white] + sorted_edges[left_white + 1])
            right_center = 0.5 * (sorted_edges[right_white] + sorted_edges[right_white + 1])
            center = 0.5 * (left_center + right_center)
            width_left = sorted_edges[left_white + 1] - sorted_edges[left_white]
            width_right = sorted_edges[right_white + 1] - sorted_edges[right_white]
            base_width = float(max(min(width_left, width_right), 1e-3))
            key_width = 0.6 * base_width
            left = float(np.clip(center - key_width / 2.0, 0.0, width))
            right = float(np.clip(center + key_width / 2.0, 0.0, width))
            if right < left:
                left, right = right, left
            bounds.append([left, right])
    return bounds


def _build_geometry_metadata_from_edges(edges: np.ndarray, width: float, canonical_hw: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    key_bounds = _key_bounds_from_white_edges(edges, width)
    if key_bounds is None:
        return None
    tile_bounds = [
        (float(lo) * width, float(hi) * width) for lo, hi in _uniform_bounds(_DEFAULT_META_TILES)
    ]
    return {
        "rectified_width": float(width),
        "key_bounds_px": key_bounds,
        "target_hw": [int(canonical_hw[0]), int(canonical_hw[1])],
        "tile_bounds_px": tile_bounds,
    }


def _canonical_white_edges(width: float) -> np.ndarray:
    return np.linspace(0.0, float(width), num=52 + 1, dtype=np.float32)


def _apply_warp_ctrl(edges: np.ndarray, warp_ctrl: Optional[np.ndarray]) -> np.ndarray:
    if warp_ctrl is None or warp_ctrl.size < 4:
        return edges
    ctrl = np.asarray(warp_ctrl, dtype=np.float32)
    if ctrl.ndim != 2 or ctrl.shape[1] != 2:
        return edges
    order = np.argsort(ctrl[:, 0])
    pre = ctrl[order, 0]
    post = ctrl[order, 1]
    warped = np.interp(edges, pre, post, left=post[0], right=post[-1])
    return warped.astype(np.float32, copy=False)


def _uniform_bounds(num_tiles: int) -> List[Tuple[float, float]]:
    if num_tiles <= 0:
        return []
    edges = np.linspace(0.0, 1.0, num_tiles + 1, dtype=np.float32)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(num_tiles)]


try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None  # type: ignore

try:
    import decord  # type: ignore

    _HAVE_DECORD = True
except Exception:  # pragma: no cover - decord optional
    decord = None  # type: ignore
    _HAVE_DECORD = False


def _extract_crop_values(
    meta: Optional[Sequence[float] | Dict[str, Any]],
) -> Optional[Tuple[float, float, float, float]]:
    if meta is None:
        return None
    if isinstance(meta, dict):
        keys = ("min_y", "max_y", "min_x", "max_x")
        vals: Optional[List[float]] = None
        for candidate in (keys, ("top", "bottom", "left", "right"), ("y0", "y1", "x0", "x1")):
            if all(k in meta for k in candidate):
                vals = [meta[k] for k in candidate]  # type: ignore[index]
                break
        if vals is None and "crop" in meta:
            crop_val = meta["crop"]
            if isinstance(crop_val, (list, tuple)) and len(crop_val) >= 4:
                vals = list(crop_val[:4])
        if vals is None:
            return None
    elif isinstance(meta, Sequence):
        if len(meta) < 4:
            return None
        vals = list(meta[:4])
    else:
        return None

    try:
        min_y, max_y, min_x, max_x = vals[:4]
        return (
            float(min_y),
            float(max_y),
            float(min_x),
            float(max_x),
        )
    except (TypeError, ValueError):
        return None


def _apply_crop_np(frame: np.ndarray, meta: Optional[Sequence[float] | Dict[str, Any]]) -> np.ndarray:
    coords = _extract_crop_values(meta)
    if coords is None:
        return frame

    min_y, max_y, min_x, max_x = coords
    h, w = frame.shape[:2]
    is_normalised = all(0.0 <= v <= 1.0 for v in (min_y, max_y, min_x, max_x))
    if is_normalised:
        min_y *= h
        max_y *= h
        min_x *= w
        max_x *= w

    y0 = int(math.floor(min_y))
    y1 = int(math.ceil(max_y))
    x0 = int(math.floor(min_x))
    x1 = int(math.ceil(max_x))

    y0 = max(0, min(y0, h - 1))
    x0 = max(0, min(x0, w - 1))
    y1 = max(y0 + 1, min(y1, h))
    x1 = max(x0 + 1, min(x1, w))
    return frame[y0:y1, x0:x1]


def _median_or_none(values: Sequence[float]) -> Optional[float]:
    arr = [float(v) for v in values if math.isfinite(float(v))]
    if not arr:
        return None
    arr.sort()
    mid = len(arr) // 2
    if len(arr) % 2 == 1:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def _detect_baseline(gray: np.ndarray) -> Optional[Tuple[float, float]]:
    if cv2 is None:
        return None
    edges = cv2.Canny(gray, 80, 160, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLines(edges, 1.0, np.pi / 180.0, threshold=max(80, int(gray.shape[1] * 0.4)))
    if lines is None:
        return None

    best: Optional[Tuple[float, float]] = None
    best_votes = -1
    for candidate in lines[:, 0]:
        rho, theta = float(candidate[0]), float(candidate[1])
        # Horizontal line corresponds to theta approx pi/2
        if abs(theta - (math.pi / 2.0)) > math.radians(25.0):
            continue
        votes = 1
        if best is None or votes > best_votes:
            best = (rho, theta)
            best_votes = votes

    if best is None:
        return None

    rho, theta = best
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    if abs(sin_t) < 1e-6:
        return None

    slope = -cos_t / sin_t
    intercept = rho / sin_t
    return slope, intercept


def _gaussian_smooth_1d(data: np.ndarray, kernel: int = 9) -> np.ndarray:
    if cv2 is None or data.size == 0:
        return data.astype(np.float32)
    kernel = max(3, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    smoothed = cv2.GaussianBlur(data.reshape(1, -1), (kernel, 1), 0, borderType=cv2.BORDER_REFLECT)
    return smoothed.reshape(-1).astype(np.float32)


def _find_peak(profile: np.ndarray, center: float, window: int) -> float:
    """Return the index of the strongest response around ``center``."""
    length = profile.shape[0]
    if length == 0:
        return float(center)
    left = int(max(0, math.floor(center - window)))
    right = int(min(length, math.ceil(center + window)))
    if right <= left:
        return float(center)
    segment = profile[left:right]
    if segment.size == 0:
        return float(center)
    idx = int(np.argmax(segment))
    return float(left + idx)


def _snap_edges_ransac(edges: np.ndarray, expected_spacing: float) -> np.ndarray:
    """Project detected edges onto a regular grid while limiting local drift."""
    if edges.size < 4:
        return edges
    indices = np.arange(edges.size, dtype=np.float32)
    spacing = max(float(expected_spacing), 1.0)
    inlier_thresh = max(spacing * 0.35, 1.5)
    rng = np.random.default_rng(12345)

    best_inliers: Optional[np.ndarray] = None
    best_count = -1
    best_error = float("inf")

    for _ in range(120):
        sample = rng.choice(edges.size, size=2, replace=False)
        i0, i1 = int(sample[0]), int(sample[1])
        denom = float(indices[i1] - indices[i0])
        if abs(denom) < 1e-6:
            continue
        slope = float((edges[i1] - edges[i0]) / denom)
        intercept = float(edges[i0] - slope * indices[i0])
        fitted = slope * indices + intercept
        residuals = np.abs(fitted - edges)
        inliers = residuals <= inlier_thresh
        count = int(np.count_nonzero(inliers))
        if count < 4:
            continue
        error = float(residuals[inliers].mean()) if count > 0 else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_inliers = inliers
            best_count = count
            best_error = error

    if best_inliers is None:
        slope, intercept = np.polyfit(indices, edges, deg=1)
    else:
        slope, intercept = np.polyfit(indices[best_inliers], edges[best_inliers], deg=1)

    fitted = slope * indices + intercept
    residuals = edges - fitted
    clip_val = max(spacing * 0.6, 3.0)
    residuals = np.clip(residuals, -clip_val, clip_val)
    refined = fitted + residuals
    refined = np.maximum.accumulate(refined)
    return refined.astype(np.float32)


def _compute_keyboard_edges(
    grad_profile: np.ndarray,
    width: int,
    num_white_keys: int = 52,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate white-key edge locations across the keyboard span."""
    if grad_profile.size == 0:
        indices = np.linspace(0.0, float(width - 1), num_white_keys + 1, dtype=np.float32)
        strengths = np.ones_like(indices, dtype=np.float32)
        return indices, strengths, float(indices[0]), float(indices[-1])

    profile = _gaussian_smooth_1d(grad_profile.astype(np.float32), kernel=9)
    if profile.max(initial=0.0) <= 1e-6:
        indices = np.linspace(0.0, float(width - 1), num_white_keys + 1, dtype=np.float32)
        strengths = np.ones_like(indices, dtype=np.float32)
        return indices, strengths, float(indices[0]), float(indices[-1])

    threshold = float(profile.max() * 0.2)
    active = np.where(profile >= threshold)[0]
    if active.size == 0:
        x_left = 0.0
        x_right = float(width - 1)
    else:
        x_left = float(active[0])
        x_right = float(active[-1])
        if (x_right - x_left) < width * 0.6:
            x_left = 0.0
            x_right = float(width - 1)

    span = max(x_right - x_left, 1.0)
    step = span / float(num_white_keys)
    window = max(5.0, step * 0.35)
    edges = []
    strengths = []
    for idx in range(num_white_keys + 1):
        center = x_left + step * idx
        peak = _find_peak(profile, center, int(round(window)))
        peak_clamped = int(np.clip(round(peak), 0, profile.shape[0] - 1))
        edges.append(peak)
        strengths.append(float(profile[peak_clamped]))
    edge_arr = np.array(edges, dtype=np.float32)
    edge_arr = np.clip(edge_arr, 0.0, float(width - 1))
    edge_arr = _snap_edges_ransac(edge_arr, step)
    edge_arr = np.clip(edge_arr, 0.0, float(width - 1))
    strength_arr = np.array(strengths, dtype=np.float32)
    if strength_arr.size != edge_arr.size:
        strength_arr = np.ones_like(edge_arr, dtype=np.float32)
    return edge_arr, strength_arr, float(edge_arr[0]), float(edge_arr[-1])


_BLACK_KEY_INTERVAL_PATTERN: Tuple[int, ...] = (1, 0, 1, 1, 0, 1, 1)
WHITE_ANCHOR_PRIORITY: float = 0.7
BLACK_ANCHOR_PRIORITY: float = 0.3
X_WARP_ERR_THRESHOLD: float = 2.5


def _compute_black_key_gaps(
    avg_frame: np.ndarray,
    baseline_row: int,
    keyboard_height: float,
    edges: np.ndarray,
    *,
    min_margin: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect black-key gap anchors using intensity minima."""
    if avg_frame.ndim == 3:
        avg_frame = np.mean(avg_frame, axis=-1)
    avg_frame = np.asarray(avg_frame, dtype=np.float32)
    if avg_frame.ndim != 2 or avg_frame.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    height, width = avg_frame.shape
    if width <= 1 or height <= 1:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    if keyboard_height <= 1.0:
        keyboard_height = float(height) * 0.65
    baseline_row = int(np.clip(baseline_row, 0, height - 1))

    top = int(max(0, baseline_row - int(round(keyboard_height * 0.95))))
    mid = int(max(top + 1, baseline_row - int(round(keyboard_height * 0.4))))
    if mid <= top:
        mid = min(height, top + max(1, int(round(keyboard_height * 0.3))))
    roi = avg_frame[top:mid, :]
    if roi.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    profile = roi.mean(axis=0)
    response = profile.max(initial=0.0) - profile
    if response.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    response = _gaussian_smooth_1d(response.astype(np.float32), kernel=11)

    positions: List[float] = []
    strengths: List[float] = []
    boundaries: List[int] = []

    num_edges = int(edges.shape[0])
    if num_edges < 3:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    pattern = _BLACK_KEY_INTERVAL_PATTERN
    pattern_len = len(pattern)

    for boundary in range(1, num_edges - 1):
        if not pattern[(boundary - 1) % pattern_len]:
            continue
        left_span = float(edges[boundary] - edges[boundary - 1])
        right_span = float(edges[boundary + 1] - edges[boundary])
        span_left = max(left_span * (1.0 - min_margin), left_span * 0.35)
        span_right = max(right_span * (1.0 - min_margin), right_span * 0.35)
        search_left = int(max(0.0, math.floor(edges[boundary] - span_left)))
        search_right = int(min(width - 1.0, math.ceil(edges[boundary] + span_right)))
        if search_right <= search_left:
            continue
        window = response[search_left:search_right]
        if window.size == 0:
            continue
        idx_rel = int(np.argmax(window))
        peak_col = float(search_left + idx_rel)
        strength = float(window[idx_rel])
        positions.append(peak_col)
        strengths.append(strength)
        boundaries.append(boundary)

    if not positions:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    return (
        np.array(positions, dtype=np.float32),
        np.array(strengths, dtype=np.float32),
        np.array(boundaries, dtype=np.int32),
    )


def _canonical_black_key_positions(
    canon_edges: np.ndarray, boundary_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if canon_edges.ndim != 1 or canon_edges.size < 3 or boundary_indices.size == 0:
        empty = np.zeros((0,), dtype=np.float32)
        mask = np.zeros((0,), dtype=bool)
        return empty, mask
    canon = np.asarray(canon_edges, dtype=np.float32)
    boundary_indices = np.asarray(boundary_indices, dtype=np.int32)
    mask = (boundary_indices > 0) & (boundary_indices < canon.shape[0] - 1)
    if not np.any(mask):
        empty = np.zeros((0,), dtype=np.float32)
        return empty, mask
    valid_indices = boundary_indices[mask]
    centers = 0.5 * (canon[valid_indices - 1] + canon[valid_indices + 1])
    return centers.astype(np.float32), mask


def _apply_homography_to_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homog = np.hstack([pts, ones]).T
    mapped = matrix @ homog
    denom = np.maximum(mapped[2], 1e-6)
    x = mapped[0] / denom
    y = mapped[1] / denom
    return np.stack([x, y], axis=1)


def _build_x_warp_controls(anchor_src: np.ndarray, anchor_dst: np.ndarray, width: int) -> np.ndarray:
    if anchor_src.size == 0 or anchor_dst.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    src = np.asarray(anchor_src, dtype=np.float32).ravel()
    dst = np.asarray(anchor_dst, dtype=np.float32).ravel()
    valid = np.isfinite(src) & np.isfinite(dst)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32)
    src = src[valid]
    dst = dst[valid]
    width_f = float(max(width - 1, 1))
    src = np.clip(src, 0.0, width_f)
    dst = np.clip(dst, 0.0, width_f)

    ctrl_src = np.concatenate(([0.0], src, [width_f]))
    ctrl_dst = np.concatenate(([0.0], dst, [width_f]))
    order = np.argsort(ctrl_src)
    ctrl_src = ctrl_src[order]
    ctrl_dst = ctrl_dst[order]
    ctrl_dst = np.clip(ctrl_dst, 0.0, width_f)
    for _ in range(2):
        if ctrl_dst.size <= 4:
            break
        smooth = 0.25 * ctrl_dst[:-2] + 0.5 * ctrl_dst[1:-1] + 0.25 * ctrl_dst[2:]
        ctrl_dst[1:-1] = smooth
        ctrl_dst = np.clip(ctrl_dst, 0.0, width_f)
        ctrl_dst = np.maximum.accumulate(ctrl_dst)
    ctrl_dst = np.clip(ctrl_dst, 0.0, width_f)
    ctrl_dst[0] = 0.0
    ctrl_dst[-1] = width_f
    key_width = width_f / 52.0
    max_delta = key_width * 0.5
    deltas = np.clip(ctrl_dst - ctrl_src, -max_delta, max_delta)
    deltas[0] = 0.0
    deltas[-1] = 0.0
    ctrl = np.stack([ctrl_src, ctrl_src + deltas], axis=-1)
    ctrl[:, 1] = np.clip(ctrl[:, 1], 0.0, width_f)
    ctrl[:, 1] = np.maximum.accumulate(ctrl[:, 1])
    ctrl[0, 1] = 0.0
    ctrl[-1, 1] = width_f
    return ctrl.astype(np.float32)


def _aggregate_gradients(frames: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    grad_x = None
    grad_y = None
    for frame in frames:
        gray = frame
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY) if cv2 is not None else np.mean(gray, axis=-1)
        if cv2 is not None:
            gray = cv2.medianBlur(gray.astype(np.uint8), 5)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        else:
            grayf = gray.astype(np.float32)
            gx = np.abs(np.gradient(grayf, axis=1))
            gy = np.abs(np.gradient(grayf, axis=0))
        gx = np.abs(gx).sum(axis=0)
        gy = np.abs(gy).sum(axis=1)
        grad_x = gx if grad_x is None else grad_x + gx
        grad_y = gy if grad_y is None else grad_y + gy
    if grad_x is None:
        grad_x = np.zeros(frames[0].shape[1], dtype=np.float32)
    if grad_y is None:
        grad_y = np.zeros(frames[0].shape[0], dtype=np.float32)
    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def _estimate_keyboard_height(
    grad_y_profile: np.ndarray,
    baseline_row: int,
    height: int,
) -> int:
    if baseline_row <= 0 or grad_y_profile.size == 0:
        return max(int(height * 0.65), 1)
    smoothed = _gaussian_smooth_1d(grad_y_profile.astype(np.float32), kernel=9)
    cutoff = max(0, baseline_row - 3)
    if cutoff <= 0:
        return max(int(height * 0.65), 1)
    upper = smoothed[:cutoff]
    if upper.size == 0:
        return max(int(height * 0.65), 1)
    top_idx = int(np.argmax(upper))
    est = baseline_row - top_idx
    low = max(int(height * 0.35), 1)
    high = max(int(height * 0.9), low + 1)
    return int(min(max(est, low), high))


def _homography_to_grid(
    homography: np.ndarray,
    source_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    warp_ctrl: Optional[np.ndarray] = None,
) -> torch.Tensor:
    h_dst, w_dst = target_hw
    h_src, w_src = source_hw
    ys = np.linspace(0.0, float(h_dst - 1), h_dst, dtype=np.float32)
    xs = np.linspace(0.0, float(w_dst - 1), w_dst, dtype=np.float32)
    xs_pre = xs
    if warp_ctrl is not None and warp_ctrl.size >= 4:
        ctrl = np.asarray(warp_ctrl, dtype=np.float32)
        pre = ctrl[:, 0]
        post = ctrl[:, 1]
        order = np.argsort(post)
        post_sorted = post[order]
        pre_sorted = np.clip(pre[order], 0.0, float(w_dst - 1))
        xs_pre = np.interp(xs, post_sorted, pre_sorted, left=pre_sorted[0], right=pre_sorted[-1])
        xs_pre = np.clip(xs_pre, 0.0, float(w_dst - 1))
    xv, yv = np.meshgrid(xs_pre, ys)
    ones = np.ones_like(xv)
    coords = np.stack([xv, yv, ones], axis=-1).reshape(-1, 3).T
    mapped = homography @ coords
    denom = mapped[2] + 1e-6
    x_src = (mapped[0] / denom).reshape(h_dst, w_dst)
    y_src = (mapped[1] / denom).reshape(h_dst, w_dst)
    w_src_f = max(float(w_src), 1.0)
    h_src_f = max(float(h_src), 1.0)
    x_norm = ((x_src + 0.5) / w_src_f) * 2.0 - 1.0
    y_norm = ((y_src + 0.5) / h_src_f) * 2.0 - 1.0
    grid = np.stack(
        [
            np.clip(x_norm, -2.0, 2.0),
            np.clip(y_norm, -2.0, 2.0),
        ],
        axis=-1,
    )
    return torch.from_numpy(grid.astype(np.float32))


def _homography_to_vector(matrix: np.ndarray) -> np.ndarray:
    """Flatten a homography into an 8D vector with bottom-right entry normalised."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError(f"expected homography shape (3, 3); got {arr.shape}")
    scale = float(arr[2, 2]) if abs(float(arr[2, 2])) > 1e-6 else 1.0
    arr = arr / scale
    return np.array(
        [
            float(arr[0, 0]),
            float(arr[0, 1]),
            float(arr[0, 2]),
            float(arr[1, 0]),
            float(arr[1, 1]),
            float(arr[1, 2]),
            float(arr[2, 0]),
            float(arr[2, 1]),
        ],
        dtype=np.float32,
    )


def _solve_regularized_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    base_h: np.ndarray,
    *,
    reg_lambda: float = 0.05,
    weights: Optional[Sequence[float]] = None,
) -> Optional[np.ndarray]:
    """Solve a least-squares homography with Tikhonov regularisation."""
    if src_pts.shape[0] < 4:
        return None

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    n = src.shape[0]
    A = np.zeros((2 * n, 8), dtype=np.float32)
    b = np.zeros((2 * n,), dtype=np.float32)
    weight_arr: Optional[np.ndarray] = None
    if weights is not None:
        weight_arr = np.asarray(list(weights), dtype=np.float32).reshape(-1)
        if weight_arr.shape[0] != n:
            raise ValueError("weights must match number of points")
        min_w = float(np.max(weight_arr)) * 1e-3 if np.max(weight_arr, initial=0.0) > 0 else 0.0
        weight_arr = np.clip(weight_arr, min_w, None)

    for i in range(n):
        x, y = float(src[i, 0]), float(src[i, 1])
        u, v = float(dst[i, 0]), float(dst[i, 1])
        row = 2 * i
        A[row, 0:3] = [x, y, 1.0]
        A[row, 6:8] = [-u * x, -u * y]
        b[row] = u
        A[row + 1, 3:6] = [x, y, 1.0]
        A[row + 1, 6:8] = [-v * x, -v * y]
        b[row + 1] = v
        if weight_arr is not None:
            w = math.sqrt(float(weight_arr[i]))
            A[row : row + 2] *= w
            b[row : row + 2] *= w

    if reg_lambda > 0.0:
        lam = float(reg_lambda)
        sqrt_lam = math.sqrt(lam)
        A_reg = sqrt_lam * np.eye(8, dtype=np.float32)
        b_reg = sqrt_lam * _homography_to_vector(base_h)
        A = np.vstack([A, A_reg])
        b = np.concatenate([b, b_reg])

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    H = np.array(
        [
            [sol[0], sol[1], sol[2]],
            [sol[3], sol[4], sol[5]],
            [sol[6], sol[7], 1.0],
        ],
        dtype=np.float32,
    )
    return H


def _invert_homography(matrix: Any) -> np.ndarray:
    """Return the inverse of a 3x3 homography as float32 array."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError(f"expected homography shape (3, 3); got {arr.shape}")
    return np.linalg.inv(arr)


def _load_warp_ctrl(raw: Any) -> Optional[np.ndarray]:
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size % 2 != 0:
            return None
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def resolve_registration_cache_path(path_candidate: Optional[os.PathLike[str] | str] = None) -> Path:
    """Resolve registration cache path honoring env overrides and repo defaults."""

    env_override = os.environ.get("TIVIT_REG_REFINED")
    if path_candidate:
        candidate = Path(path_candidate)
    elif env_override:
        candidate = Path(env_override)
    else:
        candidate = Path("reg_refined.json")
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _reconstruct_geometry_from_result(result: "RegistrationResult") -> Optional[Dict[str, Any]]:
    target_hw = tuple(result.target_hw)
    if len(target_hw) < 2:
        return None
    width = float(target_hw[1])
    edges = _canonical_white_edges(width)
    warped = _apply_warp_ctrl(edges, result.x_warp_ctrl)
    return _build_geometry_metadata_from_edges(warped, width, target_hw)


@dataclass
class RegistrationResult:
    homography: np.ndarray
    source_hw: Tuple[int, int]
    target_hw: Tuple[int, int]
    err_before: float
    err_after: float
    err_white_edges: float
    err_black_gaps: float
    frames: int
    status: str
    baseline_slope: float
    baseline_intercept: float
    keyboard_height: float
    timestamp: float
    x_warp_ctrl: Optional[np.ndarray] = None
    grid: Optional[torch.Tensor] = None
    geometry_meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> Dict[str, Any]:
        warp_payload: Optional[List[List[float]]]
        if self.x_warp_ctrl is None:
            warp_payload = None
        else:
            warp_payload = [
                [float(row[0]), float(row[1])] for row in np.asarray(self.x_warp_ctrl, dtype=np.float32)
            ]
        return {
            "homography": [float(x) for x in self.homography.reshape(-1)],
            "source_hw": list(self.source_hw),
            "target_hw": list(self.target_hw),
            "err_before": float(self.err_before),
            "err_after": float(self.err_after),
            "err_white_edges": float(self.err_white_edges),
            "err_black_gaps": float(self.err_black_gaps),
            "frames": int(self.frames),
            "status": str(self.status),
            "baseline_slope": float(self.baseline_slope),
            "baseline_intercept": float(self.baseline_intercept),
            "keyboard_height": float(self.keyboard_height),
            "timestamp": float(self.timestamp),
            "x_warp_ctrl": warp_payload,
            "geometry_meta": self.geometry_meta,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "RegistrationResult":
        homography = np.asarray(payload.get("homography", []), dtype=np.float32).reshape(3, 3)
        source_hw = tuple(int(v) for v in payload.get("source_hw", (0, 0)))  # type: ignore[arg-type]
        target_hw = tuple(int(v) for v in payload.get("target_hw", (0, 0)))  # type: ignore[arg-type]
        return cls(
            homography=homography,
            source_hw=source_hw,  # type: ignore[arg-type]
            target_hw=target_hw,  # type: ignore[arg-type]
            err_before=float(payload.get("err_before", 0.0)),
            err_after=float(payload.get("err_after", 0.0)),
            err_white_edges=float(payload.get("err_white_edges", float(payload.get("err_after", 0.0)))),
            err_black_gaps=float(payload.get("err_black_gaps", 0.0)),
            frames=int(payload.get("frames", 0)),
            status=str(payload.get("status", "unknown")),
            baseline_slope=float(payload.get("baseline_slope", 0.0)),
            baseline_intercept=float(payload.get("baseline_intercept", 0.0)),
            keyboard_height=float(payload.get("keyboard_height", 0.0)),
            timestamp=float(payload.get("timestamp", time.time())),
            x_warp_ctrl=_load_warp_ctrl(payload.get("x_warp_ctrl")),
            grid=None,
            geometry_meta=payload.get("geometry_meta") if isinstance(payload.get("geometry_meta"), dict) else None,
        )


@contextmanager
def _file_lock(path: Path):
    lock_handle = None
    try:
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = open(lock_path, "w")
        try:
            import fcntl  # type: ignore

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except Exception:  # pragma: no cover - Windows fallback
            pass
        yield
    finally:
        if lock_handle:
            try:
                import fcntl  # type: ignore

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            lock_handle.close()


class RegistrationRefiner:
    """Refines per-video keyboard registration using homography estimation."""

    def __init__(
        self,
        canonical_hw: Sequence[int],
        *,
        cache_path: Optional[Path] = None,
        sample_frames: int = 96,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if len(canonical_hw) < 2:
            raise ValueError("canonical_hw must provide (H, W)")
        self.canonical_hw: Tuple[int, int] = (int(canonical_hw[0]), int(canonical_hw[1]))
        self.sample_frames = int(max(sample_frames, 8))
        cache_candidate = Path(cache_path) if cache_path is not None else Path("reg_refined.json")
        if not cache_candidate.is_absolute():
            cache_candidate = PROJECT_ROOT / cache_candidate
        self.cache_path = cache_candidate
        self.logger = logger or LOGGER
        self._cache: Dict[str, RegistrationResult] = {}
        self._log_canonical_hw_once()
        self._load_cache()
        self._log_cache_summary_once()

    # ------------------------------------------------------------------ I/O --

    def _log_canonical_hw_once(self) -> None:
        h, w = self.canonical_hw
        canon: Tuple[int, int] = (int(h), int(w))
        if canon in _CANONICAL_HW_LOGGED:
            return
        _CANONICAL_HW_LOGGED.add(canon)
        self.logger.info("reg_refined: canonical_hw=%s cache=%s", canon, self.cache_path)

    def _log_cache_summary_once(self) -> None:
        key = (self.canonical_hw, self.cache_path)
        if key in _CACHE_SUMMARY_LOGGED:
            return
        _CACHE_SUMMARY_LOGGED.add(key)
        self.log_cache_summary(max_keys=5)

    def _load_cache(self) -> None:
        path = self.cache_path
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            self.logger.warning("reg_refined: unable to read cache %s (%s)", path, exc)
            return

        if not isinstance(data, dict):
            return
        stored = 0
        skipped_target_hw = 0
        skipped_invalid = 0
        metadata_updated = False
        for key, payload in data.items():
            if not isinstance(payload, dict):
                skipped_invalid += 1
                continue
            try:
                result = RegistrationResult.from_json(payload)
            except Exception as exc:
                skipped_invalid += 1
                self.logger.debug("reg_refined: failed to parse cache entry key=%s (%s)", key, exc)
                continue
            if tuple(result.target_hw) != self.canonical_hw:
                skipped_target_hw += 1
                self.logger.debug(
                    "reg_refined: skip cache entry key=%s target_hw=%s canonical_hw=%s",
                    key,
                    tuple(result.target_hw),
                    self.canonical_hw,
                )
                continue
            if result.geometry_meta is None or "tile_bounds_px" not in result.geometry_meta:
                rebuilt = _reconstruct_geometry_from_result(result)
                if rebuilt is not None:
                    result.geometry_meta = rebuilt
                    metadata_updated = True
            self._cache[str(key)] = result
            stored += 1
        self.logger.info(
            "reg_refined: loaded %d cache entries from %s (skipped_target_hw=%d skipped_invalid=%d)",
            stored,
            path,
            skipped_target_hw,
            skipped_invalid,
        )
        if metadata_updated:
            try:
                self._persist_cache()
            except Exception:
                self.logger.debug("reg_refined: failed to persist geometry metadata update")

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def export_cache_payload(self) -> Dict[str, Dict[str, Any]]:
        return {key: result.to_json() for key, result in self._cache.items()}

    def get_cache_entry_payload(self, video_id: str) -> Optional[Dict[str, Any]]:
        canon = canonical_video_id(video_id)
        entry = self._cache.get(canon)
        if entry is None:
            return None
        return entry.to_json()

    def get_geometry_metadata(self, video_id: str) -> Optional[Dict[str, Any]]:
        canon = canonical_video_id(video_id)
        entry = self._cache.get(canon)
        if entry is None:
            return None
        needs_rebuild = entry.geometry_meta is None or "tile_bounds_px" not in entry.geometry_meta
        if needs_rebuild:
            rebuilt = _reconstruct_geometry_from_result(entry)
            if rebuilt is None:
                return None
            entry.geometry_meta = rebuilt
            try:
                self._persist_cache()
            except Exception:
                pass
        return json.loads(json.dumps(entry.geometry_meta))

    def export_geometry_cache(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for key, entry in self._cache.items():
            if entry.geometry_meta:
                result[key] = json.loads(json.dumps(entry.geometry_meta))
        return result

    def peek_cache_keys(self, max_keys: int = 5) -> List[str]:
        max_keys = max(int(max_keys), 0)
        if max_keys == 0:
            return []
        return sorted(self._cache.keys())[:max_keys]

    def _persist_cache(self) -> None:
        path = self.cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: result.to_json() for key, result in self._cache.items()}
        tmp_path = path.with_suffix(".tmp")
        with _file_lock(path):
            try:
                with tmp_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                os.replace(tmp_path, path)
            except Exception as exc:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                self.logger.warning("reg_refined: failed to persist cache %s (%s)", path, exc)

    def log_cache_summary(
        self,
        *,
        video_id: Optional[str] = None,
        max_keys: int = 15,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Emit a human-readable snapshot of the current cache contents."""

        max_keys = max(int(max_keys), 0)
        target_logger = logger or self.logger
        keys = sorted(self._cache.keys())
        target_logger.info(
            "reg_refined: cache_path=%s entries=%d canonical_hw=%s",
            self.cache_path,
            len(keys),
            self.canonical_hw,
        )
        if keys and max_keys:
            target_logger.info("reg_refined: cache_keys_preview=%s", keys[:max_keys])
        elif not keys:
            target_logger.info("reg_refined: cache empty")
        if not video_id:
            return
        canon = canonical_video_id(video_id)
        entry = self._cache.get(canon)
        if entry is None:
            target_logger.info("reg_refined: cache_entry[%s]=missing", canon)
            return
        target_logger.info(
            "reg_refined: cache_entry[%s] status=%s target_hw=%s source_hw=%s keyboard_height=%.1f err_after=%.2f frames=%d",
            canon,
            entry.status,
            entry.target_hw,
            entry.source_hw,
            entry.keyboard_height,
            entry.err_after,
            entry.frames,
        )

    # --------------------------------------------------------------- Sampling --

    def _sample_video_frames(
        self,
        video_path: Path,
        crop_meta: Optional[Sequence[float] | Dict[str, Any]],
        *,
        max_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        target_frames = int(max_frames or self.sample_frames)
        target_frames = int(np.clip(target_frames, 60, 100))
        if target_frames <= 0:
            return []

        frames: List[np.ndarray] = []

        if _HAVE_DECORD:
            assert decord is not None
            try:
                vr = decord.VideoReader(str(video_path))
                total = len(vr)
                if total <= 0:
                    return []
                step = max(total // target_frames, 1)
                indices = list(range(0, total, step))
                if len(indices) > target_frames:
                    indices = indices[:target_frames]
                for idx in indices:
                    img = vr[idx].asnumpy()  # type: ignore[attr-defined]
                    if img is None:
                        continue
                    frame = _apply_crop_np(img, crop_meta)
                    frames.append(frame)
                if frames:
                    return frames
            except Exception as exc:  # pragma: no cover - fallback to cv2
                self.logger.debug("reg_refined: decord failed for %s (%s)", video_path, exc)

        if cv2 is None:
            return frames

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return frames
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0 or not math.isfinite(total):
                total = target_frames * 4
            indices = np.linspace(0, max(total - 1, 0), num=target_frames, dtype=int)
            seen = set()
            for idx in indices:
                idx = int(idx)
                if idx in seen:
                    continue
                seen.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(_apply_crop_np(frame_rgb, crop_meta))
                if len(frames) >= target_frames:
                    break
        finally:
            cap.release()
        return frames

    # ------------------------------------------------------------ Computation --

    def _compute_refinement(
        self,
        video_id: str,
        video_path: Path,
        crop_meta: Optional[Sequence[float] | Dict[str, Any]],
    ) -> RegistrationResult:
        frames = self._sample_video_frames(video_path, crop_meta)
        canonical = self.canonical_hw
        if not frames:
            return RegistrationResult(
                homography=np.eye(3, dtype=np.float32),
                source_hw=canonical,
                target_hw=canonical,
                err_before=0.0,
                err_after=0.0,
                err_white_edges=0.0,
                err_black_gaps=0.0,
                frames=0,
                status="fallback_no_frames",
                baseline_slope=0.0,
                baseline_intercept=0.0,
                keyboard_height=float(canonical[0]),
                timestamp=time.time(),
                x_warp_ctrl=None,
            )

        heights = [frame.shape[0] for frame in frames]
        widths = [frame.shape[1] for frame in frames]
        height = int(np.median(heights))
        width = int(np.median(widths))
        if height <= 0 or width <= 0:
            return RegistrationResult(
                homography=np.eye(3, dtype=np.float32),
                source_hw=canonical,
                target_hw=canonical,
                err_before=0.0,
                err_after=0.0,
                err_white_edges=0.0,
                err_black_gaps=0.0,
                frames=len(frames),
                status="fallback_bad_dims",
                baseline_slope=0.0,
                baseline_intercept=0.0,
                keyboard_height=float(canonical[0]),
                timestamp=time.time(),
                x_warp_ctrl=None,
            )

        geometry_meta: Optional[Dict[str, Any]] = None
        grayscale_frames: List[np.ndarray] = []
        slopes: List[float] = []
        intercepts: List[float] = []

        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if cv2 is not None else frame
            if frame.ndim == 3:
                if cv2 is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = np.mean(np.asarray(frame, dtype=np.float32), axis=-1)
            else:
                gray = frame
            gray = gray.astype(np.uint8)
            if cv2 is not None:
                gray = cv2.medianBlur(gray, 5)
            baseline = _detect_baseline(gray)
            if baseline:
                slopes.append(baseline[0])
                intercepts.append(baseline[1])
            grayscale_frames.append(gray)

        slope = _median_or_none(slopes)
        intercept = _median_or_none(intercepts)
        if slope is None or intercept is None:
            return RegistrationResult(
                homography=np.eye(3, dtype=np.float32),
                source_hw=(height, width),
                target_hw=canonical,
                err_before=0.0,
                err_after=0.0,
                err_white_edges=0.0,
                err_black_gaps=0.0,
                frames=len(frames),
                status="fallback_no_baseline",
                baseline_slope=0.0,
                baseline_intercept=0.0,
                keyboard_height=float(canonical[0]),
                timestamp=time.time(),
                x_warp_ctrl=None,
            )

        if grayscale_frames:
            stack = np.stack(grayscale_frames, axis=0)
            avg_frame = np.median(stack, axis=0).astype(np.float32)
            del stack
        else:
            avg_frame = np.zeros((height, width), dtype=np.float32)
        grad_x_profile, grad_y_profile = _aggregate_gradients(grayscale_frames)

        edges, edge_strengths, x_left, x_right = _compute_keyboard_edges(grad_x_profile, width)
        canon_edges = np.linspace(0.0, float(canonical[1] - 1), edges.shape[0], dtype=np.float32)

        scale_x = (canonical[1] - 1) / max(width - 1, 1)
        scale_y = (canonical[0] - 1) / max(height - 1, 1)
        x_before = edges * scale_x
        err_before = float(np.mean(np.abs(x_before - canon_edges)))

        xs = edges
        ys = np.clip(slope * xs + intercept, 0.0, height - 1.0)
        baseline_row = int(np.clip(np.median(ys), 0.0, height - 1.0))
        keyboard_height = _estimate_keyboard_height(grad_y_profile, baseline_row, height)
        keyboard_height = max(float(keyboard_height), 1.0)
        target_baseline_y = baseline_row * scale_y
        canonical_keyboard_height = keyboard_height * scale_y
        target_top_y = max(0.0, target_baseline_y - canonical_keyboard_height)
        source_top = np.clip(ys - keyboard_height, 0.0, height - 1.0)

        edge_strengths = np.abs(edge_strengths.astype(np.float32)) + 1e-6
        edge_weight_scale = float(np.max(edge_strengths))
        edge_weights = edge_strengths / max(edge_weight_scale, 1e-6)
        edge_weights = np.clip(edge_weights, 0.05, 1.0) * WHITE_ANCHOR_PRIORITY

        black_positions, black_strengths, black_boundaries = _compute_black_key_gaps(
            avg_frame,
            baseline_row,
            keyboard_height,
            xs,
        )
        canon_black_positions = np.zeros((0,), dtype=np.float32)
        black_weights = np.zeros((0,), dtype=np.float32)
        if black_positions.size > 0 and black_boundaries.size == black_positions.size:
            canon_pos, valid_mask = _canonical_black_key_positions(canon_edges, black_boundaries)
            if valid_mask.size == black_positions.size and np.any(valid_mask):
                black_positions = black_positions[valid_mask]
                black_strengths = black_strengths[valid_mask]
                black_boundaries = black_boundaries[valid_mask]
                canon_black_positions = canon_pos
            else:
                black_positions = np.zeros((0,), dtype=np.float32)
                black_strengths = np.zeros((0,), dtype=np.float32)
                black_boundaries = np.zeros((0,), dtype=np.int32)
        else:
            black_positions = np.zeros((0,), dtype=np.float32)
            black_strengths = np.zeros((0,), dtype=np.float32)
            black_boundaries = np.zeros((0,), dtype=np.int32)
        if black_positions.size > 0 and canon_black_positions.size == black_positions.size:
            black_strengths = np.abs(black_strengths.astype(np.float32)) + 1e-6
            black_weight_scale = float(np.max(black_strengths))
            black_weights = np.clip(black_strengths / max(black_weight_scale, 1e-6), 0.05, 1.0) * BLACK_ANCHOR_PRIORITY
        else:
            canon_black_positions = np.zeros((0,), dtype=np.float32)
            black_weights = np.zeros((0,), dtype=np.float32)
            black_positions = np.zeros((0,), dtype=np.float32)
            black_boundaries = np.zeros((0,), dtype=np.int32)

        src_pts: List[List[float]] = []
        dst_pts: List[List[float]] = []
        point_weights: List[float] = []

        for x_val, y_base, y_top, canon_x, weight in zip(xs, ys, source_top, canon_edges, edge_weights):
            src_pts.append([float(x_val), float(y_top)])
            dst_pts.append([float(canon_x), float(target_top_y)])
            point_weights.append(float(weight))
            src_pts.append([float(x_val), float(y_base)])
            dst_pts.append([float(canon_x), float(target_baseline_y)])
            point_weights.append(float(weight))

        if black_positions.size > 0 and canon_black_positions.size == black_positions.size:
            black_y_base = np.clip(slope * black_positions + intercept, 0.0, height - 1.0)
            black_y_top = np.clip(black_y_base - keyboard_height, 0.0, height - 1.0)
            for x_gap, y_base, y_top, canon_x, weight in zip(
                black_positions, black_y_base, black_y_top, canon_black_positions, black_weights
            ):
                src_pts.append([float(x_gap), float(y_top)])
                dst_pts.append([float(canon_x), float(target_top_y)])
                point_weights.append(float(weight))
                src_pts.append([float(x_gap), float(y_base)])
                dst_pts.append([float(canon_x), float(target_baseline_y)])
                point_weights.append(float(weight))

        corner_weight = 0.15
        corners_src = [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [0.0, float(height - 1)],
            [float(width - 1), float(height - 1)],
        ]
        corners_dst = [
            [0.0, 0.0],
            [float(canonical[1] - 1), 0.0],
            [0.0, float(canonical[0] - 1)],
            [float(canonical[1] - 1), float(canonical[0] - 1)],
        ]
        src_pts.extend(corners_src)
        dst_pts.extend(corners_dst)
        point_weights.extend([corner_weight] * len(corners_src))

        src_arr = np.asarray(src_pts, dtype=np.float32)
        dst_arr = np.asarray(dst_pts, dtype=np.float32)
        weights_arr = np.asarray(point_weights, dtype=np.float32)
        valid_mask = weights_arr > 1e-3
        if not np.all(valid_mask):
            src_arr = src_arr[valid_mask]
            dst_arr = dst_arr[valid_mask]
            weights_arr = weights_arr[valid_mask]
        base_h = np.array(
            [
                [scale_x, 0.0, 0.0],
                [0.0, scale_y, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        status = "ok"
        err_after = err_before
        err_white_edges = err_before
        err_black_gaps = err_before if black_positions.size > 0 else 0.0
        warp_ctrl = np.zeros((0, 2), dtype=np.float32)
        H: Optional[np.ndarray]

        if src_arr.shape[0] < 4 or np.count_nonzero(weights_arr > 0.05) < 4:
            H = base_h
            status = "fallback_points"
            err_black_gaps = err_before if black_positions.size > 0 else 0.0
        else:
            weights_norm = weights_arr / max(float(np.max(weights_arr)), 1e-6)
            weights_norm = np.clip(weights_norm, 0.05, 1.0)
            inlier_mask: Optional[np.ndarray] = None
            if cv2 is not None:
                _, mask = cv2.findHomography(
                    src_arr,
                    dst_arr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=2500,
                    confidence=0.997,
                )
                if mask is not None and mask.size == src_arr.shape[0]:
                    mask = mask.ravel().astype(bool)
                    if int(mask.sum()) >= 4:
                        inlier_mask = mask

            src_fit = src_arr[inlier_mask] if inlier_mask is not None else src_arr
            dst_fit = dst_arr[inlier_mask] if inlier_mask is not None else dst_arr
            weights_fit = weights_norm[inlier_mask] if inlier_mask is not None else weights_norm

            if src_fit.shape[0] < 4 or np.count_nonzero(weights_fit > 0.05) < 4:
                H = base_h
                status = "fallback_points"
                err_black_gaps = err_before if black_positions.size > 0 else 0.0
            else:
                weights_iter = weights_fit.astype(np.float32, copy=True)
                H_candidate: Optional[np.ndarray] = None
                for _ in range(6):
                    H_try = _solve_regularized_homography(
                        src_fit,
                        dst_fit,
                        base_h,
                        reg_lambda=0.05,
                        weights=weights_iter.tolist(),
                    )
                    if H_try is None or not np.all(np.isfinite(H_try)):
                        H_candidate = None
                        break
                    proj_pts = _apply_homography_to_points(H_try, src_fit)
                    residuals = np.linalg.norm(proj_pts - dst_fit, axis=1)
                    if residuals.size == 0:
                        H_candidate = H_try
                        break
                    median_resid = float(np.median(residuals))
                    if not math.isfinite(median_resid) or median_resid <= 1e-6:
                        median_resid = float(np.mean(residuals)) if residuals.size > 0 else 1.0
                    huber_delta = max(2.0, median_resid * 2.5)
                    huber_weights = np.where(residuals <= huber_delta, 1.0, huber_delta / (residuals + 1e-6))
                    weights_new = weights_fit * huber_weights
                    high_resid = residuals > (huber_delta * 2.5)
                    if np.any(high_resid):
                        weights_new[high_resid] = 0.0
                    weights_new = np.clip(weights_new, 0.0, 1.0).astype(np.float32, copy=False)
                    if np.linalg.norm(weights_new - weights_iter) <= 1e-3:
                        H_candidate = H_try
                        weights_iter = weights_new
                        break
                    weights_iter = weights_new
                    H_candidate = H_try
                    if np.count_nonzero(weights_iter > 0.05) < 4:
                        break

                if H_candidate is None:
                    H = base_h
                    status = "fallback_homography"
                    err_black_gaps = err_before if black_positions.size > 0 else 0.0
                else:
                    base_norm = float(np.linalg.norm(base_h))
                    delta_norm = float(np.linalg.norm(H_candidate - base_h))
                    if base_norm > 0.0 and delta_norm > 0.75 * base_norm:
                        blend = min(1.0, (0.75 * base_norm) / (delta_norm + 1e-6))
                        H_candidate = base_h + blend * (H_candidate - base_h)

                    white_points = np.stack([xs, ys], axis=1)
                    white_proj = _apply_homography_to_points(H_candidate, white_points)
                    white_proj_x = white_proj[:, 0]
                    white_err = np.abs(white_proj_x - canon_edges)
                    err_white_edges = float(np.mean(white_err))
                    geometry_meta = _build_geometry_metadata_from_edges(
                        white_proj_x,
                        float(canonical[1]),
                        canonical_hw=self.canonical_hw,
                    )

                    black_proj_x = np.zeros((0,), dtype=np.float32)
                    err_black_gaps = 0.0
                    if black_positions.size > 0 and canon_black_positions.size == black_positions.size:
                        black_points = np.stack(
                            [
                                black_positions,
                                np.clip(slope * black_positions + intercept, 0.0, height - 1.0),
                            ],
                            axis=1,
                        )
                        black_proj = _apply_homography_to_points(H_candidate, black_points)
                        black_proj_x = black_proj[:, 0]
                        err_black_gaps = float(np.mean(np.abs(black_proj_x - canon_black_positions)))

                    if (
                        black_positions.size > 0
                        and canon_black_positions.size == black_positions.size
                        and math.isfinite(err_black_gaps)
                    ):
                        weight_total = WHITE_ANCHOR_PRIORITY + BLACK_ANCHOR_PRIORITY
                        err_after = float(
                            (
                                err_white_edges * WHITE_ANCHOR_PRIORITY
                                + err_black_gaps * BLACK_ANCHOR_PRIORITY
                            )
                            / max(weight_total, 1e-6)
                        )
                    else:
                        err_after = err_white_edges

                    warp_ctrl = np.zeros((0, 2), dtype=np.float32)
                    apply_x_warp = err_after > X_WARP_ERR_THRESHOLD
                    if apply_x_warp:
                        anchor_src: List[np.ndarray] = [white_proj_x.astype(np.float32)]
                        anchor_dst: List[np.ndarray] = [canon_edges.astype(np.float32)]
                        if black_positions.size > 0 and black_proj_x.size == canon_black_positions.size:
                            anchor_src.append(black_proj_x.astype(np.float32))
                            anchor_dst.append(canon_black_positions.astype(np.float32))
                        warp_candidate = _build_x_warp_controls(
                            np.concatenate(anchor_src) if anchor_src else np.zeros((0,), dtype=np.float32),
                            np.concatenate(anchor_dst) if anchor_dst else np.zeros((0,), dtype=np.float32),
                            canonical[1],
                        )
                        if warp_candidate.size >= 4:
                            warp_ctrl = warp_candidate
                            ctrl_order = np.argsort(warp_ctrl[:, 0])
                            pre = warp_ctrl[ctrl_order, 0]
                            post = warp_ctrl[ctrl_order, 1]
                            white_after = np.interp(white_proj_x, pre, post, left=post[0], right=post[-1])
                            white_after = np.clip(white_after, 0.0, float(canonical[1] - 1))
                            err_white_edges = float(np.mean(np.abs(white_after - canon_edges)))
                            if black_positions.size > 0 and black_proj_x.size == canon_black_positions.size:
                                black_after = np.interp(black_proj_x, pre, post, left=post[0], right=post[-1])
                                black_after = np.clip(black_after, 0.0, float(canonical[1] - 1))
                                err_black_gaps = float(np.mean(np.abs(black_after - canon_black_positions)))
                            if (
                                black_positions.size > 0
                                and canon_black_positions.size == black_positions.size
                                and math.isfinite(err_black_gaps)
                            ):
                                weight_total = WHITE_ANCHOR_PRIORITY + BLACK_ANCHOR_PRIORITY
                                err_after = float(
                                    (
                                        err_white_edges * WHITE_ANCHOR_PRIORITY
                                        + err_black_gaps * BLACK_ANCHOR_PRIORITY
                                    )
                                    / max(weight_total, 1e-6)
                                )
                            else:
                                err_after = err_white_edges

                    if not math.isfinite(err_after):
                        H = base_h
                        status = "fallback_invalid"
                        err_after = err_before
                        err_white_edges = err_before
                        err_black_gaps = err_before if black_positions.size > 0 else 0.0
                        warp_ctrl = np.zeros((0, 2), dtype=np.float32)
                    else:
                        H = H_candidate

        if status != "ok":
            err_after = err_before
            err_white_edges = err_before
            err_black_gaps = err_before if black_positions.size > 0 else 0.0
            warp_ctrl = np.zeros((0, 2), dtype=np.float32)
            H = base_h if H is None else H
        else:
            err_after = float(err_after)

        H_arr = np.asarray(H, dtype=np.float32)
        if self.logger.isEnabledFor(logging.DEBUG):
            delta_norm = float(np.linalg.norm(H_arr - base_h))
            err_delta = float(err_before - err_after)
            self.logger.debug(
                (
                    "reg_refined.debug: %s status=%s err_before=%.2fpx err_after=%.2fpx "
                    "Δerr=%.2fpx err_white=%.2fpx err_black=%.2fpx ||ΔH||=%.3f frames=%d"
                ),
                video_id,
                status,
                err_before,
                err_after,
                err_delta,
                err_white_edges,
                err_black_gaps,
                delta_norm,
                len(frames),
            )
        H_inv = _invert_homography(H_arr)
        warp_for_grid: Optional[np.ndarray] = warp_ctrl if warp_ctrl.size >= 4 else None
        grid = _homography_to_grid(H_inv, (height, width), canonical, warp_ctrl=warp_for_grid)
        return RegistrationResult(
            homography=H_arr,
            source_hw=(height, width),
            target_hw=canonical,
            err_before=err_before,
            err_after=err_after,
            err_white_edges=err_white_edges,
            err_black_gaps=err_black_gaps,
            frames=len(frames),
            status=status,
            baseline_slope=float(slope),
            baseline_intercept=float(intercept),
            keyboard_height=float(keyboard_height),
            timestamp=time.time(),
            x_warp_ctrl=warp_for_grid.copy() if warp_for_grid is not None else None,
            grid=grid,
            geometry_meta=geometry_meta,
        )

    # --------------------------------------------------------------- Public API --

    def refine(
        self,
        *,
        video_id: str,
        video_path: Path,
        crop_meta: Optional[Sequence[float] | Dict[str, Any]],
    ) -> RegistrationResult:
        canon_id = canonical_video_id(video_id)
        cached = self._cache.get(canon_id)
        if cached is not None:
            if cached.grid is None:
                try:
                    H_inv = _invert_homography(cached.homography)
                    cached.grid = _homography_to_grid(
                        H_inv,
                        cached.source_hw,
                        cached.target_hw,
                        warp_ctrl=cached.x_warp_ctrl,
                    )
                except Exception:
                    cached.grid = None
            return cached

        result = self._compute_refinement(canon_id, video_path, crop_meta)
        if result.grid is None:
            try:
                H_inv = _invert_homography(result.homography)
                result.grid = _homography_to_grid(
                    H_inv,
                    result.source_hw,
                    result.target_hw,
                    warp_ctrl=result.x_warp_ctrl,
                )
            except Exception:
                result.grid = None
        self._cache[canon_id] = result
        if result.status.startswith("fallback"):
            self.logger.warning(
                (
                    "reg_refined: %s status=%s err_before=%.2fpx err_after=%.2fpx "
                    "err_white=%.2fpx err_black=%.2fpx frames=%d"
                ),
                canon_id,
                result.status,
                result.err_before,
                result.err_after,
                result.err_white_edges,
                result.err_black_gaps,
                result.frames,
            )
        else:
            self.logger.info(
                "reg_refined: %s err_before=%.2fpx err_after=%.2fpx err_white=%.2fpx err_black=%.2fpx frames=%d",
                canon_id,
                result.err_before,
                result.err_after,
                result.err_white_edges,
                result.err_black_gaps,
                result.frames,
            )
        self._persist_cache()
        return result

    def transform_clip(
        self,
        clip: torch.Tensor,
        *,
        video_id: str,
        video_path: Path,
        crop_meta: Optional[Sequence[float] | Dict[str, Any]],
        interp: str = "bilinear",
    ) -> torch.Tensor:
        result = self.refine(video_id=video_id, video_path=video_path, crop_meta=crop_meta)
        target_h, target_w = self.canonical_hw
        alignable = {"linear", "bilinear", "bicubic", "trilinear"}
        if result.grid is None or clip.ndim != 4:
            if interp in alignable:
                return F.interpolate(clip, size=self.canonical_hw, mode=interp, align_corners=False)
            return F.interpolate(clip, size=self.canonical_hw, mode=interp)

        grid = result.grid.to(clip.device, clip.dtype)
        clip_in = clip.contiguous()
        T = clip.shape[0]
        grid_batched = grid.unsqueeze(0).expand(T, -1, -1, -1).contiguous()
        mode = interp if interp in {"bilinear", "nearest"} else "bilinear"
        warped = F.grid_sample(
            clip_in,
            grid_batched,
            mode=mode,
            align_corners=False,
            padding_mode="border",
        )
        if warped.shape[-2:] != (target_h, target_w):
            if interp in alignable:
                warped = F.interpolate(warped, size=self.canonical_hw, mode=interp, align_corners=False)
            else:
                warped = F.interpolate(warped, size=self.canonical_hw, mode=interp)
        return warped


__all__ = ["RegistrationRefiner", "RegistrationResult", "resolve_registration_cache_path"]
