"""Utilities for mapping rectified tile spans to MIDI key coverage."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _uniform_bounds(num_tiles: int) -> List[Tuple[float, float]]:
    edges = np.linspace(0.0, 1.0, num_tiles + 1, dtype=np.float32)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(num_tiles)]


def _coerce_pair(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, Mapping):
        for left_key in ("left", "x0", "start", "min_x", "min", "lo", "low"):
            if left_key in value:
                left = value[left_key]
                break
        else:
            left = None
        for right_key in ("right", "x1", "end", "max_x", "max", "hi", "high"):
            if right_key in value:
                right = value[right_key]
                break
        else:
            right = None
        if left is None or right is None:
            return None
        try:
            return float(left), float(right)
        except (TypeError, ValueError):
            return None

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _coerce_bounds(candidate: Any, num_tiles: int) -> Optional[List[Tuple[float, float]]]:
    if candidate is None:
        return None

    if isinstance(candidate, Mapping):
        # Sometimes stored as {"0": [..], "1": [..]}
        ordered = sorted(candidate.items(), key=lambda kv: kv[0])
        parsed = [_coerce_pair(val) for _, val in ordered]
        if all(pair is not None for pair in parsed) and len(parsed) == num_tiles:
            return [pair for pair in parsed if pair is not None]
        candidate = list(candidate.values())

    if isinstance(candidate, np.ndarray):
        candidate = candidate.tolist()

    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        return None

    values: List[Any] = list(candidate)
    if not values:
        return None

    # Case 1: list of explicit pairs
    parsed_pairs = [_coerce_pair(item) for item in values]
    if all(pair is not None for pair in parsed_pairs) and len(parsed_pairs) == num_tiles:
        return [pair for pair in parsed_pairs if pair is not None]

    # Case 2: boundary edges (len == num_tiles + 1)
    if len(values) == num_tiles + 1:
        try:
            edges = [float(val) for val in values]
        except (TypeError, ValueError):
            edges = []
        if len(edges) == num_tiles + 1:
            return list(zip(edges[:-1], edges[1:]))

    # Case 3: widths per tile (len == num_tiles)
    if len(values) == num_tiles:
        try:
            widths = [float(val) for val in values]
        except (TypeError, ValueError):
            widths = []
        if len(widths) == num_tiles and all(width >= 0.0 for width in widths):
            edges = [0.0]
            for width in widths:
                edges.append(edges[-1] + width)
            return list(zip(edges[:-1], edges[1:]))

    return None


def _resolve_width_hint(reg_meta: Any) -> Optional[float]:
    if not isinstance(reg_meta, Mapping):
        return None

    for key in ("frame_width", "width", "rectified_width"):
        val = reg_meta.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val)

    target_hw = reg_meta.get("target_hw") or reg_meta.get("canonical_hw")
    if isinstance(target_hw, Sequence) and len(target_hw) >= 2:
        try:
            width = float(target_hw[1])
            if width > 0:
                return width
        except (TypeError, ValueError):
            return None
    return None


def _normalize_bounds(bounds: Sequence[Tuple[float, float]], width_hint: Optional[float]) -> List[Tuple[float, float]]:
    if not bounds:
        return []
    starts = np.asarray([b[0] for b in bounds], dtype=np.float32)
    ends = np.asarray([b[1] for b in bounds], dtype=np.float32)
    mask = np.isfinite(starts) & np.isfinite(ends)
    if not np.all(mask):
        raise ValueError("tile bounds contain non-finite values")
    min_start = float(np.min(starts))
    max_end = float(np.max(ends))
    if width_hint is None or width_hint <= 0.0:
        width = max(max_end - min_start, 1e-6)
    else:
        width = float(width_hint)
    if not np.isfinite(width) or width <= 0.0:
        width = max(max_end - min_start, 1e-6)
    norm_bounds: List[Tuple[float, float]] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if end < start:
            start, end = end, start
        start_norm = (start - min_start) / width
        end_norm = (end - min_start) / width
        norm_bounds.append((float(start_norm), float(end_norm)))
    return norm_bounds


def _extract_tile_bounds(reg_meta: Any, num_tiles: int) -> List[Tuple[float, float]]:
    if isinstance(reg_meta, Mapping):
        candidate_keys = [
            "tile_bounds_norm",
            "tile_bounds_normalized",
            "tile_bounds",
            "tile_bounds_px",
            "tile_bounds_rect",
        ]
        for key in candidate_keys:
            if key in reg_meta:
                parsed = _coerce_bounds(reg_meta.get(key), num_tiles)
                if parsed is not None and len(parsed) == num_tiles:
                    return _normalize_bounds(parsed, _resolve_width_hint(reg_meta))
        tiles_meta = reg_meta.get("tiles")
        if isinstance(tiles_meta, Mapping):
            for key in ("bounds", "bounds_px", "bounds_norm"):
                parsed = _coerce_bounds(tiles_meta.get(key), num_tiles)
                if parsed is not None and len(parsed) == num_tiles:
                    return _normalize_bounds(parsed, _resolve_width_hint(reg_meta))
        if "tile_tokens" in reg_meta:
            parsed = _coerce_bounds(reg_meta["tile_tokens"], num_tiles)
            if parsed is not None and len(parsed) == num_tiles:
                return _normalize_bounds(parsed, _resolve_width_hint(reg_meta))
    return _uniform_bounds(num_tiles)


def build_tile_key_mask(
    reg_meta: Any,
    num_tiles: int,
    cushion_keys: int,
    n_keys: int = 88,
) -> np.ndarray:
    """
    Build a boolean mask of shape (tiles, n_keys) indicating which MIDI keys each
    tile is responsible for after rectification.

    Args:
        reg_meta: Registration metadata providing optional tile bounds (any shape).
        num_tiles: Number of tiles used by the encoder.
        cushion_keys: How many keys to expand on each side of a tile's span.
        n_keys: Total number of keys (default: 88 for A0–C8).

    Returns:
        np.ndarray: Boolean mask with True where ``tile`` covers ``key``.
    """

    if num_tiles <= 0:
        raise ValueError("num_tiles must be positive")
    if n_keys <= 0:
        raise ValueError("n_keys must be positive")

    bounds = _extract_tile_bounds(reg_meta, num_tiles)
    cushion = max(int(cushion_keys), 0)
    mask = np.zeros((num_tiles, n_keys), dtype=bool)

    key_min = 0
    key_max = n_keys - 1
    key_span = key_max - key_min + 1

    for idx, (start_norm, end_norm) in enumerate(bounds):
        start_norm = float(np.clip(min(start_norm, end_norm), 0.0, 1.0))
        end_norm = float(np.clip(max(start_norm, end_norm), 0.0, 1.0))
        start_key = key_min + start_norm * key_span
        end_key = key_min + end_norm * key_span
        start_idx = int(np.floor(start_key))
        end_idx = int(np.ceil(end_key) - 1)
        if end_idx < start_idx:
            end_idx = start_idx

        start_idx = max(key_min, start_idx - cushion)
        end_idx = min(key_max, end_idx + cushion)
        mask[idx, start_idx : end_idx + 1] = True

    return mask


__all__ = ["build_tile_key_mask"]
