"""Utilities for mapping rectified tile spans to MIDI key coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .registration_geometry import (
    RectifiedKeyboardGeometry,
    TileBounds,
    resolve_rectified_keyboard_geometry,
    resolve_tile_bounds,
)


@dataclass(frozen=True)
class TileMaskResult:
    mask: np.ndarray
    registration_based: bool
    fallback_reason: Optional[str]
    tile_bounds_px: np.ndarray
    rectified_width: float
    key_centers_px: np.ndarray
    boundary_keys: int
    tile_key_ranges: List[Tuple[int, int]]
    tile_key_counts: List[int]
    geometry_source: str
    tile_source: str


def _uniform_key_mask(num_tiles: int, n_keys: int, cushion: int) -> np.ndarray:
    mask = np.zeros((num_tiles, n_keys), dtype=bool)
    edges = np.linspace(0.0, n_keys, num_tiles + 1, dtype=np.float32)
    for idx in range(num_tiles):
        start = int(np.floor(edges[idx]))
        end = int(np.ceil(edges[idx + 1]) - 1)
        start = max(0, start - cushion)
        end = min(n_keys - 1, max(start, end + cushion))
        mask[idx, start : end + 1] = True
    return mask


def _apply_cushion(mask: np.ndarray, cushion: int) -> np.ndarray:
    if cushion <= 0:
        return mask
    expanded = mask.copy()
    n_keys = mask.shape[1]
    for idx in range(mask.shape[0]):
        covered = np.flatnonzero(mask[idx])
        if covered.size == 0:
            continue
        start = max(int(covered[0]) - cushion, 0)
        end = min(int(covered[-1]) + cushion, n_keys - 1)
        expanded[idx, start : end + 1] = True
    return expanded


def _mask_from_geometry(
    geometry: RectifiedKeyboardGeometry,
    tile_bounds: TileBounds,
    *,
    n_keys: int,
) -> np.ndarray:
    num_tiles = tile_bounds.pixels.shape[0]
    mask = np.zeros((num_tiles, n_keys), dtype=bool)
    key_centers = geometry.key_centers
    bounds = tile_bounds.pixels
    tile_centers = 0.5 * (bounds[:, 0] + bounds[:, 1])
    for idx in range(num_tiles):
        left = float(min(bounds[idx, 0], bounds[idx, 1]))
        right = float(max(bounds[idx, 0], bounds[idx, 1]))
        within = (key_centers >= left) & (key_centers <= right)
        if idx == 0:
            within |= key_centers < left
        if idx == num_tiles - 1:
            within |= key_centers > right
        mask[idx, within] = True
    missing = np.where(mask.sum(axis=0) == 0)[0]
    if missing.size > 0 and tile_centers.size > 0:
        kc = key_centers[missing].reshape(1, -1)
        tc = tile_centers.reshape(-1, 1)
        nearest = np.argmin(np.abs(tc - kc), axis=0)
        for key_idx, tile_idx in zip(missing.tolist(), nearest.tolist()):
            tile_idx = int(np.clip(tile_idx, 0, num_tiles - 1))
            mask[tile_idx, int(key_idx)] = True
    return mask


def _mask_valid(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or mask.size == 0:
        return False
    if not np.all(mask.sum(axis=1) > 0):
        return False
    if not np.all(mask.sum(axis=0) > 0):
        return False
    return True


def _count_boundary_keys(mask: np.ndarray) -> int:
    overlap = mask.sum(axis=0)
    return int(np.count_nonzero(overlap > 1))


def _compute_ranges(mask: np.ndarray) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for row in mask:
        covered = np.flatnonzero(row)
        if covered.size == 0:
            ranges.append((0, -1))
        else:
            ranges.append((int(covered[0]), int(covered[-1])))
    return ranges


def _finalize_result(
    mask: np.ndarray,
    *,
    registration_based: bool,
    reason: Optional[str],
    tile_bounds: TileBounds,
    geometry: RectifiedKeyboardGeometry,
    geometry_source: str,
    tile_source: str,
) -> TileMaskResult:
    counts = mask.sum(axis=1).astype(int).tolist()
    ranges = _compute_ranges(mask)
    boundary = _count_boundary_keys(mask)
    return TileMaskResult(
        mask=mask,
        registration_based=registration_based,
        fallback_reason=reason,
        tile_bounds_px=tile_bounds.pixels,
        rectified_width=float(geometry.width),
        key_centers_px=geometry.key_centers,
        boundary_keys=boundary,
        tile_key_ranges=ranges,
        tile_key_counts=counts,
        geometry_source=geometry_source,
        tile_source=tile_source,
    )


def build_tile_key_mask(
    reg_meta: Any,
    num_tiles: int,
    cushion_keys: int,
    n_keys: int = 88,
) -> TileMaskResult:
    """Compute tile-to-key coverage using registration metadata when available."""

    if num_tiles <= 0:
        raise ValueError("num_tiles must be positive")
    if n_keys <= 0:
        raise ValueError("n_keys must be positive")

    cushion = max(int(cushion_keys), 0)
    geometry = resolve_rectified_keyboard_geometry(reg_meta, n_keys=n_keys)
    tile_bounds = resolve_tile_bounds(reg_meta, num_tiles=num_tiles, width=geometry.width)
    geometry_ready = geometry.source != "uniform" and geometry.valid
    tile_ready = tile_bounds.pixels.size > 0

    if not geometry_ready or not tile_ready:
        reason = "missing_key_geometry" if not geometry_ready else "missing_tile_bounds"
        fallback_mask = _uniform_key_mask(num_tiles, n_keys, cushion)
        return _finalize_result(
            fallback_mask,
            registration_based=False,
            reason=reason,
            tile_bounds=tile_bounds,
            geometry=geometry,
            geometry_source=geometry.source,
            tile_source=tile_bounds.source,
        )

    mask = _mask_from_geometry(geometry, tile_bounds, n_keys=n_keys)
    mask = _apply_cushion(mask, cushion)
    if not _mask_valid(mask):
        fallback_mask = _uniform_key_mask(num_tiles, n_keys, cushion)
        return _finalize_result(
            fallback_mask,
            registration_based=False,
            reason="invalid_mask",
            tile_bounds=tile_bounds,
            geometry=geometry,
            geometry_source=geometry.source,
            tile_source=tile_bounds.source,
        )

    return _finalize_result(
        mask,
        registration_based=True,
        reason=None,
        tile_bounds=tile_bounds,
        geometry=geometry,
        geometry_source=geometry.source,
        tile_source=tile_bounds.source,
    )


__all__ = ["TileMaskResult", "build_tile_key_mask"]
