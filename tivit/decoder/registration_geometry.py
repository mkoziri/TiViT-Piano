"""Registration metadata helpers for rectified keyboard geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_DEFAULT_WIDTH = 800.0
_MIDI_LOW = 21
_MIDI_HIGH = 108
_WHITE_PITCHES = {0, 2, 4, 5, 7, 9, 11}


def _as_sequence(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        if not value:
            return None
        ordered = sorted(value.items(), key=lambda kv: kv[0])
        return [item for _, item in ordered]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return None


def _coerce_float_sequence(value: Any, *, min_len: int = 0) -> Optional[List[float]]:
    seq = _as_sequence(value)
    if seq is None:
        return None
    floats: List[float] = []
    for entry in seq:
        try:
            floats.append(float(entry))
        except (TypeError, ValueError):
            return None
    if len(floats) < min_len:
        return None
    return floats


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
    seq = _as_sequence(value)
    if seq is None or len(seq) < 2:
        return None
    try:
        return float(seq[0]), float(seq[1])
    except (TypeError, ValueError):
        return None


def _coerce_bounds(candidate: Any, expected: int) -> Optional[List[Tuple[float, float]]]:
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        parsed = [_coerce_pair(val) for _, val in sorted(candidate.items(), key=lambda kv: kv[0])]
        parsed = [pair for pair in parsed if pair is not None]
        if len(parsed) == expected and all(pair is not None for pair in parsed):
            return parsed
        candidate = list(candidate.values())

    if isinstance(candidate, np.ndarray):
        candidate = candidate.tolist()

    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
        return None

    values = list(candidate)
    if not values:
        return None

    parsed_pairs = [_coerce_pair(item) for item in values]
    if all(pair is not None for pair in parsed_pairs) and len(parsed_pairs) == expected:
        return [pair for pair in parsed_pairs if pair is not None]

    if len(values) == expected + 1:
        try:
            edges = [float(val) for val in values]
        except (TypeError, ValueError):
            edges = []
        if len(edges) == expected + 1:
            return list(zip(edges[:-1], edges[1:]))

    if len(values) == expected:
        try:
            widths = [float(val) for val in values]
        except (TypeError, ValueError):
            widths = []
        if len(widths) == expected and all(width >= 0.0 for width in widths):
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
        if isinstance(val, (int, float)) and float(val) > 0:
            return float(val)
    target_hw = reg_meta.get("target_hw") or reg_meta.get("canonical_hw")
    seq = _as_sequence(target_hw)
    if seq and len(seq) >= 2:
        try:
            width = float(seq[1])
        except (TypeError, ValueError):
            return None
        if width > 0:
            return width
    return None


def _normalize_bounds(
    bounds: Sequence[Tuple[float, float]],
    width_hint: Optional[float],
) -> List[Tuple[float, float]]:
    if not bounds:
        return []
    starts = np.asarray([b[0] for b in bounds], dtype=np.float32)
    ends = np.asarray([b[1] for b in bounds], dtype=np.float32)
    mask = np.isfinite(starts) & np.isfinite(ends)
    if not np.all(mask):
        return []
    min_start = float(np.min(starts))
    max_end = float(np.max(ends))
    width = float(width_hint) if (width_hint is not None and width_hint > 0) else (max_end - min_start)
    width = float(width) if np.isfinite(width) and width > 1e-6 else (max_end - min_start)
    width = max(width, 1e-6)
    normalized: List[Tuple[float, float]] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        if end < start:
            start, end = end, start
        start_norm = (start - min_start) / width
        end_norm = (end - min_start) / width
        normalized.append((float(start_norm), float(end_norm)))
    return normalized


def _is_normalized(values: Iterable[float], *, tol: float = 1e-3) -> bool:
    vals = list(values)
    if not vals:
        return False
    return all(-tol <= v <= 1.0 + tol for v in vals)


def _midi_is_white(midi: int) -> bool:
    return (midi % 12) in _WHITE_PITCHES


def _canonical_white_edges(width: float) -> np.ndarray:
    span = max(float(width), 1.0)
    return np.linspace(0.0, span, num=52 + 1, dtype=np.float32)


def _canonical_key_geometry(width: float, n_keys: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = _canonical_white_edges(width)
    centers_white = 0.5 * (edges[:-1] + edges[1:])
    widths_white = edges[1:] - edges[:-1]
    centers = np.zeros(n_keys, dtype=np.float32)
    left = np.zeros(n_keys, dtype=np.float32)
    right = np.zeros(n_keys, dtype=np.float32)
    white_idx = 0
    for idx in range(n_keys):
        midi = _MIDI_LOW + idx
        if _midi_is_white(midi):
            white_idx = min(white_idx, centers_white.shape[0] - 1)
            left[idx] = edges[white_idx]
            right[idx] = edges[white_idx + 1]
            centers[idx] = centers_white[white_idx]
            white_idx += 1
        else:
            left_white = max(white_idx - 1, 0)
            right_white = min(white_idx, centers_white.shape[0] - 1)
            left_center = centers_white[left_white]
            right_center = centers_white[right_white]
            center = 0.5 * (left_center + right_center)
            width_left = widths_white[left_white]
            width_right = widths_white[right_white]
            key_width = 0.6 * min(width_left, width_right)
            left[idx] = max(center - key_width / 2.0, 0.0)
            right[idx] = min(center + key_width / 2.0, float(width))
            centers[idx] = center
    return centers, left, right


def _bounds_from_centers(centers: np.ndarray, width: float) -> Tuple[np.ndarray, np.ndarray]:
    if centers.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    centers_clipped = np.clip(centers.astype(np.float32), 0.0, float(width))
    diffs = np.diff(centers_clipped)
    if np.any(diffs < -1e-3):
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    edges = np.zeros(centers_clipped.shape[0] + 1, dtype=np.float32)
    edges[0] = 0.0
    edges[-1] = float(width)
    edges[1:-1] = 0.5 * (centers_clipped[:-1] + centers_clipped[1:])
    left = np.clip(edges[:-1], 0.0, float(width))
    right = np.clip(edges[1:], 0.0, float(width))
    return left, right


def _extract_key_bounds_px(
    reg_meta: Any,
    n_keys: int,
    width: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(reg_meta, Mapping):
        return None
    candidate_keys = [
        "key_bounds_px",
        "key_bounds_rect",
        "key_bounds",
        "key_edges",
    ]
    for key in candidate_keys:
        if key not in reg_meta:
            continue
        parsed = _coerce_bounds(reg_meta.get(key), n_keys)
        if not parsed:
            continue
        bounds = np.asarray(parsed, dtype=np.float32)
        if bounds.shape[0] != n_keys or bounds.shape[1] != 2:
            continue
        values = bounds.reshape(-1)
        normalized = _is_normalized(values)
        if normalized:
            bounds = bounds * float(width)
        left = np.clip(bounds[:, 0], 0.0, float(width))
        right = np.clip(bounds[:, 1], 0.0, float(width))
        swapped = right < left
        if np.any(swapped):
            tmp = left.copy()
            left = np.where(swapped, right, left)
            right = np.where(swapped, tmp, right)
        if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
            continue
        return left.astype(np.float32, copy=False), right.astype(np.float32, copy=False)
    return None


def _extract_key_centers_px(
    reg_meta: Any,
    n_keys: int,
    width: float,
) -> Optional[np.ndarray]:
    if not isinstance(reg_meta, Mapping):
        return None
    candidate_keys = [
        "key_centers_px",
        "key_centers_rect",
        "key_centers",
    ]
    for key in candidate_keys:
        if key not in reg_meta:
            continue
        seq = _coerce_float_sequence(reg_meta.get(key), min_len=n_keys)
        if not seq:
            continue
        arr = np.asarray(seq[:n_keys], dtype=np.float32)
        normalized = _is_normalized(arr)
        if normalized:
            arr = arr * float(width)
        if np.any(~np.isfinite(arr)):
            continue
        if np.any(np.diff(arr) < -1e-3):
            continue
        return np.clip(arr, 0.0, float(width))
    return None


@dataclass(frozen=True)
class RectifiedKeyboardGeometry:
    width: float
    key_centers: np.ndarray
    key_left: np.ndarray
    key_right: np.ndarray
    source: str

    @property
    def valid(self) -> bool:
        return bool(
            self.key_centers.size
            and self.key_centers.shape == self.key_left.shape == self.key_right.shape
        )


def resolve_rectified_keyboard_geometry(
    reg_meta: Any,
    *,
    n_keys: int = 88,
    default_width: float = _DEFAULT_WIDTH,
) -> RectifiedKeyboardGeometry:
    width = float(_resolve_width_hint(reg_meta) or default_width)
    if width <= 0:
        width = float(default_width)
    bounds = _extract_key_bounds_px(reg_meta, n_keys, width)
    if bounds is not None:
        left, right = bounds
        centers = 0.5 * (left + right)
        return RectifiedKeyboardGeometry(width, centers, left, right, source="metadata_bounds")
    centers = _extract_key_centers_px(reg_meta, n_keys, width)
    if centers is not None:
        left, right = _bounds_from_centers(centers, width)
        if left.size == centers.size:
            return RectifiedKeyboardGeometry(width, centers, left, right, source="metadata_centers")
    centers, left, right = _canonical_key_geometry(width, n_keys)
    return RectifiedKeyboardGeometry(width, centers, left, right, source="uniform")


@dataclass(frozen=True)
class TileBounds:
    normalized: np.ndarray
    pixels: np.ndarray
    source: str


def resolve_tile_bounds(
    reg_meta: Any,
    *,
    num_tiles: int,
    width: float,
) -> TileBounds:
    normalized = _extract_tile_bounds_normalized(reg_meta, num_tiles, width_hint=width)
    source = "metadata" if normalized is not None else "uniform"
    if normalized is None:
        normalized = _uniform_bounds(num_tiles)
    norm_arr = np.asarray(normalized, dtype=np.float32)
    px = np.clip(norm_arr * float(width), 0.0, float(width))
    return TileBounds(norm_arr, px, source)


def _extract_tile_bounds_normalized(
    reg_meta: Any,
    num_tiles: int,
    *,
    width_hint: Optional[float],
) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(reg_meta, Mapping):
        return None
    candidate_keys = [
        "tile_bounds_norm",
        "tile_bounds_normalized",
        "tile_bounds",
        "tile_bounds_px",
        "tile_bounds_rect",
    ]
    for key in candidate_keys:
        if key not in reg_meta:
            continue
        parsed = _coerce_bounds(reg_meta.get(key), num_tiles)
        if parsed is None:
            continue
        normalized = _normalize_bounds(parsed, width_hint)
        if normalized:
            return normalized
    tiles_meta = reg_meta.get("tiles")
    if isinstance(tiles_meta, Mapping):
        for key in ("bounds", "bounds_px", "bounds_norm"):
            parsed = _coerce_bounds(tiles_meta.get(key), num_tiles)
            if parsed is None:
                continue
            normalized = _normalize_bounds(parsed, width_hint)
            if normalized:
                return normalized
    if "tile_tokens" in reg_meta:
        parsed = _coerce_bounds(reg_meta.get("tile_tokens"), num_tiles)
        if parsed is not None:
            normalized = _normalize_bounds(parsed, width_hint)
            if normalized:
                return normalized
    return None


def _uniform_bounds(num_tiles: int) -> List[Tuple[float, float]]:
    if num_tiles <= 0:
        return []
    edges = np.linspace(0.0, 1.0, num_tiles + 1, dtype=np.float32)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(num_tiles)]


def build_canonical_registration_metadata(
    width: float,
    num_tiles: int,
    *,
    n_keys: int = 88,
) -> Dict[str, Any]:
    """Return a metadata-like payload using canonical key/tile geometry."""

    span = max(float(width), 1.0)
    _, left, right = _canonical_key_geometry(span, n_keys)
    key_bounds = [[float(l), float(r)] for l, r in zip(left.tolist(), right.tolist())]
    tile_bounds_px = [
        (float(lo) * span, float(hi) * span) for lo, hi in _uniform_bounds(num_tiles)
    ]
    return {
        "rectified_width": span,
        "key_bounds_px": key_bounds,
        "tile_bounds_px": tile_bounds_px,
    }


__all__ = [
    "RectifiedKeyboardGeometry",
    "TileBounds",
    "resolve_rectified_keyboard_geometry",
    "resolve_tile_bounds",
    "build_canonical_registration_metadata",
]
