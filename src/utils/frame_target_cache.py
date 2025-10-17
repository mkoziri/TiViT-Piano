"""Utilities for persisting per-clip frame target tensors.

Purpose:
    Provide a lightweight disk cache keyed by dataset/video properties so the
    expensive per-frame target construction only runs when required.  The cache
    mirrors the behaviour of :class:`utils.av_sync.AVLagCache` but stores torch
    tensors representing pianoroll labels for a particular combination of
    split, video, temporal alignment, and frame-target configuration.

Usage:
    >>> cache = FrameTargetCache()
    >>> key_hash, meta = make_target_cache_key(...)
    >>> tensors, hit = cache.load(key_hash)
    >>> if tensors is None:
    ...     tensors = build_targets_somehow()
    ...     cache.save(key_hash, meta, tensors)

The helper intentionally keeps the interface tiny so dataset loaders can call
``load`` and ``save`` without worrying about the on-disk format.  All tensors
are stored in CPU memory for portability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, TypedDict

import torch

from .identifiers import canonical_video_id

LOGGER = logging.getLogger(__name__)


class FrameTargetMeta(TypedDict):
    """Metadata stored alongside cached frame targets."""

    split: str
    video_id: str
    lag_frames: int
    lag_ms: float
    fps: float
    frames: int
    tolerance: float
    dilation: int
    canonical_hw: Tuple[int, int]


def _round_lag_frames(lag_ms: float, fps: float) -> Tuple[int, float]:
    try:
        lag_val = float(lag_ms)
    except (TypeError, ValueError):
        lag_val = 0.0
    try:
        fps_val = float(fps)
    except (TypeError, ValueError):
        fps_val = 0.0

    if not math.isfinite(fps_val) or fps_val <= 0:
        return 0, 0.0

    lag_frames = int(round(lag_val * fps_val / 1000.0))
    lag_ms_frame = (lag_frames / fps_val) * 1000.0
    return lag_frames, lag_ms_frame


def _round_lag_ms_legacy(lag_ms: float, fps: float) -> Tuple[int, float]:
    """Replicate the historic ms-based rounding used for compatibility."""

    from .av_sync import round_lag_ms_for_cache  # local import to avoid cycle

    rounded_ms = float(round_lag_ms_for_cache(lag_ms, fps))
    try:
        fps_val = float(fps)
    except (TypeError, ValueError):
        fps_val = 0.0
    if not math.isfinite(fps_val) or fps_val <= 0:
        lag_frames = int(round(rounded_ms))
    else:
        lag_frames = int(round(rounded_ms * fps_val / 1000.0))
    return lag_frames, rounded_ms


def make_target_cache_key(
    *,
    split: str,
    video_id: str,
    lag_ms: float,
    fps: float,
    frames: int,
    tolerance: float,
    dilation: int,
    canonical_hw: Sequence[int],
    canonicalize: bool = True,
    scheme: str = "frame",
) -> Tuple[str, FrameTargetMeta]:
    """Serialise cache key inputs and return a SHA1 hash plus metadata.

    Parameters
    ----------
    canonicalize:
        When ``True`` (default), ``video_id`` is converted to ``video_###`` canonical
        form for the cache key.  Pass ``False`` to construct compatibility keys that
        preserve legacy identifiers such as ``video_106.0``.
    scheme:
        ``"frame"`` (default) rounds lag to frame units and is the canonical key
        generator.  ``"legacy_ms"`` matches the historic millisecond-based rounding
        to locate old cache entries.
    """

    hw_values = list(canonical_hw)
    if len(hw_values) < 2:
        raise ValueError("canonical_hw must provide at least (H, W)")
    hw_tuple = (int(hw_values[0]), int(hw_values[1]))
    video_key = canonical_video_id(video_id) if canonicalize else str(video_id)
    scheme_key = str(scheme or "frame").lower()
    if scheme_key == "frame":
        lag_frames, lag_ms_value = _round_lag_frames(lag_ms, fps)
    elif scheme_key == "legacy_ms":
        lag_frames, lag_ms_value = _round_lag_ms_legacy(lag_ms, fps)
    else:
        raise ValueError(f"Unknown cache key scheme: {scheme}")
    key_data: FrameTargetMeta = {
        "split": str(split),
        "video_id": video_key,
        "lag_frames": int(lag_frames),
        "lag_ms": float(lag_ms_value),
        "fps": float(fps),
        "frames": int(frames),
        "tolerance": float(tolerance),
        "dilation": int(dilation),
        "canonical_hw": hw_tuple,
    }
    key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    key_hash = hashlib.sha1(key_json.encode("utf-8")).hexdigest()
    return key_hash, key_data


# Backwards compatibility alias for older imports
make_frame_target_cache_key = make_target_cache_key


class FrameTargetCache:
    """Disk-backed cache storing frame target tensors per configuration."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        default_dir = project_root / "runs" / "frame_targets"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key_hash: str) -> Path:
        return self.cache_dir / f"{key_hash}.pt"

    def load(self, key_hash: str) -> Tuple[Optional[Dict[str, torch.Tensor]], bool]:
        """Return cached tensors and whether the cache file existed."""

        path = self._path_for(key_hash)
        if not path.exists():
            return None, False
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Failed to load frame target cache %s (%s)", path, exc)
            return None, False

        data = None
        if isinstance(payload, dict):
            data = payload.get("data", payload)
        if not isinstance(data, dict):
            LOGGER.warning("Frame target cache %s missing data payload", path)
            return None, True

        tensors: Dict[str, torch.Tensor] = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                tensors[key] = value.clone()
        return tensors, True

    def save(
        self,
        key_hash: str,
        metadata: Mapping[str, object],
        data: Mapping[str, torch.Tensor],
    ) -> bool:
        """Persist tensors for ``key_hash``; return ``True`` on success."""

        path = self._path_for(key_hash)
        payload = {
            "meta": dict(metadata),
            "data": {
                key: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
                for key, tensor in data.items()
            },
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            return True
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Failed to write frame target cache %s (%s)", path, exc)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return False
