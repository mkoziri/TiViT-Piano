"""Utilities for persisting per-clip frame target tensors.

Purpose:
    Provide a lightweight disk cache keyed by dataset/video properties so the
    expensive per-frame target construction only runs when required.  The cache
    mirrors the behaviour of :class:`utils.av_sync.AVLagCache` but stores torch
    tensors representing pianoroll labels for a particular combination of
    split, video, temporal alignment, and frame-target configuration.

Usage:
    >>> cache = FrameTargetCache()
    >>> key_hash, meta = make_frame_target_cache_key(...)
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
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)


def make_frame_target_cache_key(
    *,
    split: str,
    video_id: str,
    lag_ms: float,
    fps: float,
    frames: int,
    tolerance: float,
    dilation: int,
    canonical_hw: Sequence[int],
) -> Tuple[str, Dict[str, object]]:
    """Serialise cache key inputs and return a SHA1 hash plus metadata.

    The ``lag_ms`` component is rounded to the nearest millisecond before
    hashing so that minor floating point jitter does not cause unnecessary
    cache misses.
    """

    hw_values = list(canonical_hw)
    if len(hw_values) < 2:
        raise ValueError("canonical_hw must provide at least (H, W)")
    hw_tuple = (int(hw_values[0]), int(hw_values[1]))
    key_data: Dict[str, object] = {
        "split": str(split),
        "video_id": str(video_id),
        "lag_ms": int(round(float(lag_ms))),
        "fps": float(fps),
        "frames": int(frames),
        "tolerance": float(tolerance),
        "dilation": int(dilation),
        "canonical_hw": hw_tuple,
    }
    key_json = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    key_hash = hashlib.sha1(key_json.encode("utf-8")).hexdigest()
    return key_hash, key_data


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
        self, key_hash: str, metadata: Mapping[str, object], data: Mapping[str, torch.Tensor]
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
        try:
            torch.save(payload, path)
            return True
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.warning("Failed to write frame target cache %s (%s)", path, exc)
            return False