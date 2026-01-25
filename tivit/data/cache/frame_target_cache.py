"""Purpose:
    Persist frame-target tensors for specific video/lag configurations so
    datasets only rebuild targets when inputs change.

Key Functions/Classes:
    - FrameTargetCache: Disk-backed helper with load/save primitives.
    - make_target_cache_key(): Generates stable SHA1 keys and metadata blobs.
    - FrameTargetMeta: TypedDict describing stored cache metadata.

CLI:
    Not a standalone CLI; used by dataset loaders when preparing targets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, TypedDict

import torch

from tivit.data.targets.identifiers import canonical_video_id

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

    from tivit.data.targets.av_sync import round_lag_ms_for_cache  # local import to avoid cycle

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
    """Serialise cache key inputs and return a SHA1 hash plus metadata."""

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
        project_root = Path(__file__).resolve().parents[3]
        default_dir = project_root / "runs" / "frame_targets"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock_timeout_logged = False

    def _path_for(self, key_hash: str) -> Path:
        return self.cache_dir / f"{key_hash}.pt"

    def _lock_path(self, key_hash: str) -> Path:
        return self._path_for(key_hash).with_suffix(".lock")

    def _acquire_lock(self, lock_path: Path, timeout: float) -> bool:
        if timeout is None or timeout < 0:
            timeout = 0.0
        deadline = time.perf_counter() + timeout if timeout > 0 else None
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                if deadline is None or time.perf_counter() >= deadline:
                    return False
                time.sleep(0.05)

    def _release_lock(self, lock_path: Path) -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def load(self, key_hash: str) -> Tuple[Optional[Dict[str, torch.Tensor]], bool]:
        """Return cached tensors and whether the cache file existed."""

        path = self._path_for(key_hash)
        lock_path = self._lock_path(key_hash)
        if not self._acquire_lock(lock_path, 1.0):
            if not self._lock_timeout_logged:
                LOGGER.warning("Frame target cache lock timeout for key=%s; skipping load", key_hash[:8])
                self._lock_timeout_logged = True
            return None, False
        payload = None
        try:
            if not path.exists():
                return None, False
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.warning("Failed to load frame target cache %s (%s)", path, exc)
                return None, False
        finally:
            self._release_lock(lock_path)

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

    def save(self, key_hash: str, metadata: Mapping[str, object], data: Mapping[str, torch.Tensor]) -> bool:
        """Persist tensors for ``key_hash``; return ``True`` on success."""

        path = self._path_for(key_hash)
        lock_path = self._lock_path(key_hash)
        if not self._acquire_lock(lock_path, 1.0):
            if not self._lock_timeout_logged:
                LOGGER.warning("Frame target cache lock timeout for key=%s; skipping save", key_hash[:8])
                self._lock_timeout_logged = True
            return False
        payload = {
            "meta": dict(metadata),
            "data": {key: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor for key, tensor in data.items()},
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
        finally:
            self._release_lock(lock_path)


class NullFrameTargetCache(FrameTargetCache):
    """Cache stub that never persists frame-target tensors."""

    def __init__(self) -> None:
        self.cache_dir = Path(".")
        self._lock_timeout_logged = False

    def load(self, _: str) -> Tuple[Optional[Dict[str, torch.Tensor]], bool]:
        return None, False

    def save(self, *_: object, **__: object) -> bool:
        return False
