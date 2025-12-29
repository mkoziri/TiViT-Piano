"""
Purpose:
    Provide A/V synchronization utilities with a single front-door API for the
    new data layout. Centralises lag resolution and application.

Conventions:
    lag_seconds > 0 means audio is ahead of video; event times should be shifted
    by +lag_seconds to align with video frames.

Key Functions/Classes:
    - SyncInfo: Captures lag_seconds and source metadata.
    - resolve_sync(): Resolve lag from metadata/cache.
    - apply_sync(): Idempotently shift events and stamp sync info.

CLI Arguments:
    (none)

Usage:
    sync = resolve_sync(video_id, sample_meta, lag_cache)
    sample = apply_sync(sample, sync)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional

try:
    from tivit.data.targets.av_sync import AVLagCache  # type: ignore
except Exception:  # pragma: no cover - optional
    AVLagCache = None  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class SyncInfo:
    """Lag metadata for a sample or video."""

    lag_seconds: float
    source: str = "default"


def resolve_sync(video_id: str, sample_meta: Mapping[str, Any] | None = None, lag_cache: Any | None = None) -> SyncInfo:
    """
    Determine lag for ``video_id``.

    Uses metadata fields when present:
      - ``lag_seconds`` or ``lag_sec``
      - ``lag_ms`` (converted to seconds)
    """

    lag_seconds = 0.0
    source = "default"
    # Prefer explicit metadata when available.
    if isinstance(sample_meta, Mapping):
        if "lag_seconds" in sample_meta:
            try:
                lag_seconds = float(sample_meta["lag_seconds"])
                source = "meta_seconds"
            except Exception:
                pass
        elif "lag_sec" in sample_meta:
            try:
                lag_seconds = float(sample_meta["lag_sec"])
                source = "meta_seconds"
            except Exception:
                pass
        elif "lag_ms" in sample_meta:
            try:
                lag_seconds = float(sample_meta["lag_ms"]) / 1000.0
                source = "meta_ms"
            except Exception:
                pass
        # If metadata carries lag_ms, persist to cache for future lookups.
        if lag_cache is not None and hasattr(lag_cache, "set") and "lag_ms" in sample_meta:
            try:
                lag_cache.set(video_id, float(sample_meta["lag_ms"]))
            except Exception:
                pass
    # Fall back to cache when no metadata override was found.
    if source == "default" and lag_cache is not None and hasattr(lag_cache, "get"):
        try:
            lag_ms = lag_cache.get(video_id)
            if lag_ms is not None:
                lag_seconds = float(lag_ms) / 1000.0
                source = "cache_ms"
        except Exception:
            pass
    return SyncInfo(lag_seconds=lag_seconds, source=source)


def apply_sync(sample: Optional[MutableMapping[str, Any]], sync: SyncInfo) -> Optional[MutableMapping[str, Any]]:
    """
    Apply lag to sample labels in-place; idempotent via a flag.

    Expects event labels under ``sample.get('events')`` (list of (onset, offset, pitch)).
    """

    if sample is None:
        return sample
    if getattr(sample, "_sync_applied", False) or sample.get("_sync_applied"):
        return sample

    events = sample.get("events")
    if isinstance(events, list):
        shifted = []
        for evt in events:
            if not isinstance(evt, (tuple, list)) or len(evt) < 3:
                continue
            onset, offset, pitch = evt[:3]
            shifted.append((float(onset) + sync.lag_seconds, float(offset) + sync.lag_seconds, int(pitch)))
        sample["events"] = shifted

    sample["_sync_applied"] = True
    sample["_sync_source"] = sync.source
    return sample


__all__ = ["SyncInfo", "resolve_sync", "apply_sync"]
