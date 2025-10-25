"""Disk-backed cache for deterministic evaluation window snapshots."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

CACHE_DIR = Path("runs") / "cache"
CACHE_PATH = CACHE_DIR / "eval_windows.json"
_LOCK_TIMEOUT = 1.0


def _lock_path() -> Path:
    return CACHE_PATH.with_suffix(".lock")


def _acquire_lock(timeout: float = _LOCK_TIMEOUT) -> bool:
    deadline = time.perf_counter() + timeout
    lock = _lock_path()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.perf_counter() >= deadline:
                return False
            time.sleep(0.05)


def _release_lock() -> None:
    lock = _lock_path()
    try:
        os.unlink(lock)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _normalise_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    return json.loads(json.dumps(metadata, sort_keys=True, default=str))


def _fingerprint(metadata: Mapping[str, Any]) -> str:
    normalised = _normalise_metadata(metadata)
    payload = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    # Use a short hash to keep JSON compact; collisions are extremely unlikely.
    import hashlib

    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    with CACHE_PATH.open("r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    return data


def _dump_cache(cache: Mapping[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, CACHE_PATH)


class EvalSnapshot(Tuple[Sequence[Tuple[int, int]], Sequence[bool], Mapping[int, Sequence[int]], Mapping[str, Any]]):
    __slots__ = ()


def load_snapshot(metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Return cached snapshot payload when metadata fingerprint matches."""

    if not _acquire_lock():
        return None
    try:
        cache = _load_cache()
        fingerprint = _fingerprint(metadata)
        entry = cache.get(fingerprint)
        if not isinstance(entry, Mapping):
            return None
        payload = dict(entry)
        entries = payload.get("entries")
        flags = payload.get("flags")
        if not isinstance(entries, list) or not isinstance(flags, list):
            return None
        payload["entries"] = [(int(rec), int(start)) for rec, start in entries]
        payload["flags"] = [bool(flag) for flag in flags]
        candidates = payload.get("candidates") or {}
        payload["candidates"] = {int(k): [int(vv) for vv in v] for k, v in candidates.items()}
        by_video = payload.get("by_video") or {}
        payload["by_video"] = {int(k): [int(vv) for vv in v] for k, v in by_video.items()}
        stats = payload.get("stats") or {}
        payload["stats"] = {str(k): stats[k] for k in stats}
        payload["metadata"] = metadata
        return payload
    finally:
        _release_lock()


def save_snapshot(metadata: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Persist evaluation snapshot data keyed by metadata fingerprint."""

    snapshot = {
        "metadata": _normalise_metadata(metadata),
        "entries": [(int(rec), int(start)) for rec, start in payload.get("entries", [])],
        "flags": [bool(flag) for flag in payload.get("flags", [])],
        "candidates": {str(int(k)): [int(vv) for vv in v] for k, v in (payload.get("candidates") or {}).items()},
        "by_video": {str(int(k)): [int(vv) for vv in v] for k, v in (payload.get("by_video") or {}).items()},
        "stats": payload.get("stats") or {},
        "duration": float(payload.get("duration", 0.0)),
    }

    if not _acquire_lock():
        return
    try:
        cache = _load_cache()
        cache[_fingerprint(metadata)] = snapshot
        _dump_cache(cache)
    finally:
        _release_lock()


__all__ = [
    "CACHE_PATH",
    "EvalSnapshot",
    "load_snapshot",
    "save_snapshot",
]
