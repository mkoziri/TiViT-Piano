"""In-memory cache for tile/key support data shared across train and eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Literal, Optional, Tuple

from tivit.data.targets.identifiers import canonical_video_id

CacheScope = Literal["train", "eval"]
_FALLBACK_VIDEO_UID = "__fallback__"
_KEY_VERSION = "v1"


def _normalise_hw(hw: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if hw is None:
        return None
    try:
        h, w = hw
        return int(h), int(w)
    except Exception:
        return None


def make_tile_cache_key(
    video_uid: Optional[str],
    *,
    num_tiles: int,
    cushion_keys: int,
    n_keys: int,
    canonical_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Hashable, ...]:
    """
    Build a deterministic cache key for tile/key support lookups.

    The key is anchored to a stable ``video_uid`` rather than per-clip IDs so
    multiple windows from the same source reuse entries. Only configuration
    knobs that alter the mask layout are included.
    """

    stable_uid = canonical_video_id(video_uid) if video_uid is not None else ""
    if not stable_uid:
        stable_uid = _FALLBACK_VIDEO_UID
    return (
        _KEY_VERSION,
        stable_uid,
        int(num_tiles),
        int(cushion_keys),
        int(n_keys),
        _normalise_hw(canonical_hw),
    )


@dataclass
class _CacheCounts:
    shared: int
    eval: int


class TileSupportCache:
    """Two-tier cache with a persistent shared layer and an eval-only layer."""

    def __init__(self) -> None:
        self._shared: Dict[Hashable, Any] = {}
        self._eval: Dict[Hashable, Any] = {}

    @staticmethod
    def _validate_scope(scope: CacheScope) -> CacheScope:
        if scope not in ("train", "eval"):
            raise ValueError(f"Unknown cache scope '{scope}' (expected 'train' or 'eval')")
        return scope

    def get(self, scope: CacheScope, key: Hashable) -> Any:
        """Retrieve from shared (always) plus eval layer when scope='eval'."""

        self._validate_scope(scope)
        if key in self._shared:
            return self._shared[key]
        if scope == "eval":
            return self._eval.get(key)
        return None

    def put(self, scope: CacheScope, key: Hashable, value: Any) -> Any:
        """Store ``value`` in the shared or eval layer depending on ``scope``."""

        scope = self._validate_scope(scope)
        if scope == "train":
            self._shared[key] = value
        else:
            self._eval[key] = value
        return value

    def get_or_compute(
        self,
        scope: CacheScope,
        key: Hashable,
        compute_fn: Callable[[], Any],
    ) -> Tuple[Any, bool]:
        """
        Return cached value when present else compute/store it.

        Returns:
            (value, from_cache)
        """

        cached = self.get(scope, key)
        if cached is not None:
            return cached, True
        value = compute_fn()
        self.put(scope, key, value)
        return value, False

    def clear_eval(self) -> int:
        """Drop eval-layer entries while leaving the shared layer intact."""

        count = len(self._eval)
        self._eval.clear()
        return count

    def clear_shared(self) -> int:
        """Drop shared-layer entries (training cache)."""

        count = len(self._shared)
        self._shared.clear()
        return count

    def clear_all(self) -> _CacheCounts:
        """Drop both cache layers and return how many entries were cleared."""

        cleared = _CacheCounts(shared=len(self._shared), eval=len(self._eval))
        self._shared.clear()
        self._eval.clear()
        return cleared

    def counts(self) -> _CacheCounts:
        """Return entry counts for logging/debug."""

        return _CacheCounts(shared=len(self._shared), eval=len(self._eval))


__all__ = ["TileSupportCache", "make_tile_cache_key", "CacheScope"]
