"""Per-video cache helpers built on top of the shared tile cache."""

from __future__ import annotations

from typing import Any, Callable, Hashable, Tuple

from tivit.data.targets.identifiers import canonical_video_id
from .cache_api import Cache, CacheScope


class VideoCache:
    """Two-scope cache (train/eval) that normalises video IDs."""

    def __init__(self) -> None:
        self._cache = Cache()

    def _key(self, video_uid: str | Hashable, *parts: Hashable) -> Tuple[Hashable, ...]:
        return (canonical_video_id(str(video_uid)),) + tuple(parts)

    def get(self, scope: CacheScope, video_uid: str | Hashable, *parts: Hashable) -> Any:
        return self._cache.get(scope, self._key(video_uid, *parts))

    def put(self, scope: CacheScope, video_uid: str | Hashable, value: Any, *parts: Hashable) -> Any:
        return self._cache.put(scope, self._key(video_uid, *parts), value)

    def get_or_compute(
        self,
        scope: CacheScope,
        video_uid: str | Hashable,
        compute_fn: Callable[[], Any],
        *parts: Hashable,
    ) -> Tuple[Any, bool]:
        return self._cache.get_or_compute(scope, self._key(video_uid, *parts), compute_fn)

    def clear_eval(self) -> int:
        return self._cache.clear_eval()

    def clear_shared(self) -> int:
        return self._cache.clear_shared()

    def clear_all(self):
        return self._cache.clear_all()

    def counts(self):
        return self._cache.counts()


__all__ = ["VideoCache"]
