"""Cache interface used in the new layout."""

from __future__ import annotations

from typing import Any, Callable, Hashable, Tuple

from tivit.decoder.tile_support_cache import CacheScope, TileSupportCache


class Cache(TileSupportCache):
    """Alias for the existing tile/key cache."""

    def get_or_compute_value(self, scope: CacheScope, key: Hashable, fn: Callable[[], Any]) -> Tuple[Any, bool]:
        return self.get_or_compute(scope, key, fn)


__all__ = ["Cache", "CacheScope", "TileSupportCache"]

