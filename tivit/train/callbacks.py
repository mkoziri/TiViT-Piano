"""Training callbacks (minimal scaffolding)."""

from __future__ import annotations

from typing import Any


class Callback:
    def on_epoch_start(self, epoch: int, **_: Any) -> None: ...

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any] | None = None, **_: Any) -> None: ...

    def on_train_end(self, **_: Any) -> None: ...


__all__ = ["Callback"]

