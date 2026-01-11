"""Training callbacks (minimal scaffolding)."""

# Purpose:
# - Define a lightweight callback interface for training lifecycle hooks.
# - Provide a container that fans out events to multiple callbacks.
#
# Key Functions/Classes:
# - Callback: base class with epoch/train-end hooks.
# - CallbackList: composite that forwards hooks to child callbacks.
#
# CLI Arguments:
# - (none; import-only utilities).
#
# Usage:
# - from tivit.train.callbacks import CallbackList

from __future__ import annotations

from typing import Any, Iterable, Sequence


class Callback:
    def on_epoch_start(self, epoch: int, **_: Any) -> None: ...

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any] | None = None, **_: Any) -> None: ...

    def on_train_end(self, **_: Any) -> None: ...


class CallbackList(Callback):
    """Lightweight container that forwards events to child callbacks."""

    def __init__(self, callbacks: Iterable[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = list(callbacks or [])

    def on_epoch_start(self, epoch: int, **kwargs: Any) -> None:  # type: ignore[override]
        for cb in self.callbacks:
            cb.on_epoch_start(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:  # type: ignore[override]
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics=metrics, **kwargs)

    def on_train_end(self, **kwargs: Any) -> None:  # type: ignore[override]
        for cb in self.callbacks:
            cb.on_train_end(**kwargs)


__all__ = ["Callback", "CallbackList"]
