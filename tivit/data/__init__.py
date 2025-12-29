"""Data loading facade aligned with the new package layout."""

from importlib import import_module
from typing import Any

from .loaders import make_dataloader

_LAZY_DATASETS = {"omaps": "tivit.data.datasets.omaps", "pianoyt": "tivit.data.datasets.pianoyt", "pianovam": "tivit.data.datasets.pianovam"}


def __getattr__(name: str) -> Any:
    """Lazily import dataset modules only when accessed."""

    if name in _LAZY_DATASETS:
        return import_module(_LAZY_DATASETS[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["make_dataloader"] + sorted(_LAZY_DATASETS.keys())
