"""Dataset adapters wrapped for the unified layout."""

from importlib import import_module
from typing import Any, Callable

from .base import DatasetAdapter


def _lazy_builder(module_name: str) -> Callable[..., Any]:
    def _builder(*args: Any, **kwargs: Any) -> Any:
        module = import_module(f".{module_name}", __name__)
        return module.build_dataset(*args, **kwargs)

    return _builder


build_omaps = _lazy_builder("omaps")
build_pianoyt = _lazy_builder("pianoyt")
build_pianovam = _lazy_builder("pianovam")

__all__ = ["DatasetAdapter", "build_omaps", "build_pianoyt", "build_pianovam"]
