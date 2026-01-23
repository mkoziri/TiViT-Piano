"""Lightweight registries for datasets, models, heads, losses, priors, and postproc.

Purpose:
- Provide a simple name→callable registry with lazy module loading.
- Expose default registrations for built-in datasets/models/pipeline pieces.
- Keep decoding-only priors in postproc; this registry targets training-time components.

Key Functions/Classes:
- Registry: Core registry type with register/get/build helpers.
- register_default_components: Populate all registries with built-ins.

CLI Arguments:
- (none; import-only utilities).

Usage:
- Import and resolve components: ``from tivit.core.registry import MODELS, register_default_components``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Tuple, TypeVar

T = TypeVar("T")


class Registry:
    """Simple name→callable registry with lowercase keys."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any] | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a callable under ``name``; usable as decorator or direct call."""
        key = name.lower().strip()

        def _decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            self._items[key] = target
            return target

        if fn is not None:
            return _decorator(fn)
        return _decorator

    def get(self, name: str) -> Callable[..., Any]:
        """Return the callable registered under ``name`` (case-insensitive)."""
        key = name.lower().strip()
        if key not in self._items:
            raise KeyError(f"{self.kind} '{name}' is not registered")
        return self._items[key]

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a registered builder with ``*args``/``**kwargs``."""
        return self.get(name)(*args, **kwargs)

    def available(self) -> Tuple[str, ...]:
        """List registered names sorted alphabetically."""
        return tuple(sorted(self._items.keys()))


DATASETS = Registry("dataset")
MODELS = Registry("model")
HEADS = Registry("head")
LOSSES = Registry("loss")
PRIORS = Registry("prior")
POSTPROC = Registry("postproc")

_DEFAULTS_LOADED = False


def _lazy(path: str, attr: str) -> Callable[..., Any]:
    """Delay module import until the registered callable is invoked."""
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        module = import_module(path)
        return getattr(module, attr)(*args, **kwargs)

    return _wrapped


def _register_dataset_defaults() -> None:
    """Register built-in datasets."""
    DATASETS.register("omaps")(_lazy("tivit.data.datasets.omaps", "build_dataset"))
    DATASETS.register("pianoyt")(_lazy("tivit.data.datasets.pianoyt", "build_dataset"))
    DATASETS.register("pianovam")(_lazy("tivit.data.datasets.pianovam", "build_dataset"))


def _register_model_defaults() -> None:
    """Register built-in backbones."""
    MODELS.register("vivit")(_lazy("tivit.models.backbones.vivit", "build_model"))
    MODELS.register("vit_small")(_lazy("tivit.models.backbones.vit_small", "build_model"))
    MODELS.register("vits_tile")(_lazy("tivit.models.backbones.vit_small", "build_model"))


def _register_head_defaults() -> None:
    """Register built-in heads."""
    HEADS.register("onset")(_lazy("tivit.models.heads.onset", "build_head"))
    HEADS.register("offset")(_lazy("tivit.models.heads.offset", "build_head"))
    HEADS.register("pitch")(_lazy("tivit.models.heads.pitch", "build_head"))
    HEADS.register("hand")(_lazy("tivit.models.heads.hand", "build_head"))
    HEADS.register("clef")(_lazy("tivit.models.heads.clef", "build_head"))


def _register_loss_defaults() -> None:
    """Register built-in loss factories."""
    LOSSES.register("multitask")(_lazy("tivit.losses.multitask_loss", "build_loss"))
    LOSSES.register("bce")(_lazy("tivit.losses.bce", "build_loss"))
    LOSSES.register("focal")(_lazy("tivit.losses.focal", "build_loss"))


def _register_prior_defaults() -> None:
    """Register built-in priors."""
    PRIORS.register("hand_gating")(_lazy("tivit.priors.hand_gating", "build_prior"))
    PRIORS.register("chord_smoothness")(_lazy("tivit.priors.chord_smoothness", "build_prior"))
    PRIORS.register("none")(_lazy("tivit.priors.base", "null_prior"))


def _register_postproc_defaults() -> None:
    """Register built-in post-processing steps."""
    POSTPROC.register("thresholding")(_lazy("tivit.postproc.thresholding", "build_postproc"))
    POSTPROC.register("event_decode")(_lazy("tivit.postproc.event_decode", "build_decoder"))
    POSTPROC.register("min_note_length")(_lazy("tivit.postproc.min_note_length", "build_constraint"))
    POSTPROC.register("harmony_filter")(_lazy("tivit.postproc.harmony_filter", "build_constraint"))


def register_default_components() -> None:
    """Populate registries with built-in components (idempotent)."""

    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    _register_dataset_defaults()
    _register_model_defaults()
    _register_head_defaults()
    _register_loss_defaults()
    _register_prior_defaults()
    _register_postproc_defaults()
    _DEFAULTS_LOADED = True


__all__ = [
    "Registry",
    "DATASETS",
    "MODELS",
    "HEADS",
    "LOSSES",
    "PRIORS",
    "POSTPROC",
    "register_default_components",
]
