"""TiViT-Piano top-level package shim.

This repository historically stored its subpackages (such as
``theory``) directly at the repository root. To offer a clean import path like
``import tivit.theory`` without rearranging the on-disk layout we populate the
``tivit`` module dynamically, aliasing the real implementation modules.

The module re-exports the most frequently used theory helpers so callers can
write ``from tivit import KeyAwarePrior`` or ``from tivit.theory import ...``
after simply adding the repository root to ``PYTHONPATH``.
"""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType
from typing import Final, TYPE_CHECKING


def _import_theory() -> ModuleType:
    """Load the underlying :mod:`theory` package and register it."""

    module = import_module("theory")
    sys.modules.setdefault(__name__ + ".theory", module)
    return module


if TYPE_CHECKING:  # pragma: no cover - typing helper
    from theory.key_prior import KeyAwarePrior as _KeyAwarePrior
    from theory.key_prior import KeyPriorConfig as _KeyPriorConfig
    from theory.key_prior import build_key_profiles as _build_key_profiles
    from theory.key_prior_runtime import (
        KeyPriorRuntimeSettings as _KeyPriorRuntimeSettings,
        resolve_key_prior_settings as _resolve_key_prior_settings,
        apply_key_prior_to_logits as _apply_key_prior_to_logits,
    )

    _THEORY: Final[ModuleType] = import_module("theory")
    theory: ModuleType = _THEORY
    KeyAwarePrior = _KeyAwarePrior
    KeyPriorConfig = _KeyPriorConfig
    build_key_profiles = _build_key_profiles
    KeyPriorRuntimeSettings = _KeyPriorRuntimeSettings
    resolve_key_prior_settings = _resolve_key_prior_settings
    apply_key_prior_to_logits = _apply_key_prior_to_logits
else:
    _THEORY: Final[ModuleType] = _import_theory()
    theory: ModuleType = _THEORY
    KeyAwarePrior = _THEORY.KeyAwarePrior
    KeyPriorConfig = _THEORY.KeyPriorConfig
    build_key_profiles = _THEORY.build_key_profiles
    KeyPriorRuntimeSettings = _THEORY.KeyPriorRuntimeSettings
    resolve_key_prior_settings = _THEORY.resolve_key_prior_settings
    apply_key_prior_to_logits = _THEORY.apply_key_prior_to_logits

__all__ = [
    "theory",
    "KeyAwarePrior",
    "KeyPriorConfig",
    "build_key_profiles",
    "KeyPriorRuntimeSettings",
    "resolve_key_prior_settings",
    "apply_key_prior_to_logits",
]


def __getattr__(name: str) -> object:
    """Delegate attribute access to :mod:`theory` for convenience."""

    try:
        return getattr(_THEORY, name)
    except AttributeError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc


def __dir__() -> list[str]:  # pragma: no cover - trivial
    """Expose dynamic attributes to :func:`dir`."""

    return sorted(set(__all__) | set(dir(_THEORY)))
