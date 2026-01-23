"""AMP helpers with version-compatible fallbacks.

Purpose:
    - Provide autocast and GradScaler wrappers that work across torch versions.
    - Offer a safe no-op scaler when AMP is disabled or unsupported.

Key Functions/Classes:
    - autocast(): Return a compatible autocast context manager.
    - grad_scaler(): Return a GradScaler or no-op fallback.

CLI:
    None. Import from training or evaluation loops.
"""

from __future__ import annotations

import contextlib
import inspect
from typing import Any, cast

import torch


def _supports_device_type(target) -> bool:
    try:
        return "device_type" in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


_AUTOCAST_FN = getattr(torch, "autocast", None)
_AUTOCAST_DEVICE_TYPE = _supports_device_type(_AUTOCAST_FN) if _AUTOCAST_FN else False
_AMP_MODULE = getattr(torch, "amp", None)
_SCALER_FN = getattr(_AMP_MODULE, "GradScaler", None) if _AMP_MODULE is not None else None
if _SCALER_FN is None:
    _SCALER_FN = getattr(torch.cuda.amp, "GradScaler", None)
_SCALER_DEVICE_TYPE = _supports_device_type(_SCALER_FN) if _SCALER_FN else False


class _NoOpGradScaler:
    """Minimal scaler interface used when AMP is disabled or unavailable."""

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, *args, **kwargs) -> None:
        return None

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state) -> None:
        return None

    def is_enabled(self) -> bool:
        return False


def _coerce_device(device: torch.device | str | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def autocast(device: torch.device | str | None, *, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """Return a compatible autocast context manager for the device."""
    device = _coerce_device(device)
    if _AUTOCAST_DEVICE_TYPE:
        return cast(contextlib.AbstractContextManager[Any], torch.autocast(device_type=device.type, enabled=enabled))
    if device.type == "cuda":
        return cast(contextlib.AbstractContextManager[Any], torch.cuda.amp.autocast(enabled=enabled))
    return contextlib.nullcontext()


def grad_scaler(device: torch.device | str | None, *, enabled: bool):
    """Return a GradScaler instance that matches the installed torch API."""
    if not enabled:
        return _NoOpGradScaler()
    if _SCALER_FN is None:
        raise RuntimeError("GradScaler is unavailable in the installed torch package.")
    device = _coerce_device(device)
    if _SCALER_DEVICE_TYPE:
        return _SCALER_FN(device_type=device.type, enabled=enabled)
    if device.type == "cuda":
        return _SCALER_FN(enabled=enabled)
    return _NoOpGradScaler()


__all__ = ["autocast", "grad_scaler"]
