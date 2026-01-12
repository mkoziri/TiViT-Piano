"""AMP helpers with version-compatible fallbacks."""

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


def autocast(device: torch.device, *, enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """Return a compatible autocast context manager for the device."""
    if _AUTOCAST_DEVICE_TYPE:
        return cast(contextlib.AbstractContextManager[Any], torch.autocast(device_type=device.type, enabled=enabled))
    if device.type == "cuda":
        return cast(contextlib.AbstractContextManager[Any], torch.cuda.amp.autocast(enabled=enabled))
    return contextlib.nullcontext()


def grad_scaler(device: torch.device, *, enabled: bool):
    """Return a GradScaler instance that matches the installed torch API."""
    if _SCALER_FN is None:
        raise RuntimeError("GradScaler is unavailable in the installed torch package.")
    if _SCALER_DEVICE_TYPE:
        return _SCALER_FN(device_type=device.type, enabled=enabled)
    return _SCALER_FN(enabled=enabled)


__all__ = ["autocast", "grad_scaler"]
