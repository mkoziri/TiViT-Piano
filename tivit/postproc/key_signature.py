"""Key signature prior wrapper for decoding.

Purpose:
    - Wrap the theory key-prior runtime to rescore logits for selected heads.
    - Provide a small Prior-compatible adapter for post-processing use.

Key Functions/Classes:
    - KeySignaturePrior: Applies the key-aware prior to logits when enabled.
    - build_prior(): Factory that returns a KeySignaturePrior.

CLI Arguments:
    (none)

Usage:
    prior = build_prior(cfg)
    adjusted = prior.apply_to_logits({"onset": onset_logits, "offset": offset_logits})
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from theory.key_prior_runtime import apply_key_prior_to_logits, resolve_key_prior_settings
from tivit.priors.base import Prior


class KeySignaturePrior(Prior):
    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        cfg.setdefault("enabled", True)
        self.settings = resolve_key_prior_settings(cfg)
        self.fps = _coerce_float(cfg.get("fps"), 30.0) or 30.0
        self.midi_low = _coerce_int(cfg.get("midi_low"))
        self.midi_high = _coerce_int(cfg.get("midi_high"))

    def apply_to_logits(self, logits: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if not self.settings.enabled:
            return logits
        try:
            adjusted = apply_key_prior_to_logits(
                logits,
                self.settings,
                fps=float(self.fps),
                midi_low=self.midi_low,
                midi_high=self.midi_high,
            )
        except Exception:
            # Defensive fallback: never break decoding because of optional prior.
            return logits
        return adjusted if adjusted is not None else logits


def build_prior(cfg: Mapping[str, Any] | None = None) -> Prior:
    return KeySignaturePrior(cfg)


def _coerce_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_int(value: object, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


__all__ = ["KeySignaturePrior", "build_prior"]
