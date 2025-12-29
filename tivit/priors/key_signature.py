"""Key signature prior wrapper."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from theory.key_prior_runtime import apply_key_prior_to_logits, resolve_key_prior_settings
from .base import Prior


class KeySignaturePrior(Prior):
    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.settings = resolve_key_prior_settings(cfg or {})

    def apply_to_logits(self, logits: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        try:
            adjusted = apply_key_prior_to_logits(logits, self.settings)
            return adjusted if adjusted is not None else logits
        except Exception:
            # Defensive fallback: never break decoding because of optional prior.
            return logits


def build_prior(cfg: Mapping[str, Any] | None = None) -> Prior:
    return KeySignaturePrior(cfg)


__all__ = ["KeySignaturePrior", "build_prior"]

