"""Frame probabilities → events decoding helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch

from tivit.decoder.decode import (
    DECODER_DEFAULTS,
    decode_hysteresis,
    normalize_decoder_params,
    resolve_decoder_from_config,
    resolve_decoder_gates,
)


def build_decoder(metrics_cfg: Mapping[str, Any] | None = None):
    normalized = resolve_decoder_from_config(metrics_cfg or {})
    gates = resolve_decoder_gates(normalized)

    def _decode(probs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        decoded = {}
        for head, cfg in gates.items():
            roll = probs.get(head)
            if roll is None:
                continue
            decoded[head] = decode_hysteresis(
                roll,
                open_thr=cfg["open"],
                hold_thr=cfg["hold"],
                min_on=cfg["min_on"],
                min_off=cfg["min_off"],
                merge_gap=cfg["merge_gap"],
                median=cfg["median"],
            )
        return decoded

    return _decode


__all__ = [
    "DECODER_DEFAULTS",
    "decode_hysteresis",
    "normalize_decoder_params",
    "resolve_decoder_from_config",
    "resolve_decoder_gates",
    "build_decoder",
]

