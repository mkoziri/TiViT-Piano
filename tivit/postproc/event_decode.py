"""Event decoding helpers (logits/probs → event masks).

Purpose:
    - Normalize decoder config and build hysteresis gates for onset/offset heads.
    - Accept logits or probabilities and optionally apply the key prior when a full config is provided.
    - Provide a compact decoder callable for evaluation and post-processing paths.

Key Functions/Classes:
    - build_decoder(): Return a decoder callable from a full config or metrics-only mapping.
    - build_decoder_from_logits(): Convenience wrapper for logits input with key-prior support.
    - resolve_decoder_from_config(), resolve_decoder_gates(): Normalization helpers used by downstream code.

CLI Arguments:
    (none)

Usage:
    decoder = build_decoder(cfg)  # full config, logits in
    masks = decoder({"onset_logits": onset_logits, "offset_logits": offset_logits})
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Tuple

import torch

from tivit.decoder.decode import (
    DECODER_DEFAULTS,
    decode_hysteresis,
    normalize_decoder_params,
    resolve_decoder_from_config,
    resolve_decoder_gates,
)
from tivit.postproc.hand_gate_runtime import apply_hand_gate_from_config
from tivit.postproc.key_prior_runtime import apply_key_prior_from_config


def _looks_like_full_config(cfg: Mapping[str, Any]) -> bool:
    return any(key in cfg for key in ("training", "dataset", "priors"))


def _extract_metrics_cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(cfg, Mapping):
        return {}
    if _looks_like_full_config(cfg):
        training_cfg = cfg.get("training", {})
        if isinstance(training_cfg, Mapping):
            metrics_cfg = training_cfg.get("metrics", {})
            if isinstance(metrics_cfg, Mapping):
                return metrics_cfg
        return {}
    return cfg


def _normalize_logits_inputs(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = dict(inputs)
    aliases = {
        "onset_logits": "onset",
        "offset_logits": "offset",
        "pitch_logits": "pitch",
        "hand_logits": "hand",
    }
    for src, dst in aliases.items():
        if dst not in normalized and src in normalized:
            normalized[dst] = normalized[src]
    return normalized


def build_decoder(
    cfg: Mapping[str, Any] | None = None,
    *,
    input_is_logits: bool | None = None,
) -> Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]:
    """Return a decoder function; accepts probabilities or logits based on config."""
    full_cfg = cfg if isinstance(cfg, Mapping) and _looks_like_full_config(cfg) else None
    metrics_cfg = _extract_metrics_cfg(cfg)
    normalized = resolve_decoder_from_config(metrics_cfg or {})
    gates = {}
    for head, cfg in normalized.items():
        defaults = DECODER_DEFAULTS.get(head, {})
        fallback_open = float(cfg.get("open", defaults.get("open", 0.5)))
        default_hold = float(cfg.get("hold", defaults.get("hold", 0.0)))
        open_thr, hold_thr = resolve_decoder_gates(
            cfg,
            fallback_open=fallback_open,
            default_hold=default_hold,
        )
        gates[head] = {**cfg, "open": open_thr, "hold": hold_thr}

    if input_is_logits is None:
        input_is_logits = full_cfg is not None

    def _decode(inputs: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        if input_is_logits:
            logits_map = _normalize_logits_inputs(inputs)
            if full_cfg is not None:
                adjusted = apply_key_prior_from_config(logits_map, full_cfg)
                if adjusted:
                    logits_map.update(adjusted)
            probs = {head: torch.sigmoid(tensor) for head, tensor in logits_map.items() if torch.is_tensor(tensor)}
        else:
            probs = inputs
        if full_cfg is not None:
            gated = apply_hand_gate_from_config(probs, inputs, full_cfg, input_is_logits=bool(input_is_logits))
            if gated:
                probs = {**probs, **gated}
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


def build_decoder_from_logits(cfg: Mapping[str, Any]) -> Callable[[Mapping[str, torch.Tensor]], Mapping[str, torch.Tensor]]:
    """Decode logits into event masks with optional key-prior rescoring."""
    return build_decoder(cfg, input_is_logits=True)


__all__ = [
    "DECODER_DEFAULTS",
    "decode_hysteresis",
    "normalize_decoder_params",
    "resolve_decoder_from_config",
    "resolve_decoder_gates",
    "build_decoder",
    "build_decoder_from_logits",
]
