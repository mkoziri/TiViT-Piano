"""Hand-gating helpers for decode-time priors.

Purpose:
    - Apply a hand-based prior to per-key probabilities at decode time.
    - Use the hand head's left/right probabilities to downweight keys outside
      the predicted hand region.
Key Functions/Classes:
    - HandGateSettings: Resolved decode-time settings container.
    - resolve_hand_gate_settings: Parse config into runtime settings.
    - apply_hand_gate_from_config: Apply hand gating to decoded head probabilities.
CLI Arguments:
    (none)
Usage:
    gated = apply_hand_gate_from_config(probs, inputs, cfg, input_is_logits=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HandGateSettings:
    enabled: bool
    apply_to: tuple[str, ...]
    strength: float
    mode: str
    split_midi: int
    note_min: int


def _as_tuple(value: object, *, fallback: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).lower() for item in value if item is not None)
    if isinstance(value, str):
        return (value.lower(),)
    return tuple(fallback)


def _resolve_clef_thresholds(frame_cfg: Mapping[str, Any], settings_cfg: Mapping[str, Any]) -> tuple[int, int]:
    raw = settings_cfg.get("clef_thresholds", None)
    if raw is None:
        raw = frame_cfg.get("clef_thresholds", (60, 64))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return int(raw[0]), int(raw[1])
        except (TypeError, ValueError):
            return 60, 64
    return 60, 64


def resolve_hand_gate_settings(cfg: Mapping[str, Any]) -> Optional[HandGateSettings]:
    """Parse decode-time hand gating settings from a full config."""
    if not isinstance(cfg, Mapping):
        return None
    decoder_cfg = cfg.get("decoder", {})
    if not isinstance(decoder_cfg, Mapping):
        return None
    post_cfg = decoder_cfg.get("post", {})
    if not isinstance(post_cfg, Mapping):
        return None
    gate_cfg = post_cfg.get("hand_gate", {})
    if not isinstance(gate_cfg, Mapping):
        return None
    if not bool(gate_cfg.get("enabled", False)):
        return None

    priors_cfg = cfg.get("priors", {})
    if not isinstance(priors_cfg, Mapping):
        priors_cfg = {}
    hand_priors = priors_cfg.get("hand_gating", {})
    if not isinstance(hand_priors, Mapping):
        hand_priors = {}

    apply_to = _as_tuple(hand_priors.get("apply_to"), fallback=("pitch", "onset", "offset"))
    if not apply_to:
        return None

    strength = hand_priors.get("decode_strength", None)
    if strength is None:
        strength = hand_priors.get("strength", 1.0)
    try:
        strength_val = float(strength)
    except (TypeError, ValueError):
        strength_val = 1.0

    mode = hand_priors.get("decode_mode", "multiply")
    mode = str(mode or "multiply").lower()

    dataset_cfg = cfg.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    frame_cfg = dataset_cfg.get("frame_targets", {})
    if not isinstance(frame_cfg, Mapping):
        frame_cfg = {}
    note_min = hand_priors.get("note_min", frame_cfg.get("note_min", 21))
    try:
        note_min_val = int(note_min)
    except (TypeError, ValueError):
        note_min_val = 21

    clef_left, _clef_right = _resolve_clef_thresholds(frame_cfg, hand_priors)
    split_midi = clef_left

    return HandGateSettings(
        enabled=True,
        apply_to=apply_to,
        strength=strength_val,
        mode=mode,
        split_midi=int(split_midi),
        note_min=note_min_val,
    )


def _extract_hand_tensor(inputs: Mapping[str, torch.Tensor]) -> Optional[torch.Tensor]:
    for key in ("hand", "hand_logits", "hand_probs"):
        value = inputs.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _align_hand_probs(hand_probs: torch.Tensor, roll: torch.Tensor) -> Optional[torch.Tensor]:
    if hand_probs.ndim == 1:
        if hand_probs.numel() != 2:
            return None
        hand_probs = hand_probs.view(1, 1, 2)
    elif hand_probs.ndim == 2:
        if hand_probs.shape[-1] != 2:
            return None
        if roll.ndim == 2:
            if hand_probs.shape[0] == 1 and roll.shape[0] != 1:
                hand_probs = hand_probs.expand(roll.shape[0], 2)
        else:
            if hand_probs.shape[0] == roll.shape[0]:
                hand_probs = hand_probs.unsqueeze(1)
            else:
                hand_probs = hand_probs.unsqueeze(0)
    elif hand_probs.ndim == 3:
        if hand_probs.shape[-1] != 2:
            return None
        if roll.ndim == 2:
            if hand_probs.shape[0] != 1:
                return None
            hand_probs = hand_probs.squeeze(0)
    else:
        return None

    if roll.ndim == 2 and hand_probs.ndim != 2:
        return None
    if roll.ndim == 3 and hand_probs.ndim != 3:
        return None

    if roll.ndim == 2:
        if hand_probs.shape[0] != roll.shape[0]:
            if hand_probs.shape[0] == 1:
                hand_probs = hand_probs.expand(roll.shape[0], 2)
            else:
                return None
        return hand_probs

    if hand_probs.shape[0] not in (1, roll.shape[0]):
        return None
    if hand_probs.shape[0] == 1 and roll.shape[0] > 1:
        hand_probs = hand_probs.expand(roll.shape[0], -1, -1)
    if hand_probs.shape[1] not in (1, roll.shape[1]):
        return None
    if hand_probs.shape[1] == 1 and roll.shape[1] > 1:
        hand_probs = hand_probs.expand(hand_probs.shape[0], roll.shape[1], 2)
    return hand_probs


def _build_gate_weights(
    hand_probs: torch.Tensor,
    roll: torch.Tensor,
    settings: HandGateSettings,
) -> Optional[torch.Tensor]:
    aligned = _align_hand_probs(hand_probs, roll)
    if aligned is None:
        return None

    strength = settings.strength
    base = 1.0 - strength
    left_weight = base + strength * aligned[..., 0].clamp(0.0, 1.0)
    right_weight = base + strength * aligned[..., 1].clamp(0.0, 1.0)

    if settings.mode not in {"multiply", "scale", "scale_prob"}:
        return None

    P = int(roll.shape[-1])
    split_idx = max(0, min(P, int(settings.split_midi - settings.note_min)))
    if split_idx <= 0 or split_idx >= P:
        return torch.ones_like(roll)

    key_ids = torch.arange(P, device=roll.device)
    left_mask = key_ids < split_idx
    weights = torch.ones_like(roll)
    if roll.ndim == 2:
        weights[:, left_mask] = left_weight.unsqueeze(-1)
        weights[:, ~left_mask] = right_weight.unsqueeze(-1)
    else:
        weights[:, :, left_mask] = left_weight.unsqueeze(-1)
        weights[:, :, ~left_mask] = right_weight.unsqueeze(-1)
    return weights


def apply_hand_gate_from_config(
    probs: Mapping[str, torch.Tensor],
    inputs: Mapping[str, torch.Tensor],
    cfg: Mapping[str, Any],
    *,
    input_is_logits: bool,
) -> Mapping[str, torch.Tensor]:
    """Apply hand gating to per-key probabilities; returns updated head tensors."""
    settings = resolve_hand_gate_settings(cfg)
    if settings is None:
        return {}

    hand_tensor = _extract_hand_tensor(inputs)
    if hand_tensor is None:
        return {}

    if hand_tensor.shape[-1] != 2:
        return {}

    if input_is_logits:
        hand_probs = F.softmax(hand_tensor.float(), dim=-1)
    else:
        hand_probs = hand_tensor.float()

    updated: dict[str, torch.Tensor] = {}
    for head in settings.apply_to:
        roll = probs.get(head)
        if not torch.is_tensor(roll):
            continue
        weights = _build_gate_weights(hand_probs, roll, settings)
        if weights is None:
            continue
        gated = (roll * weights).clamp(0.0, 1.0)
        updated[head] = gated
    return updated


__all__ = ["HandGateSettings", "resolve_hand_gate_settings", "apply_hand_gate_from_config"]
