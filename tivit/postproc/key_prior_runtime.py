"""Key-prior runtime helpers for decoding.

Purpose:
    - Resolve key-prior settings from the full config plus dataset metadata.
    - Apply the key-aware prior to logits when decoder.post.key_prior.enabled is true.
    - Provide a small, safe wrapper for post-processing code paths.

Key Functions/Classes:
    - resolve_key_prior_runtime(): Parse config and return settings + runtime FPS/MIDI range.
    - apply_key_prior_from_config(): Apply the key prior and return updated logits.

CLI Arguments:
    (none)

Usage:
    adjusted = apply_key_prior_from_config({"onset": onset_logits}, cfg)
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from theory.key_prior_runtime import (
    KeyPriorRuntimeSettings,
    apply_key_prior_to_logits,
    resolve_key_prior_settings,
)


def _resolve_decode_fps(dataset_cfg: Mapping[str, Any]) -> float:
    decode_fps = float(dataset_cfg.get("decode_fps", 0.0) or 0.0)
    hop_seconds = float(dataset_cfg.get("hop_seconds", 0.0) or 0.0)
    if decode_fps <= 0.0 and hop_seconds > 0.0:
        decode_fps = 1.0 / hop_seconds
    if decode_fps <= 0.0:
        decode_fps = 30.0
    return decode_fps


def _resolve_midi_range(dataset_cfg: Mapping[str, Any], key_cfg: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    midi_low = None
    midi_high = None
    frame_cfg = dataset_cfg.get("frame_targets")
    if isinstance(frame_cfg, Mapping):
        note_min = frame_cfg.get("note_min")
        note_max = frame_cfg.get("note_max")
        if isinstance(note_min, (int, float)):
            midi_low = int(note_min)
        if isinstance(note_max, (int, float)):
            midi_high = int(note_max)
    key_midi_low = key_cfg.get("midi_low")
    if midi_low is None and isinstance(key_midi_low, (int, float)):
        midi_low = int(key_midi_low)
    key_midi_high = key_cfg.get("midi_high")
    if midi_high is None and isinstance(key_midi_high, (int, float)):
        midi_high = int(key_midi_high)
    return midi_low, midi_high


def resolve_key_prior_runtime(
    cfg: Mapping[str, Any] | None,
) -> Optional[Tuple[KeyPriorRuntimeSettings, float, Optional[int], Optional[int]]]:
    """Return key-prior settings plus runtime FPS/MIDI range, or None if disabled."""
    if not isinstance(cfg, Mapping):
        return None
    decoder_cfg = cfg.get("decoder", {})
    decoder_post = decoder_cfg.get("post", {}) if isinstance(decoder_cfg, Mapping) else {}
    decoder_key = decoder_post.get("key_prior", {}) if isinstance(decoder_post, Mapping) else {}
    decoder_enabled = bool(decoder_key.get("enabled", False))

    priors_cfg = cfg.get("priors", {}) if isinstance(cfg, Mapping) else {}
    key_cfg = priors_cfg.get("key_signature", {}) if isinstance(priors_cfg, Mapping) else {}
    if not isinstance(key_cfg, Mapping):
        key_cfg = {}

    enabled = decoder_enabled
    raw = dict(key_cfg)
    raw["enabled"] = bool(enabled)
    settings = resolve_key_prior_settings(raw)
    if not settings.enabled:
        return None

    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    decode_fps = _resolve_decode_fps(dataset_cfg)
    midi_low, midi_high = _resolve_midi_range(dataset_cfg, key_cfg)
    return settings, decode_fps, midi_low, midi_high


def apply_key_prior_from_config(
    logits_by_head: Mapping[str, Any],
    cfg: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Apply key prior if enabled; return a dict of updated logits."""
    runtime = resolve_key_prior_runtime(cfg)
    if runtime is None:
        return {}
    settings, fps, midi_low, midi_high = runtime
    try:
        return apply_key_prior_to_logits(
            logits_by_head,
            settings,
            fps=fps,
            midi_low=midi_low,
            midi_high=midi_high,
        )
    except Exception:
        return {}


__all__ = ["resolve_key_prior_runtime", "apply_key_prior_from_config"]
