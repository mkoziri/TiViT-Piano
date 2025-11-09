"""Purpose:
    Provide shared temporal decoder utilities (parameter normalization, masking,
    and hysteresis decoding) used during training, calibration, and evaluation.

Key Functions/Classes:
    - DECODER_DEFAULTS: Canonical onset/offset decoder hyperparameters.
    - resolve_decoder_from_config(): Merge legacy + structured config overrides
      into normalized per-head decoder settings.
    - resolve_decoder_gates(): Convert normalized settings into concrete
      open/hold thresholds using fallbacks when overrides are missing.
    - decode_hysteresis(): Apply hysteresis with duration smoothing to
      pianoroll probabilities; helpers like pool_roll_BT() and
      build_threshold_mask() assist downstream scripts.

CLI:
    Not a standalone CLI module; import from scripts requiring decoder logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from .postproc import build_event_set
from .postproc import dp_duration, snap

DECODER_DEFAULTS = {
    "onset": {
        "open": 0.36,
        "hold": 0.28,
        "min_on": 2,
        "min_off": 2,
        "merge_gap": 1,
        "median": 3,
    },
    "offset": {
        "open": 0.32,
        "hold": 0.24,
        "min_on": 2,
        "min_off": 2,
        "merge_gap": 1,
        "median": 3,
    },
}


def apply_decoder_values(
    target: dict[str, dict[str, Any]],
    heads: Iterable[str],
    source: Mapping[str, Any] | None,
) -> None:
    if not isinstance(source, Mapping):
        return
    for key, value in source.items():
        if value is None:
            continue
        key_str = str(key)
        norm_key = "merge_gap" if key_str == "gap_merge" else key_str
        if norm_key not in {
            "open",
            "hold",
            "low_ratio",
            "min_on",
            "min_off",
            "merge_gap",
            "median",
        }:
            continue
        for head in heads:
            target.setdefault(head, {})[norm_key] = value


def normalize_decoder_params(
    raw: Mapping[str, dict[str, Any]],
    *,
    fallback_open: Optional[Mapping[str, float]] = None,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for head in ("onset", "offset"):
        defaults = DECODER_DEFAULTS[head]
        source = raw.get(head, {}) if isinstance(raw, Mapping) else {}
        entry: dict[str, Any] = {}

        open_defined = "open" in source and source.get("open") is not None
        hold_defined = "hold" in source and source.get("hold") is not None

        open_candidate = source.get("open")
        if open_candidate is None and fallback_open:
            open_candidate = fallback_open.get(head)
        if open_candidate is None:
            open_candidate = defaults["open"]
        try:
            open_val = float(open_candidate)
        except (TypeError, ValueError):
            open_val = defaults["open"]
        if not math.isfinite(open_val):
            open_val = defaults["open"]
        open_val = max(0.0, min(open_val, 1.0))

        ratio_candidate = source.get("low_ratio")
        ratio_val: Optional[float] = None
        if ratio_candidate is not None:
            try:
                ratio_val = float(ratio_candidate)
            except (TypeError, ValueError):
                ratio_val = None
            else:
                if not math.isfinite(ratio_val):
                    ratio_val = None
                elif ratio_val < 0.0:
                    ratio_val = 0.0

        hold_candidate = source.get("hold")
        if hold_candidate is None and ratio_val is not None and open_val > 0.0:
            hold_candidate = ratio_val * open_val
        if hold_candidate is None:
            hold_candidate = defaults["hold"]
        try:
            hold_val = float(hold_candidate)
        except (TypeError, ValueError):
            hold_val = defaults["hold"]
        if not math.isfinite(hold_val):
            hold_val = defaults["hold"]
        if hold_val < 0.0:
            hold_val = 0.0
        if hold_val > open_val and open_val > 0.0:
            hold_val = open_val

        for key in ("min_on", "min_off", "merge_gap"):
            default_val = defaults[key]
            src_val = source.get(key, default_val)
            try:
                int_val = int(src_val)
            except (TypeError, ValueError):
                int_val = default_val
            if int_val < 0:
                int_val = 0
            entry[key] = int_val

        median_default = defaults["median"]
        median_candidate = source.get("median", median_default)
        try:
            median_val = int(median_candidate)
        except (TypeError, ValueError):
            median_val = median_default
        if median_val < 1:
            median_val = 1
        if median_val % 2 == 0:
            median_val += 1

        if ratio_val is None:
            if open_val > 0.0:
                ratio_val = hold_val / open_val if open_val > 0.0 else 0.0
            else:
                ratio_val = 0.0

        entry.update(
            {
                "open": open_val,
                "hold": hold_val,
                "low_ratio": max(0.0, ratio_val or 0.0),
                "median": median_val,
                "open_defined": bool(open_defined),
                "hold_defined": bool(hold_defined),
            }
        )
        normalized[head] = entry
    return normalized


def resolve_decoder_from_config(
    metrics_cfg: Mapping[str, Any],
    *,
    fallback_open: Optional[Mapping[str, float]] = None,
) -> dict[str, dict[str, Any]]:
    collected: dict[str, dict[str, Any]] = {"onset": {}, "offset": {}}

    if not isinstance(metrics_cfg, Mapping):
        metrics_cfg = {}

    legacy_global: dict[str, Any] = {}
    legacy_map = {
        "open": metrics_cfg.get("decoder_open"),
        "hold": metrics_cfg.get("decoder_hold"),
        "low_ratio": metrics_cfg.get("decoder_low_ratio"),
        "min_on": metrics_cfg.get("decoder_min_on"),
        "min_off": metrics_cfg.get("decoder_min_off"),
        "merge_gap": metrics_cfg.get("decoder_merge_gap"),
        "gap_merge": metrics_cfg.get("decoder_gap_merge"),
        "median": metrics_cfg.get("decoder_median"),
    }
    for key, value in legacy_map.items():
        if value is not None:
            legacy_global[key] = value
    if legacy_global:
        apply_decoder_values(collected, ("onset", "offset"), legacy_global)

    for section_key in ("temporal_decoder", "proxy_decoder"):
        section = metrics_cfg.get(section_key)
        if not isinstance(section, Mapping):
            continue
        apply_decoder_values(collected, ("onset", "offset"), section)
        shared = section.get("shared")
        apply_decoder_values(collected, ("onset", "offset"), shared)
        for head in ("onset", "offset"):
            head_cfg = section.get(head)
            apply_decoder_values(collected, (head,), head_cfg)

    decoder_cfg = metrics_cfg.get("decoder")
    if isinstance(decoder_cfg, Mapping):
        apply_decoder_values(collected, ("onset", "offset"), decoder_cfg)
        shared = decoder_cfg.get("shared")
        apply_decoder_values(collected, ("onset", "offset"), shared)
        for head in ("onset", "offset"):
            head_cfg = decoder_cfg.get(head)
            apply_decoder_values(collected, (head,), head_cfg)

    return normalize_decoder_params(collected, fallback_open=fallback_open)


def resolve_decoder_gates(
    entry: Mapping[str, Any],
    *,
    fallback_open: float,
    default_hold: float,
) -> Tuple[float, float]:
    fallback = float(fallback_open)
    if not math.isfinite(fallback):
        fallback = 0.5
    elif fallback < 0.0:
        fallback = 0.0
    elif fallback > 1.0:
        fallback = 1.0

    open_defined = bool(entry.get("open_defined"))
    open_candidate = entry.get("open", fallback)
    try:
        open_val = float(open_candidate)
    except (TypeError, ValueError):
        open_val = fallback
    if not math.isfinite(open_val):
        open_val = fallback
    if not open_defined:
        open_val = fallback
    open_val = max(0.0, min(open_val, 1.0))

    hold_defined = bool(entry.get("hold_defined"))
    hold_candidate = entry.get("hold", default_hold)
    ratio_candidate = entry.get("low_ratio")
    if not hold_defined:
        ratio_val: Optional[float] = None
        if ratio_candidate is not None:
            try:
                ratio_val = float(ratio_candidate)
            except (TypeError, ValueError):
                ratio_val = None
            else:
                if not math.isfinite(ratio_val):
                    ratio_val = None
                elif ratio_val < 0.0:
                    ratio_val = 0.0
        if ratio_val is not None and open_val > 0.0:
            hold_candidate = ratio_val * open_val
        else:
            hold_candidate = default_hold
    try:
        hold_val = float(hold_candidate)
    except (TypeError, ValueError):
        hold_val = float(default_hold)
    if not math.isfinite(hold_val):
        hold_val = float(default_hold)
    if hold_val < 0.0:
        hold_val = 0.0
    if hold_val > open_val:
        hold_val = open_val

    return open_val, hold_val


def format_decoder_settings(decoder_kind: str, decoder_params: Mapping[str, Any]) -> str:
    if decoder_kind != "hysteresis":
        return f"decoder={decoder_kind}"
    onset = decoder_params.get("onset", {})
    offset = decoder_params.get("offset", {})
    return (
        "decoder=hysteresis "
        f"onset_open={onset.get('open', 0.0):.4f} "
        f"onset_hold={onset.get('hold', 0.0):.4f} "
        f"onset_min_on={int(onset.get('min_on', 0))} "
        f"onset_merge_gap={int(onset.get('merge_gap', 0))} "
        f"offset_open={offset.get('open', 0.0):.4f} "
        f"offset_hold={offset.get('hold', 0.0):.4f} "
        f"offset_min_off={int(offset.get('min_off', 0))} "
        f"offset_merge_gap={int(offset.get('merge_gap', 0))}"
    )


def decoder_notice_text(decoder_kind: str, decoder_params: Mapping[str, Any]) -> str:
    if decoder_kind != "hysteresis":
        return f"{decoder_kind} decoder active"
    onset = decoder_params.get("onset", {})
    offset = decoder_params.get("offset", {})
    return (
        "hysteresis "
        f"onset(open={onset.get('open', 0.0):.2f} "
        f"hold={onset.get('hold', 0.0):.2f} "
        f"min_on={int(onset.get('min_on', 0))} "
        f"merge_gap={int(onset.get('merge_gap', 0))} "
        f"median={int(onset.get('median', 1))}) "
        f"offset(open={offset.get('open', 0.0):.2f} "
        f"hold={offset.get('hold', 0.0):.2f} "
        f"min_off={int(offset.get('min_off', 0))} "
        f"merge_gap={int(offset.get('merge_gap', 0))} "
        f"median={int(offset.get('median', 1))})"
    )


def topk_mask(values: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        return torch.zeros_like(values, dtype=torch.bool)
    dim = values.dim()
    if dim < 1:
        raise ValueError(f"Expected tensor with at least 1 dim for top-k mask, got {dim}")
    last = values.shape[-1]
    count_eff = min(max(int(count), 0), last)
    if count_eff <= 0:
        return torch.zeros_like(values, dtype=torch.bool)
    if count_eff >= last:
        return torch.ones_like(values, dtype=torch.bool)
    topk_idx = values.topk(count_eff, dim=-1).indices
    mask = torch.zeros_like(values, dtype=torch.bool)
    return mask.scatter(-1, topk_idx, True)


def build_threshold_mask(
    values: torch.Tensor,
    threshold: float,
    *,
    mode: str,
    cap_count: int,
    top_k: int,
) -> torch.Tensor:
    mask = values >= float(threshold)
    if mode == "top_k_cap" and top_k > 0:
        mask = mask & topk_mask(values, top_k)
    if cap_count > 0:
        mask = mask & topk_mask(values, cap_count)
    return mask


def pool_roll_BT(x_btP: torch.Tensor, Tprime: int) -> torch.Tensor:
    """Downsample a (B,T,P) pianoroll along time using max pooling."""

    x = x_btP.permute(0, 2, 1)  # (B,P,T)
    x = F.adaptive_max_pool1d(x, Tprime)  # (B,P,T')
    return x.permute(0, 2, 1).contiguous()  # (B,T',P)


def median_filter_time(clip_probs: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply a 1D median filter along the time axis of a (T,P) pianoroll."""

    if kernel_size <= 1:
        return clip_probs
    if clip_probs.ndim != 2:
        raise ValueError(f"median filter expects 2D tensor, got {clip_probs.ndim}D")
    pad = kernel_size // 2
    probs_PT = clip_probs.transpose(0, 1)  # (P,T)
    padded = F.pad(probs_PT.unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)
    windows = padded.unfold(-1, kernel_size, 1)  # (P,T,kernel)
    filtered = windows.median(dim=-1).values  # (P,T)
    return filtered.transpose(0, 1).contiguous()


def decode_hysteresis(
    probs: torch.Tensor,
    open_thr: float,
    hold_thr: float,
    min_on: int,
    min_off: int,
    merge_gap: int,
    median: int,
) -> torch.Tensor:
    """Decode per-key probabilities with hysteresis and duration heuristics."""

    if probs.ndim not in (2, 3):
        raise ValueError(f"Expected probs with 2 or 3 dims, got {probs.ndim}")
    high_thr = float(open_thr)
    low_thr = float(hold_thr)
    if not math.isfinite(high_thr):
        high_thr = 0.5
    if not math.isfinite(low_thr):
        low_thr = high_thr
    if low_thr > high_thr:
        low_thr = high_thr
    if high_thr < 0.0:
        high_thr = 0.0
    if low_thr < 0.0:
        low_thr = 0.0
    min_on = max(0, int(min_on))
    min_off = max(0, int(min_off))
    merge_gap = max(0, int(merge_gap))
    if probs.numel() == 0:
        return torch.zeros_like(probs, dtype=torch.bool)

    def _decode_clip(clip_probs: torch.Tensor) -> torch.Tensor:
        processed = median_filter_time(clip_probs, median) if median > 1 else clip_probs
        T, P = processed.shape
        mask = torch.zeros((T, P), dtype=torch.bool, device=clip_probs.device)
        for pitch in range(P):
            seq = processed[:, pitch]
            vals = seq.tolist()
            segments = []
            state = False
            start_idx = 0
            for t, raw_val in enumerate(vals):
                val = float(raw_val) if math.isfinite(raw_val) else 0.0
                if not state:
                    if val >= high_thr:
                        state = True
                        start_idx = t
                else:
                    if val < low_thr:
                        segments.append([start_idx, t])
                        state = False
            if state:
                segments.append([start_idx, T])
            if not segments:
                continue
            merged = []
            for seg_start, seg_end in segments:
                if not merged:
                    merged.append([seg_start, seg_end])
                    continue
                prev_start, prev_end = merged[-1]
                gap = seg_start - prev_end
                should_merge = False
                if merge_gap >= 0 and gap <= merge_gap:
                    should_merge = True
                if min_off > 0 and gap < min_off:
                    should_merge = True
                if should_merge:
                    merged[-1][1] = seg_end
                else:
                    merged.append([seg_start, seg_end])
            for seg_start, seg_end in merged:
                if seg_end - seg_start < min_on:
                    continue
                mask[seg_start:seg_end, pitch] = True
        return mask

    if probs.ndim == 2:
        return _decode_clip(probs)

    batches = [_decode_clip(clip_probs) for clip_probs in probs]
    if not batches:
        return torch.zeros_like(probs, dtype=torch.bool)
    return torch.stack(batches, dim=0)


def apply_postprocessing(
    mask: torch.Tensor,
    probs: torch.Tensor,
    *,
    fps: float,
    cfg: Mapping[str, Any] | None = None,
    require_stats: bool = False,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Run optional snap/DP post-processing on a decoded mask."""

    cfg = cfg or {}
    decoder_cfg = cfg.get("decoder", {}) if isinstance(cfg, Mapping) else {}
    post_cfg = decoder_cfg.get("post", {}) if isinstance(decoder_cfg, Mapping) else {}
    snap_cfg = post_cfg.get("snap", {}) if isinstance(post_cfg, Mapping) else {}
    dp_cfg = post_cfg.get("dp", {}) if isinstance(post_cfg, Mapping) else {}
    need_snap = bool(snap_cfg.get("enabled", False))
    need_dp = bool(dp_cfg.get("enabled", False))
    need_bundle = need_snap or need_dp or require_stats
    if not need_bundle:
        return mask, {}

    events = build_event_set(mask, probs, fps)
    summary: Dict[str, Any] = {}
    if need_snap or require_stats:
        events = snap.apply(events, snap_cfg)
        snap_stats = events.stats.get("snap")
        if snap_stats:
            summary["snap"] = snap_stats
    if need_dp or require_stats:
        events = dp_duration.apply(events, dp_cfg, snap_cfg)
        dp_stats = events.stats.get("dp")
        if dp_stats:
            summary["dp"] = dp_stats

    updated_mask = events.mask.to(device=mask.device)
    return updated_mask, summary


__all__ = [
    "DECODER_DEFAULTS",
    "apply_decoder_values",
    "normalize_decoder_params",
    "resolve_decoder_from_config",
    "resolve_decoder_gates",
    "format_decoder_settings",
    "decoder_notice_text",
    "topk_mask",
    "build_threshold_mask",
    "pool_roll_BT",
    "median_filter_time",
    "decode_hysteresis",
    "apply_postprocessing",
]
