#!/usr/bin/env python3
"""Purpose:
    Sweep onset/offset thresholds and evaluate frame- and event-level metrics,
    optionally dumping logits for further analysis.

Key Functions/Classes:
    - _parse_list(): Custom parser that supports comma- or space-separated CLI
      threshold lists.
    - _event_f1(): Computes event-level F1 scores using tolerance-aware
      matching on the time grid.
    - main(): Parses CLI options, loads a checkpoint, iterates over the
      dataloader, and prints metric summaries for each threshold.

CLI:
    Run ``python scripts/eval_thresholds.py --ckpt <path>`` with optional
    ``--thresholds``/``--prob_thresholds`` lists, ``--split`` to choose a
    dataset split, and ``--dump_logits`` to save logits to NPZ. Determinism
    controls are available via ``--seed`` and ``--deterministic``.
    Decoder hysteresis overrides (for example ``--decoder-onset-open``,
    ``--decoder-onset-min-on``) let callers evaluate per-head gate sweeps
    while keeping offset settings fixed.
"""
# NOTE: This is the frozen pre-refactor implementation used when callers pass
# --legacy-eval-thresholds for quick rollback.

import sys, json, time, math, os, torch, logging
from collections import Counter
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, List, Tuple

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config, align_pitch_dim, configure_verbosity
from utils.logging_utils import QUIET_INFO_FLAG
from utils.identifiers import canonical_video_id
from utils.time_grid import frame_to_sec
from data import make_dataloader
from models import build_model
from torch.utils.data import DataLoader, Subset
from utils.determinism import configure_determinism, resolve_deterministic_flag, resolve_seed

LOGGER = logging.getLogger("eval_thresholds")
QUIET_EXTRA = {QUIET_INFO_FLAG: True}


# Default probability grid used when sweeping thresholds without an explicit
# list.  We parse lists manually so callers can provide comma-separated values
# without escaping leading minus signs.
DEFAULT_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

_DECODER_DEFAULTS = {
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


def _apply_decoder_values(
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


def _normalize_decoder_params(
    raw: Mapping[str, dict[str, Any]],
    *,
    fallback_open: Optional[Mapping[str, float]] = None,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for head in ("onset", "offset"):
        defaults = _DECODER_DEFAULTS[head]
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


def _resolve_decoder_from_config(metrics_cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
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
        _apply_decoder_values(collected, ("onset", "offset"), legacy_global)

    for section_key in ("temporal_decoder", "proxy_decoder"):
        section = metrics_cfg.get(section_key)
        if not isinstance(section, Mapping):
            continue
        _apply_decoder_values(collected, ("onset", "offset"), section)
        shared = section.get("shared")
        _apply_decoder_values(collected, ("onset", "offset"), shared)
        for head in ("onset", "offset"):
            head_cfg = section.get(head)
            _apply_decoder_values(collected, (head,), head_cfg)

    decoder_cfg = metrics_cfg.get("decoder")
    if isinstance(decoder_cfg, Mapping):
        _apply_decoder_values(collected, ("onset", "offset"), decoder_cfg)
        shared = decoder_cfg.get("shared")
        _apply_decoder_values(collected, ("onset", "offset"), shared)
        for head in ("onset", "offset"):
            head_cfg = decoder_cfg.get(head)
            _apply_decoder_values(collected, (head,), head_cfg)

    return _normalize_decoder_params(collected)


def _resolve_decoder_thresholds(
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


def _format_decoder_settings(decoder_kind: str, decoder_params: dict) -> str:
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


def _decoder_notice_text(decoder_kind: str, decoder_params: dict) -> str:
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


def _topk_mask(values: torch.Tensor, count: int) -> torch.Tensor:
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


def _build_threshold_mask(
    values: torch.Tensor,
    threshold: float,
    *,
    mode: str,
    cap_count: int,
    top_k: int,
) -> torch.Tensor:
    mask = values >= float(threshold)
    if mode == "top_k_cap" and top_k > 0:
        mask = mask & _topk_mask(values, top_k)
    if cap_count > 0:
        mask = mask & _topk_mask(values, cap_count)
    return mask


def _prepare_logits_for_dump(tensor: torch.Tensor) -> np.ndarray:
    """Flatten a tensor to (T,P) and return a contiguous float64 numpy array."""

    if tensor is None:
        raise ValueError("Expected tensor, got None")
    if tensor.ndim < 2:
        raise ValueError(f"Logits tensor must have at least 2 dims, got {tensor.ndim}")

    tensor = tensor.contiguous()
    last_dim = tensor.shape[-1]
    tensor = tensor.reshape(-1, last_dim).contiguous()

    if tensor.ndim != 2:
        raise ValueError(f"Logits tensor reshape result must be 2D, got {tensor.ndim}D")
    if not tensor.is_contiguous():
        raise ValueError("Expected contiguous tensor after reshape")

    array = np.ascontiguousarray(tensor.numpy(), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"NumPy logits array must be 2D, got {array.ndim}D")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected contiguous NumPy array for logits dump")
    return array


def _parse_list(argv, name):
    """Extract ``--<name>`` from ``argv`` allowing comma/space separation."""
    flag = f"--{name}"
    for i, arg in enumerate(list(argv)):
        if arg.startswith(flag):
            if arg == flag:
                j = i + 1
                vals = []
                while j < len(argv) and not argv[j].startswith("--"):
                    vals.append(argv[j])
                    j += 1
                if not vals:
                    raise ValueError(f"{flag} expects at least one value")
                del argv[i:j]
                arg_str = " ".join(vals)
            else:  # handle --flag=... form
                arg_str = arg.split("=", 1)[1]
                del argv[i]
            arg_str = arg_str.replace(",", " ")
            return [float(v) for v in arg_str.split() if v]
    return None


def _resolve_threshold_lists(onset_vals, offset_vals):
    """Return onset/offset lists, cloning inputs and reusing values when missing."""

    reuse_flags = {"onset_from_offset": False, "offset_from_onset": False}
    if onset_vals is None and offset_vals is None:
        return None, None, reuse_flags

    if onset_vals is None:
        onset_vals = list(offset_vals)
        reuse_flags["onset_from_offset"] = True
    else:
        onset_vals = list(onset_vals)

    if offset_vals is None:
        offset_vals = list(onset_vals)
        reuse_flags["offset_from_onset"] = True
    else:
        offset_vals = list(offset_vals)

    return onset_vals, offset_vals, reuse_flags


def _binary_f1(pred, target, eps=1e-8):
    """Binary F1 score for tensors in {0,1}.

    Returns None if both pred and target are all zeros."""
    if target.sum().item() == 0 and pred.sum().item() == 0:
        return None
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)
   
   
def _event_f1(pred, target, hop_seconds: float, tol_sec: float, eps=1e-8):
    """Event-level F1 score with time tolerance."""
    pred_pos = pred.nonzero(as_tuple=False)
    true_pos = target.nonzero(as_tuple=False)
    if pred_pos.numel() == 0 and true_pos.numel() == 0:
        return None

    # ``frame_to_sec`` returns ``TensorLike`` which can be ``int`` from Pylance's
    # perspective.  Explicitly convert to tensors so static analyzers know these
    # support indexing and broadcasting.
    pred_times = torch.as_tensor(frame_to_sec(pred_pos[:, 0], hop_seconds))
    true_times = torch.as_tensor(frame_to_sec(true_pos[:, 0], hop_seconds))
    pred_pitch = pred_pos[:, 1]
    true_pitch = true_pos[:, 1]

    used = torch.zeros(true_pos.shape[0], dtype=torch.bool)
    tp = 0
    for i in range(pred_pos.shape[0]):
        p = pred_pitch[i]
        t = pred_times[i]
        mask = (true_pitch == p) & (~used)
        if mask.any():
            cand_idx = torch.where(mask)[0]
            diffs = torch.abs(true_times[cand_idx] - t)
            min_diff, j = torch.min(diffs, dim=0)
            if min_diff.item() <= tol_sec:
                tp += 1
                used[cand_idx[j]] = True
    fp = pred_pos.shape[0] - tp
    fn = true_pos.shape[0] - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def _pool_roll_BT(x_btP: torch.Tensor, Tprime: int) -> torch.Tensor:
    """Downsample a (B,T,P) pianoroll along time using max pooling.

    This mirrors the alignment logic used during training so that frame-level
    targets match the model's temporal resolution ``Tprime``.
    """
    x = x_btP.permute(0, 2, 1)  # (B,P,T)
    x = F.adaptive_max_pool1d(x, Tprime)  # (B,P,T')
    return x.permute(0, 2, 1).contiguous()  # (B,T',P)


def _median_filter_time(clip_probs: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply a 1D median filter along the time axis of a (T,P) pianoroll."""

    if kernel_size <= 1:
        return clip_probs
    if clip_probs.ndim != 2:
        raise ValueError(f"median filter expects 2D tensor, got {clip_probs.ndim}D")
    pad = kernel_size // 2
    # Operate in (P,T) space so padding replicates edge values per key.
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
    """Decode per-key probabilities with hysteresis and simple duration rules."""

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
        if median > 1:
            processed = _median_filter_time(clip_probs, median)
        else:
            processed = clip_probs
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

    batches = []
    for clip_probs in probs:
        batches.append(_decode_clip(clip_probs))
    if not batches:
        return torch.zeros_like(probs, dtype=torch.bool)
    return torch.stack(batches, dim=0)
    
def main():
    import argparse

    argv = sys.argv[1:]
    t_main_start = time.time()
    try:
        logit_thrs = _parse_list(argv, "thresholds")
        prob_thrs = _parse_list(argv, "prob_thresholds")
        offset_logit_thrs = _parse_list(argv, "offset_thresholds")
        offset_prob_thrs = _parse_list(argv, "offset_prob_thresholds")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
    ap.add_argument("--thresholds", metavar="T", nargs="*", help="Logit threshold values")
    ap.add_argument(
        "--offset_thresholds",
        metavar="T",
        nargs="*",
        help="Logit threshold values for the offset head (default: reuse onset thresholds)",
    )
    ap.add_argument(
        "--prob_thresholds",
        metavar="P",
        nargs="*",
        help="Probability threshold values",
    )
    ap.add_argument(
        "--offset_prob_thresholds",
        metavar="P",
        nargs="*",
        help="Probability threshold values for the offset head (default: reuse onset thresholds)",
    )
    ap.add_argument("--calibration", help="JSON file with calibrated thresholds")
    ap.add_argument("--head", choices=["onset", "offset"], help="Sweep thresholds for only one head")
    # Explicit thresholds for the non-swept head when no calibration is provided
    ap.add_argument("--fixed_offset_prob", type=float)
    ap.add_argument("--fixed_offset_logit", type=float)
    ap.add_argument("--fixed_onset_prob", type=float)
    ap.add_argument("--fixed_onset_logit", type=float)
    # Optional temperature and bias parameters for logit calibration
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Scale logits by this temperature before sigmoid; >1 softens predictions",
    )
    ap.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Additive bias applied to logits before sigmoid",
    )
    ap.add_argument("--split", choices=["train", "val", "test"], help="Dataset split to evaluate")
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--only", help="Restrict evaluation to a single canonical video id")
    ap.add_argument(
        "--verbose",
        choices=["quiet", "info", "debug"],
        help="Logging verbosity (default: quiet or $TIVIT_VERBOSE)",
    )
    ap.add_argument("--no-avlag", action="store_true", help="Disable audio/video lag estimation for isolation")
    ap.add_argument(
        "--dump_logits",
        default="",
        help="Optional path to save per-frame logits as a compressed NPZ",
    )
    ap.add_argument(
        "--grid_prob_thresholds",
        action="store_true",
        help="Evaluate the Cartesian product of onset/offset probability thresholds",
    )
    ap.add_argument(
        "--sweep_k_onset",
        action="store_true",
        help="When aggregation mode is k_of_p, sweep k_onset over {1,2,3}",
    )
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable periodic progress logging",
    )
    ap.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Minimum number of seconds between progress prints",
    )
    ap.add_argument(
        "--log-file",
        type=str,
        help="Optional file path to tee progress logs",
    )
    ap.add_argument(
        "--decoder",
        choices=["auto", "none", "hysteresis"],
        default="auto",
        help="Temporal decoder applied during evaluation (default: auto → hysteresis using config)",
    )
    ap.add_argument(
        "--low_ratio",
        type=float,
        default=None,
        help="Multiplier to derive the low hysteresis threshold (default: config or 0.6)",
    )
    ap.add_argument(
        "--min_on",
        type=int,
        default=None,
        help="Drop predicted on-segments shorter than this many frames (default: config or 2)",
    )
    ap.add_argument(
        "--min_off",
        type=int,
        default=None,
        help="Merge gaps shorter than this many frames between ons (default: config or 2)",
    )
    ap.add_argument(
        "--gap_merge",
        type=int,
        default=None,
        help="Merge on-segments separated by gaps <= this many frames (default: config or 1)",
    )
    ap.add_argument(
        "--decoder-onset-open",
        type=float,
        help="Override onset decoder open gate (probability, default: config)",
    )
    ap.add_argument(
        "--decoder-onset-hold",
        type=float,
        help="Override onset decoder hold gate (probability, default: config or derived from low_ratio)",
    )
    ap.add_argument(
        "--decoder-onset-min-on",
        type=int,
        help="Override onset decoder minimum on length in frames (default: config)",
    )
    ap.add_argument(
        "--decoder-onset-merge-gap",
        type=int,
        help="Override onset decoder merge gap in frames (default: config)",
    )
    ap.add_argument(
        "--decoder-offset-open",
        type=float,
        help="Override offset decoder open gate (probability, default: config)",
    )
    ap.add_argument(
        "--decoder-offset-hold",
        type=float,
        help="Override offset decoder hold gate (probability, default: config or derived from low_ratio)",
    )
    ap.add_argument(
        "--decoder-offset-min-off",
        type=int,
        help="Override offset decoder minimum off length in frames (default: config)",
    )
    ap.add_argument(
        "--decoder-offset-merge-gap",
        type=int,
        help="Override offset decoder merge gap in frames (default: config)",
    )
    ap.add_argument(
        "--median",
        type=int,
        default=None,
        help="Odd window size for optional time-axis median smoothing (default: config or 3)",
    )
    ap.add_argument("--seed", type=int, help="Seed for RNGs and dataloader shuffling")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle deterministic torch backends (default: config or enabled)",
    )
    args = ap.parse_args(argv)
    args.verbose = configure_verbosity(args.verbose)
    debug_mode = args.verbose == "debug"
    args.thresholds = logit_thrs
    args.prob_thresholds = prob_thrs
    args.offset_thresholds = offset_logit_thrs
    args.offset_prob_thresholds = offset_prob_thrs

    if args.thresholds is not None and args.prob_thresholds is not None:
        print("error: --thresholds and --prob_thresholds are mutually exclusive", file=sys.stderr)
        return
    if args.offset_thresholds is not None and args.offset_prob_thresholds is not None:
        print(
            "error: --offset_thresholds and --offset_prob_thresholds are mutually exclusive",
            file=sys.stderr,
        )
        return

    onset_logit_list, offset_logit_list, logit_reuse = _resolve_threshold_lists(
        args.thresholds, args.offset_thresholds
    )
    onset_prob_list, offset_prob_list, prob_reuse = _resolve_threshold_lists(
        args.prob_thresholds, args.offset_prob_thresholds
    )

    args.thresholds = onset_logit_list
    args.offset_thresholds = offset_logit_list
    args.prob_thresholds = onset_prob_list
    args.offset_prob_thresholds = offset_prob_list

    if args.head is None:
        has_logit_lists = args.thresholds is not None or args.offset_thresholds is not None
        has_prob_lists = args.prob_thresholds is not None or args.offset_prob_thresholds is not None
        if has_logit_lists and has_prob_lists:
            print(
                "error: specify only logit or probability threshold lists when sweeping both heads",
                file=sys.stderr,
            )
            return

    if args.thresholds is not None and args.offset_thresholds is not None:
        if len(args.thresholds) != len(args.offset_thresholds):
            print(
                "error: --thresholds and --offset_thresholds must contain the same number of values",
                file=sys.stderr,
            )
            return

    if args.prob_thresholds is not None and args.offset_prob_thresholds is not None:
        if not args.grid_prob_thresholds and len(args.prob_thresholds) != len(args.offset_prob_thresholds):
            print(
                "error: probability lists must match lengths unless --grid_prob_thresholds is enabled",
                file=sys.stderr,
            )
            return

    if args.low_ratio is not None and args.low_ratio < 0.0:
        print("error: --low_ratio must be non-negative", file=sys.stderr)
        return
    if args.min_on is not None and args.min_on < 0:
        print("error: --min_on must be >= 0", file=sys.stderr)
        return
    if args.min_off is not None and args.min_off < 0:
        print("error: --min_off must be >= 0", file=sys.stderr)
        return
    if args.gap_merge is not None and args.gap_merge < 0:
        print("error: --gap_merge must be >= 0", file=sys.stderr)
        return
    if args.median is not None and (args.median < 1 or args.median % 2 == 0):
        print("error: --median must be an odd integer >= 1", file=sys.stderr)
        return
    decoder_prob_fields = [
        ("--decoder-onset-open", args.decoder_onset_open),
        ("--decoder-onset-hold", args.decoder_onset_hold),
        ("--decoder-offset-open", args.decoder_offset_open),
        ("--decoder-offset-hold", args.decoder_offset_hold),
    ]
    for flag, value in decoder_prob_fields:
        if value is None:
            continue
        if not (0.0 <= value <= 1.0):
            print(f"error: {flag} must be within [0, 1]", file=sys.stderr)
            return
    decoder_int_fields = [
        ("--decoder-onset-min-on", args.decoder_onset_min_on),
        ("--decoder-onset-merge-gap", args.decoder_onset_merge_gap),
        ("--decoder-offset-min-off", args.decoder_offset_min_off),
        ("--decoder-offset-merge-gap", args.decoder_offset_merge_gap),
    ]
    for flag, value in decoder_int_fields:
        if value is None:
            continue
        if value < 0:
            print(f"error: {flag} must be >= 0", file=sys.stderr)
            return

    onset_probs_final = list(args.prob_thresholds) if args.prob_thresholds is not None else []
    offset_probs_final = list(args.offset_prob_thresholds) if args.offset_prob_thresholds is not None else []
    if args.grid_prob_thresholds:
        combos = len(onset_probs_final) * len(offset_probs_final)
    else:
        combos = max(len(onset_probs_final), len(offset_probs_final))
    onset_display = "[" + ",".join(f"{p:.4f}" for p in onset_probs_final) + "]"
    offset_display = "[" + ",".join(f"{p:.4f}" for p in offset_probs_final) + "]"
    LOGGER.info(
        "[grid] onset_probs=%s offset_probs=%s (final) combos=%d",
        onset_display,
        offset_display,
        combos,
        extra=QUIET_EXTRA,
    )

    log_handle = None

    if args.head is None and args.prob_thresholds is not None and prob_reuse.get("offset_from_onset"):
        print("[eval] offset probability thresholds not provided; reusing onset list", flush=True)
    if args.head is None and args.thresholds is not None and logit_reuse.get("offset_from_onset"):
        print("[eval] offset logit thresholds not provided; reusing onset list", flush=True)
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        import atexit

        atexit.register(log_handle.close)

    def _log_progress(msg: str, *, force: bool = False) -> None:
        should_print = force or bool(args.progress)
        if should_print:
            print(msg, flush=True)
        if log_handle is not None:
            log_handle.write(msg + "\n")
            log_handle.flush()

    def _format_seconds(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes, secs = divmod(int(seconds), 60)
        return f"{minutes:02d}:{secs:02d}"

    def _dataset_video_count(ds) -> str:
        if ds is None:
            return "?"
        try:
            if hasattr(ds, "samples"):
                return str(len(getattr(ds, "samples")))
            if hasattr(ds, "videos"):
                videos_attr = getattr(ds, "videos")
                try:
                    return str(len(videos_attr))
                except TypeError:
                    pass
            return str(len(ds))
        except Exception:
            return "?"
    
    stage_durations = {}
    BAD_CLIP_RETRY_LIMIT = 3
    bad_clip_counts = Counter()
    skip_paths = set()
    lag_ms_samples = []
    lag_source_counter = Counter()
    skipped_batches = 0

    def _extract_lag_values(value):
        vals = []
        if value is None:
            return vals
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1).tolist()
            for item in flat:
                try:
                    fval = float(item)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fval):
                    vals.append(fval)
            return vals
        if isinstance(value, (list, tuple)):
            for item in value:
                vals.extend(_extract_lag_values(item))
            return vals
        try:
            fval = float(value)
        except (TypeError, ValueError):
            return vals
        if math.isfinite(fval):
            vals.append(fval)
        return vals

    def _extract_lag_sources(value):
        sources = []
        if value is None:
            return sources
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item:
                    sources.append(item)
        elif isinstance(value, str) and value:
            sources.append(value)
        return sources

    def _filter_batch(batch, keep_indices):
        if not keep_indices:
            return None
        paths_field = batch.get("path")
        total = len(paths_field) if isinstance(paths_field, list) else None
        filtered = {}
        for key, value in batch.items():
            if key == "path" and isinstance(paths_field, list):
                filtered[key] = [paths_field[i] for i in keep_indices]
                continue
            if total is not None:
                if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == total:
                    idx_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=value.device)
                    filtered[key] = value.index_select(0, idx_tensor)
                    continue
                if isinstance(value, list) and len(value) == total:
                    filtered[key] = [value[i] for i in keep_indices]
                    continue
                if isinstance(value, tuple) and len(value) == total:
                    filtered[key] = [value[i] for i in keep_indices]
                    continue
            filtered[key] = value
        return filtered

    def _handle_bad_batch(paths, exc):
        nonlocal skipped_batches
        skipped_batches += 1
        safe_paths = [str(p) for p in (paths or []) if p]
        if safe_paths:
            first = Path(safe_paths[0]).name
            extra = len(safe_paths) - 1
            clip_desc = f"{first}+{extra} more" if extra > 0 else first
        else:
            clip_desc = "<unknown>"
        err_type = type(exc).__name__
        _log_progress(
            f"[warn] batch failed ({err_type}): clip={clip_desc} error={exc}",
            force=True,
        )
        for path in set(safe_paths):
            bad_clip_counts[path] += 1
            if bad_clip_counts[path] >= BAD_CLIP_RETRY_LIMIT and path not in skip_paths:
                skip_paths.add(path)
                _log_progress(
                    f"[progress] marked clip as bad after {BAD_CLIP_RETRY_LIMIT} failures: {Path(path).name}",
                    force=True,
                )
    # Unless a calibration file is provided and no head is specified, default to
    # sweeping over probability thresholds when none were specified explicitly.
    if args.thresholds is None and args.prob_thresholds is None:
        if args.head is not None or not args.calibration:
            args.prob_thresholds = DEFAULT_THRESHOLDS.copy()


    cfg = dict(load_config("configs/config.yaml"))
    seed = resolve_seed(args.seed, cfg)
    deterministic = resolve_deterministic_flag(args.deterministic, cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["deterministic"] = deterministic
    configure_determinism(seed, deterministic)
    print(
        f"[determinism] seed={seed} deterministic={'on' if deterministic else 'off'}",
        flush=True,
    )
    dataset_raw = cfg.get("dataset")
    if isinstance(dataset_raw, dict):
        dataset_cfg = dict(dataset_raw)
    else:
        dataset_cfg = {}
    cfg["dataset"] = dataset_cfg
    if args.max_clips is not None:
        dataset_cfg["max_clips"] = args.max_clips
    if args.frames is not None:
        dataset_cfg["frames"] = args.frames
    only_id = canonical_video_id(args.only) if args.only else None
    if only_id:
        dataset_cfg["only_video"] = only_id
    env_disable = str(os.environ.get("AVSYNC_DISABLE", "")).strip().lower()
    avlag_disabled = bool(args.no_avlag) or env_disable in {"1", "true", "yes", "on"}
    if avlag_disabled:
        dataset_cfg["avlag_disabled"] = True
    if debug_mode:
        dataset_cfg["num_workers"] = 0
        dataset_cfg["persistent_workers"] = False
        dataset_cfg["pin_memory"] = False
        print("[debug] num_workers=0, persistent_workers=False, pin_memory=False", flush=True)
    decode_fps = float(dataset_cfg.get("decode_fps", 1.0))
    hop_seconds = float(dataset_cfg.get("hop_seconds", 1.0 / decode_fps))
    event_tolerance = float(
        dataset_cfg.get("frame_targets", {}).get("tolerance", hop_seconds)
    )
    split = args.split or dataset_cfg.get("split_val") or dataset_cfg.get("split") or "val"

    frames_display = dataset_cfg.get("frames")
    max_clips_display = dataset_cfg.get("max_clips")
    only_display = only_id or "-"
    frame_text = frames_display if frames_display is not None else "?"
    max_clips_text = max_clips_display if max_clips_display is not None else "?"
    print(
        f"[progress] starting (split={split}, frames={frame_text}, max_clips={max_clips_text}, only={only_display})",
        flush=True,
    )

    metrics_cfg = cfg.get("training", {}).get("metrics", {}) or {}
    agg_cfg = metrics_cfg.get("aggregation", {}) or {}
    agg_mode = str(agg_cfg.get("mode", "any")).lower()
    agg_top_k = int(agg_cfg.get("top_k", 0) or 0)
    agg_tau_sum = float(agg_cfg.get("tau_sum", 0.0) or 0.0)
    agg_k_cfg = agg_cfg.get("k", {}) or {}
    default_k_onset = int(agg_k_cfg.get("onset", 1) or 1)
    default_k_offset = int(agg_k_cfg.get("offset", 1) or 1)
    sweep_cfg = metrics_cfg.get("sweep", {}) or {}
    floor_band_raw = sweep_cfg.get("floor_band", [0.20, 0.30, 0.40])
    floor_band: List[float] = []
    if isinstance(floor_band_raw, (list, tuple)):
        for item in floor_band_raw:
            try:
                val = float(item)
            except (TypeError, ValueError):
                continue
            if 0.0 <= val <= 1.0:
                floor_band.append(val)
    else:
        try:
            val = float(floor_band_raw)
        except (TypeError, ValueError):
            val = None
        if val is not None and 0.0 <= val <= 1.0:
            floor_band.append(val)
    if not floor_band:
        floor_band = [0.20, 0.30, 0.40]

    calibration_data = None
    if args.calibration:
        with open(args.calibration) as f:
            calibration_data = json.load(f)

    include_k_column = agg_mode == "k_of_p"
    if include_k_column and args.sweep_k_onset and args.head is None:
        k_candidates = sorted({default_k_onset, 1, 2, 3})
    else:
        k_candidates = [default_k_onset]

    # build loader
    t_dataset_build0 = time.time()
    val_loader = make_dataloader(cfg, split=split, seed=seed)
    if isinstance(val_loader, dict):
        val_loader = val_loader.get(split, next(iter(val_loader.values())))
    if isinstance(val_loader, (list, tuple)):
        val_loader = val_loader[0]

    dataset = getattr(val_loader, "dataset", None)
    dataset_name = dataset.__class__.__name__ if dataset is not None else type(val_loader).__name__
    ds_len: Optional[int] = None
    dataset_count = "?"
    ok_videos = 0
    materialize_duration = 0.0
    if dataset is not None:
        materialize_stats = getattr(dataset, "_eval_materialize_stats", {}) or {}
        if isinstance(materialize_stats, dict):
            try:
                ok_videos = int(materialize_stats.get("videos") or 0)
            except (TypeError, ValueError):
                ok_videos = 0
            try:
                materialize_duration = float(materialize_stats.get("duration") or 0.0)
            except (TypeError, ValueError):
                materialize_duration = 0.0
        if materialize_duration == 0.0:
            try:
                materialize_duration = float(getattr(dataset, "_last_materialize_duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                materialize_duration = 0.0
        try:
            ds_len = len(dataset)
            dataset_count = str(ds_len)
        except TypeError:
            ds_len = None
            dataset_count = "?"
    if ds_len == 0 and ok_videos > 0:
        print("[error] dataset len is 0 after audit ok>0 – eval entries were not materialized", flush=True)
        sys.exit(1)
    dataset_elapsed = time.time() - t_dataset_build0
    stage_durations["dataset_init"] = dataset_elapsed
    batch_size_val = getattr(val_loader, "batch_size", None)
    batch_display = str(batch_size_val) if batch_size_val is not None else "?"
    worker_count = getattr(val_loader, "num_workers", None)
    worker_display = str(worker_count) if worker_count is not None else "?"
    video_count_display = _dataset_video_count(dataset)
    print(
        f"[progress] dataset ready (videos={video_count_display}, workers={worker_display})",
        flush=True,
    )
    _log_progress(
        f"[progress] dataset ready in {_format_seconds(dataset_elapsed)} ({dataset_elapsed:.2f}s) "
        f"backend={dataset_name} len={dataset_count} batch={batch_display}",
        force=True,
    )
    frame_summary = getattr(dataset, "frame_target_summary", None)
    if frame_summary:
        frame_summary_display = frame_summary
        if avlag_disabled and "lag_source=" in frame_summary_display:
            prefix, suffix = frame_summary_display.split("lag_source=", 1)
            if "," in suffix:
                _, tail = suffix.split(",", 1)
                frame_summary_display = f"{prefix}lag_source=no_avlag,{tail}"
            else:
                frame_summary_display = f"{prefix}lag_source=no_avlag"
        _log_progress(f"[progress] {frame_summary_display}", force=True)

    if ds_len is not None:
        resolved_cap = args.max_clips if args.max_clips is not None else ds_len
        target_clips = int(min(ds_len, int(resolved_cap)))
    else:
        target_clips = int(args.max_clips) if args.max_clips is not None else None

    if dataset is not None and target_clips is not None:
        try:
            base_len = len(dataset)
        except TypeError:
            base_len = None
        if base_len is not None:
            subset_cap = min(base_len, int(target_clips))
            subset_indices = list(range(subset_cap))
            dataset = Subset(dataset, subset_indices)
            num_workers = getattr(val_loader, "num_workers", 0)
            persistent_workers = getattr(val_loader, "persistent_workers", False)
            if num_workers <= 0:
                persistent_workers = False
            loader_kwargs = {
                "batch_size": getattr(val_loader, "batch_size", 1),
                "shuffle": False,
                "num_workers": num_workers,
                "pin_memory": getattr(val_loader, "pin_memory", False),
                "drop_last": getattr(val_loader, "drop_last", False),
                "collate_fn": getattr(val_loader, "collate_fn", None),
                "persistent_workers": persistent_workers,
                "timeout": getattr(val_loader, "timeout", 0),
            }
            prefetch_factor = getattr(val_loader, "prefetch_factor", None)
            if num_workers > 0 and prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
            pin_memory_device = getattr(val_loader, "pin_memory_device", None)
            if pin_memory_device:
                loader_kwargs["pin_memory_device"] = pin_memory_device
            worker_init_fn = getattr(val_loader, "worker_init_fn", None)
            if worker_init_fn is not None:
                loader_kwargs["worker_init_fn"] = worker_init_fn
            generator = getattr(val_loader, "generator", None)
            if generator is not None:
                loader_kwargs["generator"] = generator
            multiprocessing_context = getattr(val_loader, "multiprocessing_context", None)
            if multiprocessing_context is not None:
                loader_kwargs["multiprocessing_context"] = multiprocessing_context
            val_loader = DataLoader(dataset, **loader_kwargs)
            dataset = getattr(val_loader, "dataset", dataset)
            ds_len = len(dataset)
            dataset_count = str(ds_len)
            target_clips = ds_len
            dataset_elapsed = time.time() - t_dataset_build0
            stage_durations["dataset_init"] = dataset_elapsed

    if materialize_duration > 0:
        stage_durations["materialize"] = materialize_duration

    if target_clips == 0:
        _log_progress("[progress] target_clips resolved to 0; exiting early.", force=True)
        return


    per_head_sweep_vals = None
    per_head_use_logits = False
    per_head_mode = "prob"
    if args.head is not None:
        if args.head == "onset":
            if args.thresholds is not None:
                per_head_sweep_vals = args.thresholds
                per_head_use_logits = True
                per_head_mode = "logit"
            else:
                per_head_sweep_vals = args.prob_thresholds
                per_head_use_logits = False
                per_head_mode = "prob"
        else:
            if args.offset_thresholds is not None:
                per_head_sweep_vals = args.offset_thresholds
                per_head_use_logits = True
                per_head_mode = "logit"
            else:
                per_head_sweep_vals = args.offset_prob_thresholds
                per_head_use_logits = False
                per_head_mode = "prob"

        if per_head_sweep_vals is None:
            per_head_sweep_vals = args.thresholds if args.thresholds is not None else args.prob_thresholds
            per_head_use_logits = args.thresholds is not None
            per_head_mode = "logit" if per_head_use_logits else "prob"

    calib_pairs: int = 0
    logit_pairs: int = 0
    prob_pairs: int = 0
    num_prob_combos: int = 0

    if args.head is None:
        calib_pairs = 0
        if calibration_data:
            on_cal = calibration_data.get("onset", {})
            off_cal = calibration_data.get("offset", {})
            if "best_logit" in on_cal and "best_logit" in off_cal:
                calib_pairs = 1
            elif "best_prob" in on_cal and "best_prob" in off_cal:
                calib_pairs = 1
        logit_pairs = len(args.thresholds) if args.thresholds else 0
        onset_prob_list = args.prob_thresholds or []
        offset_prob_list = args.offset_prob_thresholds or onset_prob_list
        if args.grid_prob_thresholds:
            prob_pairs = len(onset_prob_list) * len(offset_prob_list)
        else:
            prob_pairs = len(onset_prob_list)
        num_prob_combos = prob_pairs * len(k_candidates)
        num_thr_pairs = calib_pairs + logit_pairs + prob_pairs
        num_combos = calib_pairs + logit_pairs + num_prob_combos
        num_k = len(k_candidates) if prob_pairs > 0 and len(k_candidates) > 1 else 1
        thr_parts = []
        if logit_pairs:
            thr_parts.append(f"logit:{logit_pairs}")
        if prob_pairs:
            if args.grid_prob_thresholds and onset_prob_list and offset_prob_list:
                thr_parts.append(f"prob_grid:{len(onset_prob_list)}x{len(offset_prob_list)}")
            else:
                thr_parts.append(f"prob_pairs:{prob_pairs}")
        if calib_pairs:
            thr_parts.append(f"calib:{calib_pairs}")
        if not thr_parts:
            thr_parts.append("none")
        thr_desc = ",".join(thr_parts)
        k_sweep_state = "on" if len(k_candidates) > 1 else "off"
    else:
        sweep_len = len(per_head_sweep_vals) if per_head_sweep_vals is not None else 0
        num_thr_pairs = sweep_len
        num_combos = sweep_len
        num_k = 1
        thr_desc = str(sweep_len)
        k_sweep_state = "off"

    target_display = str(target_clips) if target_clips is not None else "?"
    _log_progress(
        f"[progress] starting: clips={target_display} combos={num_combos} (thr={thr_desc}, k_sweep={k_sweep_state})",
        force=True,
    )

    # load model + ckpt
    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    # run model once to collect logits/probabilities and targets
    onset_logits_list, offset_logits_list = [], []
    pitch_logits_list = []
    onset_probs, offset_probs = [], []
    onset_tgts, offset_tgts = [], []
    clips_done = 0
    t_data0 = time.time()
    last_clip_print = t_data0
    heartbeat_interval = max(10.0, float(args.progress_interval or 10.0))
    last_heartbeat = t_data0
    last_clip_name = "-"
    first_batch_time = None
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if target_clips is not None and clips_done >= target_clips:
                break
            raw_paths = batch.get("path")
            if isinstance(raw_paths, (list, tuple)):
                paths = [str(p) for p in raw_paths]
            elif raw_paths is None:
                paths = []
            else:
                paths = [str(raw_paths)]
            if skip_paths and paths:
                keep_indices = [idx for idx, p in enumerate(paths) if p not in skip_paths]
                if len(keep_indices) != len(paths):
                    filtered_batch = _filter_batch(batch, keep_indices)
                    if filtered_batch is None:
                        blocked = ", ".join(Path(p).name for p in paths if p) or "<unknown>"
                        _log_progress(
                            f"[progress] skipped batch (all blocked clips): {blocked}",
                            force=True,
                        )
                        continue
                    batch = filtered_batch
                    raw_paths = batch.get("path")
                    if isinstance(raw_paths, (list, tuple)):
                        paths = [str(p) for p in raw_paths]
                    elif raw_paths is None:
                        paths = []
                    else:
                        paths = [str(raw_paths)]
            try:
                x = batch["video"]
                if first_batch_time is None:
                    first_batch_time = time.time()
                    first_wait = first_batch_time - t_data0
                    stage_durations["first_batch"] = first_wait
                    _log_progress(
                        f"[progress] first batch ready in {_format_seconds(first_wait)} ({first_wait:.2f}s) – includes decode/A/V lag warmup",
                        force=True,
                    )
                out = model(x)

                # prefer *_logits if present; fallback to old naming
                onset_logits = out["onset_logits"] if "onset_logits" in out else out.get("onset")
                offset_logits = out["offset_logits"] if "offset_logits" in out else out.get("offset")
                pitch_logits = out.get("pitch_logits")
                
                # Apply temperature scaling and bias for calibration
                onset_logits = onset_logits / args.temperature + args.bias
                offset_logits = offset_logits / args.temperature + args.bias
                onset_prob = torch.sigmoid(onset_logits)
                offset_prob = torch.sigmoid(offset_logits)

                onset_logits_list.append(onset_logits.detach().cpu())
                offset_logits_list.append(offset_logits.detach().cpu())
                onset_probs.append(onset_prob.detach().cpu())
                offset_probs.append(offset_prob.detach().cpu())

                if pitch_logits is not None:
                    if pitch_logits.dim() == 2:
                        pitch_logits = pitch_logits.unsqueeze(1)
                    pitch_logits_list.append(pitch_logits.detach().cpu())
                
                onset_tgts.append(batch["onset_roll"].float().cpu())
                offset_tgts.append(batch["offset_roll"].float().cpu())

                lag_vals = _extract_lag_values(batch.get("lag_ms"))
                if lag_vals:
                    lag_ms_samples.extend(lag_vals)
                if not avlag_disabled:
                    lag_sources = _extract_lag_sources(batch.get("lag_source"))
                    if lag_sources:
                        lag_source_counter.update(lag_sources)

                if debug_mode and len(onset_logits_list) == 1:
                    print("[DEBUG] batch video", x.shape, "onset_logits", onset_logits.shape)
                    print(
                        "[DEBUG] onset_roll nonzero=",
                        int(batch["onset_roll"].sum().item()),
                        "offset_roll nonzero=",
                        int(batch["offset_roll"].sum().item()),
                    )
                
                batch_size = int(x.shape[0]) if hasattr(x, "shape") and x.shape else 1
                clips_done += batch_size
                now = time.time()
                if paths:
                    candidate_id = canonical_video_id(Path(paths[-1]).name)
                    last_clip_name = candidate_id or Path(paths[-1]).name
                if now - last_heartbeat >= heartbeat_interval:
                    elapsed = now - t_data0
                    _log_progress(
                        f"[progress] data pass heartbeat: processed_clips={clips_done} skips={len(skip_paths)} last_clip={last_clip_name} elapsed={_format_seconds(elapsed)}",
                        force=True,
                    )
                    last_heartbeat = now
                if args.progress:
                    progress_force = i == 0 or (target_clips is not None and clips_done >= target_clips)
                    if progress_force or now - last_clip_print >= args.progress_interval:
                        elapsed = now - t_data0
                        if target_clips is not None and target_clips > 0:
                            pct = min(100.0, 100.0 * clips_done / float(target_clips))
                            pct_display = f"{pct:5.1f}"
                        else:
                            pct_display = "?"
                        if clips_done == 0:
                            eta_display = "--:--"
                        elif target_clips is None or target_clips <= 0:
                            eta_display = "--:--"
                        else:
                            remaining = max(target_clips - clips_done, 0)
                            if remaining == 0:
                                eta_display = "00:00"
                            else:
                                eta_seconds = (elapsed / clips_done) * remaining
                                eta_display = _format_seconds(eta_seconds)
                        processed_display = clips_done if target_clips is None else min(clips_done, target_clips)
                        clips_total_display = target_display
                        _log_progress(
                            f"[progress] clips {processed_display}/{clips_total_display}  ({pct_display}%)  elapsed={_format_seconds(elapsed)}  eta≈{eta_display}",
                            force=progress_force,
                        )
                        last_clip_print = now
                if target_clips is not None and clips_done >= target_clips:
                    break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                _handle_bad_batch(paths, exc)
                continue

    elapsed_data = time.time() - t_data0
    stage_durations["data_pass"] = elapsed_data
    throughput = clips_done / elapsed_data if elapsed_data > 0 else 0.0
    processed_display = clips_done if target_clips is None else min(clips_done, target_clips)
    skipped_display = len(skip_paths)
    elapsed_display = _format_seconds(elapsed_data)
    expected_display = target_clips if target_clips is not None else "?"
    _log_progress(
        f"[progress] data pass done: clips={processed_display}, expected={expected_display}, skipped={skipped_display}, elapsed={elapsed_display}",
        force=True,
    )
    _log_progress(
        f"[progress] throughput: {throughput:.2f} clips/s ({elapsed_data:.2f}s)",
        force=True,
    )
    if avlag_disabled:
        _log_progress("[progress] A/V lag ms stats: disabled (all zero).", force=True)
    elif lag_ms_samples:
        lag_arr = np.asarray(lag_ms_samples, dtype=np.float32)
        lag_mean = float(lag_arr.mean())
        lag_median = float(np.median(lag_arr))
        lag_p95 = float(np.percentile(lag_arr, 95))
        _log_progress(
            "[progress] A/V lag ms stats: mean={:.1f} median={:.1f} p95={:.1f} samples={}".format(
                lag_mean,
                lag_median,
                lag_p95,
                lag_arr.size,
            ),
            force=True,
        )
    if lag_source_counter and not avlag_disabled:
        top_sources = ", ".join(f"{src}:{cnt}" for src, cnt in lag_source_counter.most_common(3))
        _log_progress(f"[progress] lag sources top: {top_sources}", force=True)
    if skipped_batches:
        _log_progress(f"[progress] batches skipped due to errors: {skipped_batches}", force=True)
    if bad_clip_counts:
        summary_bits = ", ".join(f"{Path(p).name}:{count}" for p, count in bad_clip_counts.items())
        _log_progress(f"[progress] bad clip retries: {summary_bits}", force=True)
    if skip_paths:
        skip_names = ", ".join(Path(p).name for p in sorted(skip_paths))
        _log_progress(f"[progress] permanently skipped clips: {skip_names}", force=True)

    if not onset_logits_list:
        _log_progress("[progress] no valid clips processed; aborting.", force=True)
        print("error: no valid clips processed; aborting.", file=sys.stderr)
        return

    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    pitch_logits = torch.cat(pitch_logits_list, dim=0) if pitch_logits_list else None
    onset_probs = torch.cat(onset_probs, dim=0)
    offset_probs = torch.cat(offset_probs, dim=0)
    onset_tgts = torch.cat(onset_tgts, dim=0)
    offset_tgts = torch.cat(offset_tgts, dim=0)

    T_logits, P_logits = onset_probs.shape[1], onset_probs.shape[2]
    if onset_tgts.shape[1] != T_logits:
        onset_tgts = _pool_roll_BT(onset_tgts, T_logits)
        offset_tgts = _pool_roll_BT(offset_tgts, T_logits)
    onset_tgts = align_pitch_dim(onset_probs, onset_tgts, "onset")
    offset_tgts = align_pitch_dim(offset_probs, offset_tgts, "offset")
    
    if debug_mode:
        print(
            "[DEBUG] aligned shapes logits=",
            onset_logits.shape,
            "targets=",
            onset_tgts.shape,
        )
        print(
            "[DEBUG] targets nonzero onset=",
            int(onset_tgts.sum().item()),
            "offset=",
            int(offset_tgts.sum().item()),
        )
        diff = (torch.sigmoid(onset_logits) - onset_probs).abs().max().item()
        print(f"[DEBUG] sigmoid max abs diff={diff:.3e}")
    
    dump_path = Path(args.dump_logits).expanduser() if args.dump_logits else None
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)

        dump_arrays = {}
        onset_np = _prepare_logits_for_dump(onset_logits)
        offset_np = _prepare_logits_for_dump(offset_logits)
        dump_arrays["onset_logits"] = onset_np
        dump_arrays["offset_logits"] = offset_np

        if pitch_logits is not None:
            dump_arrays["pitch_logits"] = _prepare_logits_for_dump(pitch_logits)

        pitch_bins = next((arr.shape[1] for arr in dump_arrays.values() if arr is not None), None)
        frame_cfg = cfg.get("dataset", {}).get("frame_targets", {}) or {}
        midi_low = int(frame_cfg.get("note_min", 21))
        midi_high_cfg = frame_cfg.get("note_max")
        midi_high = int(midi_high_cfg) if midi_high_cfg is not None else midi_low
        if pitch_bins is not None:
            if midi_high - midi_low + 1 != pitch_bins:
                midi_low = 21
                midi_high = midi_low + pitch_bins - 1
            else:
                midi_high = midi_low + pitch_bins - 1

        meta = {
            "fps": decode_fps,
            "midi_low": midi_low,
            "midi_high": midi_high,
        }
        dump_arrays["meta"] = json.dumps(meta, sort_keys=True)

        np.savez_compressed(dump_path, **dump_arrays)
        print(f"[eval] dumped logits -> {dump_path}")
        
    # diagnostic prints
    print(f"[OVERALL onset probs] mean={onset_probs.mean():.3f} min={onset_probs.min():.3f} max={onset_probs.max():.3f}")
    print(f"[OVERALL offset probs] mean={offset_probs.mean():.3f} min={offset_probs.min():.3f} max={offset_probs.max():.3f}")

    def _percentile(flat: torch.Tensor, q: float) -> float:
        if flat.numel() == 0:
            return 0.0
        try:
            return float(torch.quantile(flat, q).item())
        except (RuntimeError, AttributeError):
            # Fallback for older torch – rely on numpy.
            return float(np.quantile(flat.numpy(), q))

    def _summarize_probs(name: str, tensor: torch.Tensor) -> Tuple[float, dict]:
        flat = tensor.reshape(-1).float()
        max_prob = float(flat.max().item()) if flat.numel() else 0.0
        stats = {
            0.95: _percentile(flat, 0.95),
            0.99: _percentile(flat, 0.99),
            0.995: _percentile(flat, 0.995),
        }
        _log_progress(
            "[sweep] %s prob stats: max=%.4f p95=%.4f p99=%.4f p99.5=%.4f"
            % (name, max_prob, stats[0.95], stats[0.99], stats[0.995]),
            force=True,
        )
        return max_prob, stats

    offset_lower_hint = 0.10  # base guardrail

    onset_max_prob, onset_stats = _summarize_probs("onset", onset_probs)
    offset_max_prob, offset_stats = _summarize_probs("offset", offset_probs)
    onset_peak = max(onset_max_prob, float(onset_stats.get(0.99, onset_max_prob)))
    offset_peak = max(offset_max_prob, float(offset_stats.get(0.99, offset_max_prob)))
    onset_lower_hint = float(max(0.0, min(onset_peak - 0.10, 0.95)))
    offset_lower_hint = float(max(0.0, min(offset_peak - 0.10, 0.95)))

    def _format_list(vals: List[float]) -> str:
        return "[" + ",".join(f"{v:.3f}" for v in vals) + "]"

    if args.head is None:
        if args.prob_thresholds:
            using_grid = bool(args.grid_prob_thresholds)
            onset_list = list(args.prob_thresholds)
            if args.offset_prob_thresholds is not None:
                offset_list = list(args.offset_prob_thresholds)
            else:
                offset_list = list(onset_list)

            lowest_onset_thr = min(onset_list) if onset_list else None
            lowest_offset_thr = min(offset_list) if offset_list else None
            onset_extend_needed = (
                lowest_onset_thr is not None
                and onset_peak + 1e-9 < lowest_onset_thr
            )
            offset_extend_needed = (
                lowest_offset_thr is not None
                and offset_peak + 1e-9 < lowest_offset_thr
            )
            inserted_lower_onset = None
            inserted_lower_offset = None

            if using_grid:
                onset_set = set(onset_list)
                offset_set = set(offset_list)
                onset_set.update(floor_band)
                offset_set.update(floor_band)
                if onset_extend_needed:
                    onset_set.add(onset_lower_hint)
                    inserted_lower_onset = onset_lower_hint
                if offset_extend_needed:
                    offset_set.add(offset_lower_hint)
                    inserted_lower_offset = offset_lower_hint
                args.prob_thresholds = sorted(onset_set)
                args.offset_prob_thresholds = sorted(offset_set)
            else:
                pairs = [(float(o), float(off)) for o, off in zip(onset_list, offset_list)]
                pair_map = {(round(o, 6), round(off, 6)): (o, off) for o, off in pairs}

                def _add_pair(on_val: float, off_val: float) -> None:
                    key = (round(on_val, 6), round(off_val, 6))
                    if key not in pair_map:
                        pair_map[key] = (float(on_val), float(off_val))

                for val in floor_band:
                    _add_pair(val, val)
                if onset_extend_needed:
                    _add_pair(onset_lower_hint, onset_lower_hint)
                    inserted_lower_onset = onset_lower_hint
                if offset_extend_needed:
                    _add_pair(offset_lower_hint, offset_lower_hint)
                    inserted_lower_offset = offset_lower_hint

                pairs = list(pair_map.values())
                pairs.sort(key=lambda item: (item[1], item[0]))
                args.prob_thresholds = [p[0] for p in pairs]
                args.offset_prob_thresholds = [p[1] for p in pairs]

            onset_list = list(args.prob_thresholds)
            offset_list = (
                list(args.offset_prob_thresholds)
                if args.offset_prob_thresholds is not None
                else list(onset_list)
            )
            if using_grid:
                prob_pairs = len(onset_list) * len(offset_list)
            else:
                prob_pairs = len(onset_list)
            num_prob_combos = prob_pairs * len(k_candidates)
            num_thr_pairs = calib_pairs + logit_pairs + prob_pairs
            num_combos = calib_pairs + logit_pairs + num_prob_combos

            if inserted_lower_onset is not None:
                _log_progress(
                    "[sweep] onset peak prob %.4f < min sweep %.4f → added %.4f to sweep list."
                    % (onset_peak, lowest_onset_thr, inserted_lower_onset),
                    force=True,
                )
            if inserted_lower_offset is not None:
                _log_progress(
                    "[sweep] offset peak prob %.4f < min sweep %.4f → added %.4f to sweep list."
                    % (offset_peak, lowest_offset_thr, inserted_lower_offset),
                    force=True,
                )
            _log_progress(
                f"[sweep] ensured floor band {', '.join(f'{v:.2f}' for v in sorted(floor_band))} in probability sweep.",
                force=True,
            )
            if using_grid:
                prob_desc = f"prob_grid:{len(onset_list)}x{len(offset_list)}"
            else:
                prob_desc = f"prob_pairs:{prob_pairs}"
            thr_desc = []
            if logit_pairs:
                thr_desc.append(f"logit:{logit_pairs}")
            thr_desc.append(prob_desc)
            if calib_pairs:
                thr_desc.append(f"calib:{calib_pairs}")
            _log_progress(
                f"[sweep] updated probability grids → onset={_format_list(onset_list)} "
                f"offset={_format_list(offset_list)} combos={num_combos}",
                force=True,
            )
    else:
        if per_head_mode == "prob" and per_head_sweep_vals is not None:
            values = set(float(v) for v in per_head_sweep_vals)
            values.update(floor_band)
            inserted_lower = None
            if args.head == "offset":
                lowest = min(values) if values else None
                extend_needed = lowest is not None and offset_peak + 1e-9 < lowest
                if extend_needed:
                    values.add(offset_lower_hint)
                    inserted_lower = offset_lower_hint
            elif args.head == "onset":
                lowest = min(values) if values else None
                extend_needed = lowest is not None and onset_peak + 1e-9 < lowest
                if extend_needed:
                    values.add(onset_lower_hint)
                    inserted_lower = onset_lower_hint
            per_head_sweep_vals = sorted(values)
            num_combos = len(per_head_sweep_vals)
            if inserted_lower is not None:
                peak_val = offset_peak if args.head == "offset" else onset_peak
                _log_progress(
                    "[sweep] per-head %s peak prob %.4f → added %.4f to sweep list."
                    % (args.head, peak_val, inserted_lower),
                    force=True,
                )
            _log_progress(
                f"[sweep] per-head sweep ({args.head}) values={_format_list(per_head_sweep_vals)} combos={num_combos}",
                force=True,
            )

    # Use all key/time positions rather than collapsing with ``any``.
    # Collapsing across the note dimension causes the predicted rate to be
    # either 0 or 1 for a clip, which in turn makes F1-threshold sweeps
    # uninformative.  Instead we compute metrics over the full pianoroll so
    # that the positive rate varies smoothly with the threshold.
    onset_true_bin = (onset_tgts > 0).float()
    offset_true_bin = (offset_tgts > 0).float()

    decoder_params = _resolve_decoder_from_config(metrics_cfg)
    if args.low_ratio is not None:
        decoder_params["onset"]["low_ratio"] = float(args.low_ratio)
        decoder_params["offset"]["low_ratio"] = float(args.low_ratio)
    if args.min_on is not None:
        decoder_params["onset"]["min_on"] = int(args.min_on)
        decoder_params["offset"]["min_on"] = int(args.min_on)
    if args.min_off is not None:
        decoder_params["onset"]["min_off"] = int(args.min_off)
        decoder_params["offset"]["min_off"] = int(args.min_off)
    if args.gap_merge is not None:
        decoder_params["onset"]["merge_gap"] = int(args.gap_merge)
        decoder_params["offset"]["merge_gap"] = int(args.gap_merge)
    if args.median is not None:
        decoder_params["onset"]["median"] = int(args.median)
        decoder_params["offset"]["median"] = int(args.median)
    if args.decoder_onset_open is not None:
        decoder_params["onset"]["open"] = float(args.decoder_onset_open)
        decoder_params["onset"]["open_defined"] = True
    if args.decoder_onset_hold is not None:
        decoder_params["onset"]["hold"] = float(args.decoder_onset_hold)
        decoder_params["onset"]["hold_defined"] = True
    if args.decoder_onset_min_on is not None:
        decoder_params["onset"]["min_on"] = int(args.decoder_onset_min_on)
    if args.decoder_onset_merge_gap is not None:
        decoder_params["onset"]["merge_gap"] = int(args.decoder_onset_merge_gap)
    if args.decoder_offset_open is not None:
        decoder_params["offset"]["open"] = float(args.decoder_offset_open)
        decoder_params["offset"]["open_defined"] = True
    if args.decoder_offset_hold is not None:
        decoder_params["offset"]["hold"] = float(args.decoder_offset_hold)
        decoder_params["offset"]["hold_defined"] = True
    if args.decoder_offset_min_off is not None:
        decoder_params["offset"]["min_off"] = int(args.decoder_offset_min_off)
    if args.decoder_offset_merge_gap is not None:
        decoder_params["offset"]["merge_gap"] = int(args.decoder_offset_merge_gap)
    decoder_params = _normalize_decoder_params(decoder_params)
    onset_decoder = decoder_params["onset"]
    offset_decoder = decoder_params["offset"]
    decoder_choice = args.decoder or "auto"
    if decoder_choice == "none":
        print("[decoder] requested decoder=none -> forcing hysteresis with config defaults", flush=True)
    decoder_kind = "hysteresis" if decoder_choice in {"auto", "none"} else decoder_choice
    decoder_notice_printed = False
    decoder_logits_warned = False
    decoder_settings_summary = _format_decoder_settings(decoder_kind, decoder_params)
    print(f"[decoder-settings] {decoder_settings_summary}")

    def _eval_pair(on_thr, off_thr, use_logits, *, k_onset=None):
        nonlocal decoder_notice_printed, decoder_logits_warned
        if k_onset is None:
            k_onset = default_k_onset
        k_onset = max(1, int(k_onset))
        k_offset = max(1, int(default_k_offset))
        onset_open_thr: Optional[float] = None
        onset_hold_thr: Optional[float] = None
        offset_open_thr: Optional[float] = None
        offset_hold_thr: Optional[float] = None
        base_onset_tensor = onset_logits if use_logits else onset_probs
        base_offset_tensor = offset_logits if use_logits else offset_probs
        base_onset = torch.as_tensor(base_onset_tensor) if not torch.is_tensor(base_onset_tensor) else base_onset_tensor
        base_offset = torch.as_tensor(base_offset_tensor) if not torch.is_tensor(base_offset_tensor) else base_offset_tensor

        onset_mask_bool = _build_threshold_mask(
            base_onset,
            on_thr,
            mode=agg_mode,
            cap_count=k_onset,
            top_k=agg_top_k,
        )
        offset_mask_bool = _build_threshold_mask(
            base_offset,
            off_thr,
            mode=agg_mode,
            cap_count=k_offset,
            top_k=agg_top_k,
        )

        onset_mask_float = onset_mask_bool.float()
        offset_mask_float = offset_mask_bool.float()
        onset_pred_bin = onset_mask_float
        offset_pred_bin = offset_mask_float

        if decoder_kind == "hysteresis" and not use_logits:
            if not decoder_notice_printed:
                print(f"[decoder] {_decoder_notice_text(decoder_kind, decoder_params)}", flush=True)
                decoder_notice_printed = True
            onset_open_thr, onset_hold_thr = _resolve_decoder_thresholds(
                onset_decoder,
                fallback_open=on_thr,
                default_hold=_DECODER_DEFAULTS["onset"]["hold"],
            )
            offset_open_thr, offset_hold_thr = _resolve_decoder_thresholds(
                offset_decoder,
                fallback_open=off_thr,
                default_hold=_DECODER_DEFAULTS["offset"]["hold"],
            )
            onset_probs_tensor = torch.as_tensor(onset_probs) if not torch.is_tensor(onset_probs) else onset_probs
            offset_probs_tensor = torch.as_tensor(offset_probs) if not torch.is_tensor(offset_probs) else offset_probs
            masked_onset_probs = (onset_probs_tensor * onset_mask_float).contiguous()
            masked_offset_probs = (offset_probs_tensor * offset_mask_float).contiguous()
            onset_pred_mask = decode_hysteresis(
                masked_onset_probs,
                onset_open_thr,
                onset_hold_thr,
                onset_decoder["min_on"],
                onset_decoder["min_off"],
                onset_decoder["merge_gap"],
                onset_decoder["median"],
            )
            offset_pred_mask = decode_hysteresis(
                masked_offset_probs,
                offset_open_thr,
                offset_hold_thr,
                offset_decoder["min_on"],
                offset_decoder["min_off"],
                offset_decoder["merge_gap"],
                offset_decoder["median"],
            )
            onset_pred_bin = onset_pred_mask.to(onset_probs_tensor.dtype)
            offset_pred_bin = offset_pred_mask.to(offset_probs_tensor.dtype)
        elif use_logits:
            if decoder_kind == "hysteresis" and not decoder_logits_warned:
                LOGGER.warning(
                    "Hysteresis decoder requires probability thresholds; falling back to simple logit thresholding.",
                    extra=QUIET_EXTRA,
                )
                decoder_logits_warned = True
            onset_pred_bin = onset_mask_bool.float()
            offset_pred_bin = offset_mask_bool.float()
        
        f1_on = _binary_f1(onset_pred_bin.reshape(-1), onset_true_bin.reshape(-1))
        f1_off = _binary_f1(offset_pred_bin.reshape(-1), offset_true_bin.reshape(-1))
        ev_f1_on = _event_f1(onset_pred_bin, onset_true_bin, hop_seconds, event_tolerance)
        ev_f1_off = _event_f1(offset_pred_bin, offset_true_bin, hop_seconds, event_tolerance)
        onset_pred_rate = onset_pred_bin.mean().item()
        onset_pos_rate = onset_true_bin.mean().item()

        f1_on = 0.0 if f1_on is None else f1_on
        f1_off = 0.0 if f1_off is None else f1_off
        ev_f1_on = 0.0 if ev_f1_on is None else ev_f1_on
        ev_f1_off = 0.0 if ev_f1_off is None else ev_f1_off

        return {
            "onset_thr": float(on_thr),
            "offset_thr": float(off_thr),
            "decoder_onset_open": float(onset_open_thr) if onset_open_thr is not None else None,
            "decoder_onset_hold": float(onset_hold_thr) if onset_hold_thr is not None else None,
            "decoder_offset_open": float(offset_open_thr) if offset_open_thr is not None else None,
            "decoder_offset_hold": float(offset_hold_thr) if offset_hold_thr is not None else None,
            "decoder_kind": decoder_kind,
            "f1_on": float(f1_on),
            "f1_off": float(f1_off),
            "onset_pred_rate": float(onset_pred_rate),
            "onset_pos_rate": float(onset_pos_rate),
            "ev_f1_on": float(ev_f1_on),
            "ev_f1_off": float(ev_f1_off),
            "k_onset": int(k_onset),
            "use_logits": bool(use_logits),
        }
    
    printed_header = False

    def _header():
        nonlocal printed_header
        if not printed_header:
            cols = [
                "onset_thr",
                "offset_thr",
            ]
            if include_k_column:
                cols.append("k_onset")
            cols.extend(
                [
                    "onset_f1",
                    "offset_f1",
                    "onset_pred_rate",
                    "onset_pos_rate",
                    "onset_event_f1",
                    "offset_event_f1",
                ]
            )
            print("\t".join(cols))
            printed_header = True
    
    def _print_row(res: dict):
        values = [f"{res['onset_thr']:.2f}", f"{res['offset_thr']:.2f}"]
        if include_k_column:
            values.append(str(res["k_onset"]))
        values.extend(
            [
                f"{res['f1_on']:0.3f}",
                f"{res['f1_off']:0.3f}",
                f"{res['onset_pred_rate']:0.3f}",
                f"{res['onset_pos_rate']:0.3f}",
                f"{res['ev_f1_on']:0.3f}",
                f"{res['ev_f1_off']:0.3f}",
            ]
        )
        print("\t".join(values))

    best_result = None
    total_evals = 0

    def _update_best(res: dict):
        nonlocal best_result
        nonlocal total_evals
        total_evals += 1
        ev_mean = 0.5 * (res["ev_f1_on"] + res["ev_f1_off"])
        if best_result is None:
            best_result = {**res, "ev_mean": ev_mean}
            return
        best_mean = best_result.get("ev_mean", -1.0)
        if ev_mean > best_mean + 1e-9:
            best_result = {**res, "ev_mean": ev_mean}
        elif abs(ev_mean - best_mean) <= 1e-9 and res["ev_f1_on"] > best_result["ev_f1_on"] + 1e-9:
            best_result = {**res, "ev_mean": ev_mean}
    
    combo_idx = 0
    t_grid0 = time.time()
    last_grid_print = t_grid0
    _log_progress(f"[progress] grid sweep start: combos={num_combos}", force=True)

    def _run_eval(on_thr, off_thr, use_logits, *, k_onset=None):
        nonlocal combo_idx
        nonlocal last_grid_print
        res = _eval_pair(on_thr, off_thr, use_logits, k_onset=k_onset)
        _print_row(res)
        _update_best(res)
        combo_idx += 1
        if args.progress and num_combos > 0:
            now = time.time()
            progress_force = combo_idx == 1 or combo_idx == num_combos
            if progress_force or now - last_grid_print >= args.progress_interval:
                elapsed = now - t_grid0
                if combo_idx > 0 and num_combos:
                    remaining = max(num_combos - combo_idx, 0)
                    eta_seconds = (elapsed / combo_idx) * remaining if combo_idx else 0.0
                    eta_display = _format_seconds(eta_seconds)
                else:
                    eta_display = "??:??"
                k_display = k_onset if k_onset is not None else default_k_onset
                _log_progress(
                    f"[progress] grid {combo_idx}/{num_combos}  onset_thr={on_thr:.3f}  offset_thr={off_thr:.3f}  k_onset={k_display}  elapsed={_format_seconds(elapsed)}  eta≈{eta_display}",
                    force=progress_force,
                )
                last_grid_print = now
        return res


    if args.head is None:
        # Evaluate at calibrated thresholds if provided.
        if calibration_data:
            on_cal = calibration_data.get("onset", {})
            off_cal = calibration_data.get("offset", {})
            if "best_logit" in on_cal and "best_logit" in off_cal:
                _header()
                _run_eval(
                    on_cal["best_logit"],
                    off_cal["best_logit"],
                    True,
                    k_onset=default_k_onset,
                )
            elif "best_prob" in on_cal and "best_prob" in off_cal:
                _header()
                _run_eval(
                    on_cal["best_prob"],
                    off_cal["best_prob"],
                    False,
                    k_onset=default_k_onset,
                )
            else:
                print("Calibration file missing best_logit/best_prob keys", file=sys.stderr)

        # Sweep over provided threshold grids.
        if args.thresholds:
            _header()
            offset_list = args.offset_thresholds if args.offset_thresholds else args.thresholds
            if len(offset_list) != len(args.thresholds):
                print("error: offset logit threshold count must match onset count", file=sys.stderr)
                return
            for on_thr, off_thr in zip(args.thresholds, offset_list):
                _run_eval(on_thr, off_thr, True, k_onset=default_k_onset)
        if args.prob_thresholds:
            _header()
            onset_list = args.prob_thresholds
            offset_list = args.offset_prob_thresholds if args.offset_prob_thresholds else onset_list
            for k_val in k_candidates:
                if args.grid_prob_thresholds:
                    for on_thr in onset_list:
                        for off_thr in offset_list:
                            _run_eval(on_thr, off_thr, False, k_onset=k_val)
                else:
                    if len(onset_list) != len(offset_list):
                        print("error: offset probability thresholds must match onset count", file=sys.stderr)
                        return
                    for idx, on_thr in enumerate(onset_list):
                        off_thr = offset_list[idx]
                        _run_eval(on_thr, off_thr, False, k_onset=k_val)
    else:
        # Per-head sweep
        sweep_vals = per_head_sweep_vals
        use_logits = per_head_use_logits
        mode = per_head_mode

        if sweep_vals is None:
            print(
                "error: specify --thresholds or --prob_thresholds for per-head sweep",
                file=sys.stderr,
            )
            return
        
        other_head = "offset" if args.head == "onset" else "onset"
        fixed_thr = None
        source = None

        if calibration_data:
            other_cal = calibration_data.get(other_head, {})
            if use_logits:
                if "best_logit" in other_cal:
                    fixed_thr = other_cal["best_logit"]
                elif "best_prob" in other_cal:
                    fixed_thr = torch.logit(torch.tensor(other_cal["best_prob"])).item()
            else:
                if "best_prob" in other_cal:
                    fixed_thr = other_cal["best_prob"]
                elif "best_logit" in other_cal:
                    fixed_thr = torch.sigmoid(torch.tensor(other_cal["best_logit"])).item()
            if fixed_thr is None:
                print("Calibration file missing threshold for", other_head, file=sys.stderr)
                return
            source = "calibration"
        else:
            if args.head == "onset":
                fixed_thr = args.fixed_offset_logit if use_logits else args.fixed_offset_prob
                flag_name = "--fixed_offset_logit" if use_logits else "--fixed_offset_prob"
            else:
                fixed_thr = args.fixed_onset_logit if use_logits else args.fixed_onset_prob
                flag_name = "--fixed_onset_logit" if use_logits else "--fixed_onset_prob"
            if fixed_thr is None:
                print(
                    f"error: specify {flag_name} or --calibration to fix {other_head} threshold",
                    file=sys.stderr,
                )
                return
            source = "flag"

        print(
            f"Per-head sweep: head={args.head}, mode={mode}, fixed_{other_head}={fixed_thr:.3f} (source={source})"
        )
        _header()
        for t in sweep_vals:
            if args.head == "onset":
                on_thr, off_thr = t, fixed_thr
            else:
                on_thr, off_thr = fixed_thr, t
            _run_eval(on_thr, off_thr, use_logits, k_onset=default_k_onset)

    grid_elapsed = time.time() - t_grid0
    stage_durations["grid_pass"] = grid_elapsed
    grid_rate = total_evals / grid_elapsed if grid_elapsed > 0 else 0.0
    _log_progress(
        f"[progress] grid pass done: combos={num_combos}, elapsed={_format_seconds(grid_elapsed)} ({grid_elapsed:.2f}s), rate={grid_rate:.2f}/s",
        force=True,
    )
    total_elapsed = time.time() - t_main_start
    stage_order = [
        ("dataset_init", "dataset"),
        ("materialize", "materialize"),
        ("first_batch", "first_batch"),
        ("data_pass", "data_pass"),
        ("grid_pass", "grid_sweep"),
    ]
    stage_parts = []
    for key, label in stage_order:
        if key in stage_durations:
            dur_val = stage_durations[key]
            stage_parts.append(f"{label}={_format_seconds(dur_val)}")
    stage_summary = ", ".join(stage_parts) if stage_parts else "n/a"
    _log_progress(
        f"[progress] stages: {stage_summary} | total={_format_seconds(total_elapsed)} ({total_elapsed:.2f}s)",
        force=True,
    )

    if best_result is not None and total_evals > 0:
        print(
            "[best-event] mean_event_f1={:.3f} onset_event_f1={:.3f} offset_event_f1={:.3f} k_onset={}".format(
                best_result["ev_mean"],
                best_result["ev_f1_on"],
                best_result["ev_f1_off"],
                best_result["k_onset"],
            )
        )
        if not best_result.get("use_logits", False):
            print(
                "[best-yaml] onset_prob_threshold={:.2f}, offset_prob_threshold={:.2f}, k_onset={}".format(
                    best_result["onset_thr"],
                    best_result["offset_thr"],
                    best_result["k_onset"],
                )
            )
        onset_open_best = best_result.get("decoder_onset_open")
        onset_hold_best = best_result.get("decoder_onset_hold")
        offset_open_best = best_result.get("decoder_offset_open")
        offset_hold_best = best_result.get("decoder_offset_hold")
        if (
            onset_open_best is not None
            and onset_hold_best is not None
            and offset_open_best is not None
            and offset_hold_best is not None
        ):
            print(
                "[best-decoder] onset_open={:.3f} hold={:.3f} min_on={} merge_gap={} | "
                "offset_open={:.3f} hold={:.3f} min_off={} merge_gap={}".format(
                    onset_open_best,
                    onset_hold_best,
                    onset_decoder["min_on"],
                    onset_decoder["merge_gap"],
                    offset_open_best,
                    offset_hold_best,
                    offset_decoder["min_off"],
                    offset_decoder["merge_gap"],
                )
            )

if __name__ == "__main__":
    main()
