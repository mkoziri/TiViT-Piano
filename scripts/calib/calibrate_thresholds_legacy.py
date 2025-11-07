#!/usr/bin/env python3
"""Calibrate onset/offset thresholds for TiViT-Piano checkpoints.

Purpose:
    Evaluate model predictions to derive calibrated logits/probabilities,
    compute reliability diagnostics, and persist partial results to
    ``calibration.json`` so long sweeps can resume safely.

Key Functions/Classes:
    - _collect: Run a model over a dataloader while checkpointing partial stats.
    - _compute_metrics: Sweep thresholds to compute F1, prediction rates, ECE,
      and Brier scores for onset/offset heads.
    - main: CLI entry point handling checkpoint loading, evaluation, and report
      emission.

CLI Arguments:
    --ckpt PATH (default: checkpoints/tivit_best.pt)
        Checkpoint file evaluated to produce calibration statistics.
    --split {train,val,test} (default: config split)
        Dataset split evaluated; falls back to dataset config when omitted.
    --max-clips INT (default: None)
        Override max clips evaluated; defaults to YAML configuration.
    --frames INT (default: None)
        Override frames per clip; defaults to YAML configuration.
    --timeout-mins FLOAT (default: None)
        Stop evaluation after the given minutes while keeping partial results.
    --no-avlag (default: False)
        Disable audio/video lag estimation before building frame targets.
    --seed INT (default: config or 1337)
        Seed forwarded to RNGs and dataloaders for reproducible sweeps.
    --deterministic / --no-deterministic
        Toggle deterministic PyTorch backend features (default: config).
    --verbose {quiet,info,debug} (default: env or quiet)
        Logging verbosity for the script and dependent modules.
    --debug (default: False)
        Run with single-process dataloading (num_workers=0) for quick iteration.

Usage:
    python scripts/calib/calibrate_thresholds.py --ckpt checkpoints/latest.pt --split val
"""
# NOTE: This file serves as the rollback target for --legacy-calibrate-thresholds.

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from filelock import FileLock, Timeout as FileLockTimeout

# -----------------------------------------------------------------------------
# Repo setup so we can import from src/
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from utils import load_config, align_pitch_dim, configure_verbosity
from utils.time_grid import frame_to_sec
from data import make_dataloader
from models import build_model
from utils.determinism import configure_determinism, resolve_deterministic_flag, resolve_seed


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

HEARTBEAT_SECONDS = 10.0

DEFAULT_LOGIT_GRID = torch.arange(-4.0, 2.0 + 1e-9, 0.05)
DEFAULT_PROB_GRID = torch.arange(0.01, 0.99 + 1e-9, 0.01)
_OFFSET_EXTRA = torch.arange(0.99, 1.001 + 1e-9, 0.002)
OFFSET_PROB_GRID = torch.unique(torch.cat([DEFAULT_PROB_GRID, _OFFSET_EXTRA]))
OFFSET_PROB_GRID = torch.sort(OFFSET_PROB_GRID).values.clamp(max=0.999)

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


def _median_filter_time(clip_probs: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return clip_probs
    if clip_probs.ndim != 2:
        raise ValueError(f"median filter expects 2D tensor, got {clip_probs.ndim}D")
    pad = kernel_size // 2
    probs_PT = clip_probs.transpose(0, 1)
    padded = F.pad(probs_PT.unsqueeze(1), (pad, pad), mode="replicate").squeeze(1)
    windows = padded.unfold(-1, kernel_size, 1)
    filtered = windows.median(dim=-1).values
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

    batches = [_decode_clip(clip) for clip in probs]
    if not batches:
        return torch.zeros_like(probs, dtype=torch.bool)
    return torch.stack(batches, dim=0)


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


def _event_f1(pred, target, hop_seconds: float, tol_sec: float, eps: float = 1e-8):
    pred_pos = pred.nonzero(as_tuple=False)
    true_pos = target.nonzero(as_tuple=False)
    if pred_pos.numel() == 0 and true_pos.numel() == 0:
        return None

    pred_times = torch.as_tensor(frame_to_sec(pred_pos[:, 0], hop_seconds))
    true_times = torch.as_tensor(frame_to_sec(true_pos[:, 0], hop_seconds))
    pred_pitch = pred_pos[:, 1]
    true_pitch = true_pos[:, 1]

    used = torch.zeros(true_pos.shape[0], dtype=torch.bool)
    tp = 0
    for i in range(pred_pos.shape[0]):
        pitch = pred_pitch[i]
        time_val = pred_times[i]
        mask = (true_pitch == pitch) & (~used)
        if mask.any():
            cand_idx = torch.where(mask)[0]
            diffs = torch.abs(true_times[cand_idx] - time_val)
            min_diff, j = torch.min(diffs, dim=0)
            if min_diff.item() <= tol_sec:
                tp += 1
                used[cand_idx[j]] = True
    fp = pred_pos.shape[0] - tp
    fn = true_pos.shape[0] - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def _print_progress(processed: int, total: Optional[int]) -> None:
    if total is not None and total > 0:
        pct = 100.0 * processed / total
        msg = f"[calib] processed {processed}/{total} clips ({pct:5.1f}%)"
    else:
        msg = f"[calib] processed {processed} clips"
    print(msg, flush=True)


def _write_calibration_file(onset_stats: dict, offset_stats: dict, path: Path = Path("calibration.json")) -> None:
    payload = {
        "onset": {
            "best_logit": onset_stats["best_logit"],
            "best_prob": float(onset_stats["best_prob"]),
            "temperature": onset_stats["temperature"],
            "logit_bias": onset_stats["logit_bias"],
            "platt_scale": onset_stats["platt_scale"],
            "platt_bias": onset_stats["platt_bias"],
            "scale": onset_stats["scale"],
            "calibrated_pred_rate": onset_stats["calibrated_pred_rate"],
            "pos_rate": onset_stats["pos_rate"],
            "provenance": "thorough",
        },
        "offset": {
            "best_logit": offset_stats["best_logit"],
            "best_prob": float(offset_stats["best_prob"]),
            "temperature": offset_stats["temperature"],
            "logit_bias": offset_stats["logit_bias"],
            "platt_scale": offset_stats["platt_scale"],
            "platt_bias": offset_stats["platt_bias"],
            "scale": offset_stats["scale"],
            "calibrated_pred_rate": offset_stats["calibrated_pred_rate"],
            "pos_rate": offset_stats["pos_rate"],
            "provenance": "thorough",
        },
        "platt_onset_scale": onset_stats["platt_scale"],
        "platt_onset_bias": onset_stats["platt_bias"],
        "temperature_onset": onset_stats["temperature"],
        "platt_offset_scale": offset_stats["platt_scale"],
        "platt_offset_bias": offset_stats["platt_bias"],
        "temperature_offset": offset_stats["temperature"],
    }

    out_path = Path(path)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")
    lock_path = out_path.with_name(f"{out_path.name}.lock")
    try:
        with FileLock(str(lock_path), timeout=1.0):
            with open(tmp_path, "w") as tmp_file:
                json.dump(payload, tmp_file, indent=2)
                tmp_file.flush()
                try:
                    os.fsync(tmp_file.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, out_path)
    except FileLockTimeout:
        print("[calib] skip write (lock timeout)", flush=True)
    except Exception as exc:
        print(f"[calib] failed to write calibration: {exc}", flush=True)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _fit_platt_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    head: str,
    max_iter: int = 300,
    lr: float = 0.05,
    l2_lambda: float = 1e-4,
) -> dict:
    """Fit Platt scaling (temperature + bias) via logistic regression."""

    prev_grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    t_start = time.monotonic()

    logits = logits.detach()
    targets = targets.detach().float()
    device = logits.device
    logits = logits.to(device)
    targets = targets.to(device)

    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1)

    total = int(flat_targets.numel())
    pos = int(flat_targets.sum().item())

    def _finalize(result_dict: dict, steps: int, loss_val: Optional[float]) -> dict:
        torch.set_grad_enabled(prev_grad_mode)
        elapsed = time.monotonic() - t_start
        loss_str = "nan"
        if loss_val is not None and math.isfinite(loss_val):
            loss_str = f"{loss_val:.6f}"
        print(
            "[platt] {head}: steps={steps} loss={loss} scale={scale:.6f} bias={bias:.6f} temp={temp:.6f} dt={dt:.3f}s".format(
                head=head,
                steps=steps,
                loss=loss_str,
                scale=result_dict.get("platt_scale", 1.0),
                bias=result_dict.get("platt_bias", 0.0),
                temp=result_dict.get("temperature", 1.0),
                dt=elapsed,
            ),
            flush=True,
        )
        return result_dict

    if total == 0:
        result = {
            "temperature": 1.0,
            "logit_bias": 0.0,
            "calibrated_pred_rate": 0.0,
            "scale": 1.0,
            "platt_bias": 0.0,
            "platt_scale": 1.0,
        }
        return _finalize(result, steps=0, loss_val=None)
    if pos <= 0 or pos >= total:
        print(f"[platt] skipped (degenerate labels): pos={pos} total={total}", flush=True)
        pred_rate_default = float(flat_targets.mean().detach().cpu()) if total > 0 else 0.0
        result = {
            "temperature": 1.0,
            "logit_bias": 0.0,
            "calibrated_pred_rate": pred_rate_default,
            "scale": 1.0,
            "platt_bias": 0.0,
            "platt_scale": 1.0,
        }
        return _finalize(result, steps=0, loss_val=None)

    if total > 500_000:
        positives = torch.nonzero(flat_targets > 0.5, as_tuple=False).reshape(-1)
        negatives = torch.nonzero(flat_targets < 0.5, as_tuple=False).reshape(-1)
        target_count = min(pos, 50_000, positives.numel(), negatives.numel())
        if target_count > 0:
            orig_total = total
            positives_cpu = positives.cpu()
            negatives_cpu = negatives.cpu()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(0)
            if positives_cpu.numel() > target_count:
                pos_perm = torch.randperm(positives_cpu.numel(), generator=generator)[:target_count]
                positives_cpu = positives_cpu.index_select(0, pos_perm)
            else:
                positives_cpu = positives_cpu[:target_count]
            if negatives_cpu.numel() > target_count:
                neg_perm = torch.randperm(negatives_cpu.numel(), generator=generator)[:target_count]
                negatives_cpu = negatives_cpu.index_select(0, neg_perm)
            else:
                negatives_cpu = negatives_cpu[:target_count]
            selected = torch.cat([positives_cpu, negatives_cpu], dim=0).to(flat_logits.device)
            flat_logits = flat_logits.index_select(0, selected)
            flat_targets = flat_targets.index_select(0, selected)
            flat_logits = flat_logits.clone()
            flat_targets = flat_targets.clone()
            total = int(flat_targets.numel())
            pos = int((flat_targets > 0.5).sum().item())
            neg = total - pos
            print(f"[platt] downsampled N={orig_total} -> N_sub={total} (pos={pos}, neg={neg})", flush=True)

    scale = torch.nn.Parameter(torch.ones((), device=device))
    bias = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.Adam([scale, bias], lr=lr)

    best_loss = float("inf")
    best_state = (scale.detach().clone(), bias.detach().clone())
    stagnation = 0
    prev_loss = None
    loss_history = []
    steps_run = 0

    for step in range(max_iter):
        opt.zero_grad()
        logits_adj = scale * flat_logits + bias
        loss = F.binary_cross_entropy_with_logits(logits_adj, flat_targets)
        if l2_lambda > 0.0:
            loss = loss + l2_lambda * ((scale - 1.0) ** 2 + bias**2)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu())
        if loss_val < best_loss - 1e-7:
            best_loss = loss_val
            best_state = (scale.detach().clone(), bias.detach().clone())
            stagnation = 0
        else:
            stagnation += 1

        if prev_loss is not None and abs(prev_loss - loss_val) < 5e-7:
            stagnation += 1
        prev_loss = loss_val
        loss_history.append(loss_val)
        steps_run = step + 1

        elapsed = time.monotonic() - t_start
        if elapsed > 5.0:
            break
        if step >= 20:
            ref_loss = loss_history[-21]
            rel_delta = abs(loss_val - ref_loss) / max(ref_loss, 1e-6)
            if rel_delta < 1e-3:
                break
        if stagnation >= 25:
            break

    final_scale, final_bias = best_state
    logits_adj = final_scale * flat_logits + final_bias
    probs = torch.sigmoid(logits_adj)

    scale_val = float(final_scale.detach().cpu())
    bias_val = float(final_bias.detach().cpu())
    pred_rate = float(probs.mean().detach().cpu())
    temperature = float(1.0 / max(scale_val, 1e-6))

    result = {
        "temperature": temperature,
        "logit_bias": bias_val,
        "calibrated_pred_rate": pred_rate,
        "scale": scale_val,
        "platt_bias": bias_val,
        "platt_scale": scale_val,
    }

    return _finalize(result, steps=steps_run, loss_val=best_loss if math.isfinite(best_loss) else None)


def _pool_roll_BT(x_btP: torch.Tensor, Tprime: int) -> torch.Tensor:
    """Downsample a (B,T,P) pianoroll along time using max pooling."""
    x = x_btP.permute(0, 2, 1)  # (B,P,T)
    x = F.adaptive_max_pool1d(x, Tprime)  # (B,P,T')
    return x.permute(0, 2, 1).contiguous()  # (B,T',P)

def _binary_f1(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Binary F1 score for tensors in {0,1}."""
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)

def _reliability_curve(probs: np.ndarray, targets: np.ndarray, n_bins: int, name: str):
    """Compute reliability data and save a diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    bin_sums = np.bincount(bin_ids, weights=probs, minlength=n_bins)
    bin_true = np.bincount(bin_ids, weights=targets, minlength=n_bins)
    bin_cnts = np.bincount(bin_ids, minlength=n_bins)

    nonzero = bin_cnts > 0
    prob_mean = np.zeros(n_bins)
    true_mean = np.zeros(n_bins)
    prob_mean[nonzero] = bin_sums[nonzero] / bin_cnts[nonzero]
    true_mean[nonzero] = bin_true[nonzero] / bin_cnts[nonzero]

    ece = np.sum(np.abs(true_mean - prob_mean) * bin_cnts / probs.size)
    brier = np.mean((probs - targets) ** 2)

    plt.figure()
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_mean[nonzero], true_mean[nonzero], marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(f"{name.capitalize()} reliability (ECE {ece:.3f}, Brier {brier:.3f})")
    plt.tight_layout()
    plt.savefig(f"calib_reliability_{name}.png")
    plt.close()
    return ece, brier

def _extract_lag_values(value):
    """Flatten lag payloads (ints, floats, tensors, lists) into plain floats."""

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
    """Collect lag source labels from nested payloads."""

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


def _dataset_video_count(ds) -> str:
    if ds is None:
        return "?"
    try:
        if hasattr(ds, "samples"):
            return str(len(getattr(ds, "samples")))
        if hasattr(ds, "videos"):
            vids = getattr(ds, "videos")
            try:
                return str(len(vids))
            except TypeError:
                pass
        return str(len(ds))
    except Exception:
        return "?"


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"

def _collect(model, loader, target_clips: Optional[int], timeout_secs: float):
    onset_logits_list, offset_logits_list = [], []
    onset_probs_list, offset_probs_list = [], []
    onset_tgts_list, offset_tgts_list = [], []
    lag_ms_samples = []
    lag_source_counts: Counter[str] = Counter()

    processed = 0
    timeout_hit = False
    target_total = None if target_clips is None else max(0, int(target_clips))
    start_time = time.monotonic()
    last_heartbeat = start_time

    if target_total is not None and target_total == 0:
        raise RuntimeError("target_clips resolved to 0; no clips available for calibration")

    with torch.no_grad():
        for batch in loader:
            remaining = None if target_total is None else target_total - processed
            if remaining is not None and remaining <= 0:
                break

            x = batch["video"]
            batch_size = x.shape[0]
            take = batch_size if remaining is None or batch_size <= remaining else max(0, int(remaining))
            if take <= 0:
                break
            idx = slice(None) if take == batch_size else slice(0, take)

            x = x[idx]
            out = model(x)

            on_logits = out["onset_logits"] if "onset_logits" in out else out.get("onset")
            off_logits = out["offset_logits"] if "offset_logits" in out else out.get("offset")

            onset_probs = torch.sigmoid(on_logits)
            offset_probs = torch.sigmoid(off_logits)

            onset_roll = batch["onset_roll"][idx].float()
            offset_roll = batch["offset_roll"][idx].float()

            T_logits = on_logits.shape[1]
            if onset_roll.shape[1] != T_logits:
                onset_roll = _pool_roll_BT(onset_roll, T_logits)
                offset_roll = _pool_roll_BT(offset_roll, T_logits)

            onset_roll = align_pitch_dim(on_logits, onset_roll, "onset")
            offset_roll = align_pitch_dim(off_logits, offset_roll, "offset")

            onset_roll = (onset_roll > 0).float()
            offset_roll = (offset_roll > 0).float()

            onset_logits_list.append(on_logits.cpu())
            offset_logits_list.append(off_logits.cpu())
            onset_probs_list.append(onset_probs.cpu())
            offset_probs_list.append(offset_probs.cpu())
            onset_tgts_list.append(onset_roll.cpu())
            offset_tgts_list.append(offset_roll.cpu())

            lag_ms_field = batch.get("lag_ms")
            if lag_ms_field is not None:
                subset = lag_ms_field
                if isinstance(lag_ms_field, (list, tuple)):
                    subset = lag_ms_field[:take]
                elif torch.is_tensor(lag_ms_field):
                    subset = lag_ms_field[:take]
                lag_vals = _extract_lag_values(subset)
                if lag_vals:
                    lag_ms_samples.extend(lag_vals)

            lag_source_field = batch.get("lag_source")
            if lag_source_field is not None:
                subset = lag_source_field
                if isinstance(lag_source_field, (list, tuple)):
                    subset = lag_source_field[:take]
                elif torch.is_tensor(lag_source_field):
                    subset = lag_source_field[:take]
                lag_sources = _extract_lag_sources(subset)
                if lag_sources:
                    lag_source_counts.update(lag_sources)

            processed += take
            _print_progress(processed, target_total)

            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                elapsed_str = _format_seconds(now - start_time)
                total_display = target_total if target_total is not None else "?"
                print(
                    f"[calib] collect heartbeat: processed={processed}/{total_display} elapsed={elapsed_str}",
                    flush=True,
                )
                last_heartbeat = now

            if timeout_secs and now - start_time >= timeout_secs:
                timeout_hit = True
                break

    if not onset_logits_list:
        raise RuntimeError("No clips processed during calibration")

    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    onset_probs = torch.cat(onset_probs_list, dim=0)
    offset_probs = torch.cat(offset_probs_list, dim=0)
    onset_tgts = torch.cat(onset_tgts_list, dim=0)
    offset_tgts = torch.cat(offset_tgts_list, dim=0)
    elapsed_total = time.monotonic() - start_time
    logits_shape_display = {
        "onset": tuple(onset_logits.shape),
        "offset": tuple(offset_logits.shape),
    }
    targets_shape_display = {
        "onset": tuple(onset_tgts.shape),
        "offset": tuple(offset_tgts.shape),
    }
    print(
        "[calib] collect done: logits_shape={} targets_shape={} elapsed={}".format(
            logits_shape_display,
            targets_shape_display,
            _format_seconds(elapsed_total),
        ),
        flush=True,
    )

    return (
        onset_logits,
        offset_logits,
        onset_probs,
        offset_probs,
        onset_tgts,
        offset_tgts,
        processed,
        timeout_hit,
        lag_ms_samples,
        lag_source_counts,
    )

def _compute_metrics(
    logits: torch.Tensor,
    probs: torch.Tensor,
    targets: torch.Tensor,
    name: str,
    *,
    prob_grid: Optional[torch.Tensor] = None,
    logit_grid: Optional[torch.Tensor] = None,
    agg_mode: str,
    agg_top_k: int,
    cap_count: int,
    decoder_kind: str,
    decoder_params: Mapping[str, Any],
    hop_seconds: float,
    event_tolerance: float,
):
    logits_flat = logits.reshape(-1)
    probs_flat = probs.reshape(-1)
    targets_flat = targets.reshape(-1)

    logit_grid_tensor = logit_grid if logit_grid is not None else DEFAULT_LOGIT_GRID
    logit_grid_tensor = logit_grid_tensor.to(logits_flat.device)
    prob_grid_tensor = prob_grid if prob_grid is not None else DEFAULT_PROB_GRID
    prob_grid_tensor = prob_grid_tensor.to(probs_flat.device)

    frame_best_logit, frame_best_f1_logit, frame_pred_rate_logit = -4.0, -1.0, 0.0
    for thr in logit_grid_tensor:
        thr_val = float(thr.item())
        pred = (logits_flat >= thr_val).float()
        f1 = _binary_f1(pred, targets_flat)
        if f1 > frame_best_f1_logit:
            frame_best_f1_logit = f1
            frame_best_logit = thr_val
            frame_pred_rate_logit = float(pred.mean().item())

    frame_best_prob, frame_best_f1_prob, frame_pred_rate_prob = 0.5, -1.0, 0.0
    for thr in prob_grid_tensor:
        thr_val = float(thr.item())
        pred = (probs_flat >= thr_val).float()
        f1 = _binary_f1(pred, targets_flat)
        if f1 > frame_best_f1_prob:
            frame_best_f1_prob = f1
            frame_best_prob = thr_val
            frame_pred_rate_prob = float(pred.mean().item())

    event_best_logit = float(frame_best_logit)
    event_best_f1_logit = -1.0
    event_pred_rate_logit = 0.0
    for thr in logit_grid_tensor:
        thr_val = float(thr.item())
        mask_bool = _build_threshold_mask(
            logits,
            thr_val,
            mode=agg_mode,
            cap_count=cap_count,
            top_k=agg_top_k,
        )
        mask_float = mask_bool.float()
        fallback_prob = 1.0 / (1.0 + math.exp(-thr_val))
        pred_bin = mask_float
        if decoder_kind == "hysteresis":
            open_thr, hold_thr = _resolve_decoder_thresholds(
                decoder_params,
                fallback_open=fallback_prob,
                default_hold=_DECODER_DEFAULTS[name]["hold"],
            )
            masked_probs = (probs * mask_float).contiguous()
            pred_mask = decode_hysteresis(
                masked_probs,
                open_thr,
                hold_thr,
                decoder_params["min_on"],
                decoder_params["min_off"],
                decoder_params["merge_gap"],
                decoder_params["median"],
            )
            pred_bin = pred_mask.to(mask_float.dtype)
        ev_f1 = _event_f1(pred_bin, targets, hop_seconds, event_tolerance)
        if ev_f1 is None:
            ev_f1 = 0.0
        pred_rate = float(pred_bin.mean().item())
        if ev_f1 > event_best_f1_logit + 1e-9:
            event_best_f1_logit = ev_f1
            event_best_logit = thr_val
            event_pred_rate_logit = pred_rate

    event_best_prob = float(frame_best_prob)
    event_best_f1_prob = -1.0
    event_pred_rate_prob = 0.0
    for thr in prob_grid_tensor:
        thr_val = float(thr.item())
        mask_bool = _build_threshold_mask(
            probs,
            thr_val,
            mode=agg_mode,
            cap_count=cap_count,
            top_k=agg_top_k,
        )
        mask_float = mask_bool.float()
        pred_bin = mask_float
        if decoder_kind == "hysteresis":
            open_thr, hold_thr = _resolve_decoder_thresholds(
                decoder_params,
                fallback_open=thr_val,
                default_hold=_DECODER_DEFAULTS[name]["hold"],
            )
            masked_probs = (probs * mask_float).contiguous()
            pred_mask = decode_hysteresis(
                masked_probs,
                open_thr,
                hold_thr,
                decoder_params["min_on"],
                decoder_params["min_off"],
                decoder_params["merge_gap"],
                decoder_params["median"],
            )
            pred_bin = pred_mask.to(mask_float.dtype)
        ev_f1 = _event_f1(pred_bin, targets, hop_seconds, event_tolerance)
        if ev_f1 is None:
            ev_f1 = 0.0
        pred_rate = float(pred_bin.mean().item())
        if ev_f1 > event_best_f1_prob + 1e-9:
            event_best_f1_prob = ev_f1
            event_best_prob = thr_val
            event_pred_rate_prob = pred_rate

    pos_rate = targets_flat.mean().item()
    ece, brier = _reliability_curve(probs_flat.numpy(), targets_flat.numpy(), 10, name)

    return {
        "best_logit": event_best_logit,
        "best_prob": event_best_prob,
        "f1_logit": event_best_f1_logit,
        "f1_prob": event_best_f1_prob,
        "pred_rate_logit": event_pred_rate_logit,
        "pred_rate_prob": event_pred_rate_prob,
        "pos_rate": pos_rate,
        "ece": ece,
        "brier": brier,
        "frame_f1_logit": frame_best_f1_logit,
        "frame_f1_prob": frame_best_f1_prob,
        "frame_pred_rate_logit": frame_pred_rate_logit,
        "frame_pred_rate_prob": frame_pred_rate_prob,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
    ap.add_argument("--split", choices=["train", "val", "test"], help="Dataset split to evaluate")
    ap.add_argument("--max-clips", type=int, help="Limit number of clips evaluated (default: config)")
    ap.add_argument("--frames", type=int, help="Frames per clip during calibration (default: config)")
    ap.add_argument(
        "--timeout-mins",
        type=float,
        help="Optional timeout in minutes; stops early while keeping partial stats",
    )
    ap.add_argument(
        "--no-avlag",
        action="store_true",
        help="Disable audio/video lag estimation when preparing frame targets",
    )
    ap.add_argument(
        "--verbose",
        choices=["quiet", "info", "debug"],
        help="Logging verbosity (default: quiet or $TIVIT_VERBOSE)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Force single-process dataloading like fast eval (num_workers=0, no pinning)",
    )
    ap.add_argument("--seed", type=int, help="Seed for RNGs and dataloaders")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle deterministic torch backends (default: config or enabled)",
    )
    args = ap.parse_args()
    args.verbose = configure_verbosity(args.verbose)

    avlag_enabled = not args.no_avlag
    env_disable = str(os.environ.get("AVSYNC_DISABLE", "")).strip().lower()
    if env_disable in {"1", "true", "yes", "on"}:
        avlag_enabled = False

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
    metrics_cfg = cfg.get("training", {}).get("metrics", {}) or {}
    agg_cfg = metrics_cfg.get("aggregation", {}) or {}
    agg_mode = str(agg_cfg.get("mode", "any")).lower()
    agg_top_k = int(agg_cfg.get("top_k", 0) or 0)
    agg_k_cfg = agg_cfg.get("k", {}) or {}
    agg_k_onset = max(1, int(agg_k_cfg.get("onset", 1) or 1))
    agg_k_offset = max(1, int(agg_k_cfg.get("offset", 1) or 1))
    decoder_kind = "hysteresis"
    decoder_params = _resolve_decoder_from_config(metrics_cfg)
    decoder_settings_summary = _format_decoder_settings(decoder_kind, decoder_params)
    print(f"[decoder-settings] {decoder_settings_summary}")
    print(f"[decoder] {_decoder_notice_text(decoder_kind, decoder_params)}")

    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    cfg["dataset"] = dataset_cfg
    if args.max_clips is not None:
        dataset_cfg["max_clips"] = int(args.max_clips)
    if args.frames is not None:
        dataset_cfg["frames"] = int(args.frames)
    if not avlag_enabled:
        dataset_cfg["avlag_disabled"] = True
    if args.debug:
        dataset_cfg["num_workers"] = 0
        dataset_cfg["persistent_workers"] = False
        dataset_cfg["pin_memory"] = False
        print(
            "[progress] debug mode: forcing num_workers=0, persistent_workers=False, pin_memory=False.",
            flush=True,
        )

    decode_fps = float(dataset_cfg.get("decode_fps", 0.0) or 0.0)
    hop_seconds = float(dataset_cfg.get("hop_seconds", 0.0) or 0.0)
    if hop_seconds <= 0.0:
        hop_seconds = 1.0 / decode_fps if decode_fps > 0 else 1.0
    frame_targets_cfg = dataset_cfg.get("frame_targets", {}) or {}
    event_tolerance = float(frame_targets_cfg.get("tolerance", hop_seconds))

    split = args.split or dataset_cfg.get("split_val") or dataset_cfg.get("split") or "val"
    frames_display = dataset_cfg.get("frames")
    max_clips_display = dataset_cfg.get("max_clips")
    frame_text = frames_display if frames_display is not None else "?"
    max_clips_text = max_clips_display if max_clips_display is not None else "?"
    print(
        f"[progress] starting (split={split}, frames={frame_text}, max_clips={max_clips_text})",
        flush=True,
    )

    stage_durations: dict = {}
    t_main_start = time.time()
    t_dataset0 = time.time()
    loader = make_dataloader(cfg, split=split, seed=seed)
    if isinstance(loader, dict):
        loader = loader.get(split) or next(iter(loader.values()))
    if isinstance(loader, (list, tuple)):
        loader = loader[0]
    base_dataset = getattr(loader, "dataset", None)

    cfg_cap = dataset_cfg.get("max_clips")
    cap_candidate: Optional[int] = None
    if isinstance(cfg_cap, int):
        cap_candidate = max(0, cfg_cap)
    if args.max_clips is not None:
        cap_candidate = max(0, args.max_clips) if cap_candidate is None else min(cap_candidate, max(0, args.max_clips))

    if base_dataset is not None and hasattr(base_dataset, "materialize_eval_entries_from_labels"):
        try:
            base_dataset.materialize_eval_entries_from_labels(
                max_total=cap_candidate,
                target_T=getattr(base_dataset, "frames", None),
            )
        except Exception as exc:
            print(f"[warn] materialize_eval_entries_from_labels failed: {exc}", flush=True)

    dataset_elapsed = time.time() - t_dataset0
    stage_durations["dataset_init"] = dataset_elapsed
    dataset = base_dataset
    dataset_name = dataset.__class__.__name__ if dataset is not None else type(loader).__name__
    batch_size_val = getattr(loader, "batch_size", None)
    batch_display = str(batch_size_val) if batch_size_val is not None else "?"
    worker_count = getattr(loader, "num_workers", None)
    worker_display = str(worker_count) if worker_count is not None else "?"
    video_count_display = _dataset_video_count(dataset)

    materialize_stats = getattr(dataset, "_eval_materialize_stats", {}) or {}
    ok_videos = 0
    materialize_duration = 0.0
    if isinstance(materialize_stats, dict):
        try:
            ok_videos = int(materialize_stats.get("videos") or 0)
        except (TypeError, ValueError):
            ok_videos = 0
        try:
            materialize_duration = float(materialize_stats.get("duration") or 0.0)
        except (TypeError, ValueError):
            materialize_duration = 0.0
    if materialize_duration == 0.0 and dataset is not None:
        try:
            materialize_duration = float(getattr(dataset, "_last_materialize_duration", 0.0) or 0.0)
        except (TypeError, ValueError):
            materialize_duration = 0.0
    if materialize_duration > 0:
        stage_durations["materialize"] = materialize_duration

    ds_len: Optional[int] = None
    dataset_count = "?"
    if dataset is not None:
        try:
            ds_len = len(dataset)
            dataset_count = str(ds_len)
        except TypeError:
            ds_len = None

    print(
        f"[progress] dataset ready (videos={video_count_display}, workers={worker_display})",
        flush=True,
    )
    print(
        f"[progress] dataset ready in {_format_seconds(dataset_elapsed)} ({dataset_elapsed:.2f}s) "
        f"backend={dataset_name} len={dataset_count} batch={batch_display}",
        flush=True,
    )
    lag_label = "guardrail" if avlag_enabled else "no_avlag"
    frame_spec = getattr(dataset, "frame_target_spec", None) if dataset is not None else None
    if frame_spec is not None:
        tol_val = float(getattr(frame_spec, "tolerance", 0.0))
        tol_str = f"{tol_val:.3f}".rstrip("0").rstrip(".")
        tol_display = f"{tol_str}s" if tol_str else f"{tol_val:.3f}s"
        summary = (
            "targets_conf: "
            f"T={frame_spec.frames}, "
            f"tol={tol_display}, "
            f"dilate={frame_spec.dilation}, "
            f"lag_source={lag_label}, "
            f"fps={frame_spec.fps:.3f}, "
            f"cache_key={frame_spec.cache_key_prefix}"
        )
        print(f"[progress] {summary}", flush=True)
    else:
        frame_summary = getattr(dataset, "frame_target_summary", None) if dataset is not None else None
        if frame_summary:
            display = frame_summary
            if not avlag_enabled and "lag_source=" in display:
                prefix, suffix = display.split("lag_source=", 1)
                if "," in suffix:
                    _, tail = suffix.split(",", 1)
                    display = f"{prefix}lag_source=no_avlag,{tail}"
                else:
                    display = f"{prefix}lag_source=no_avlag"
            print(f"[progress] {display}", flush=True)

    if materialize_stats:
        pos_total = int(materialize_stats.get("positives", 0) or 0)
        neg_total = int(materialize_stats.get("negatives", 0) or 0)
        total_entries = pos_total + neg_total
        avg_per_video = float(materialize_stats.get("avg_per_video", 0.0) or 0.0)
        print(
            "[progress] materialize ready "
            f"(entries={total_entries} pos={pos_total} neg={neg_total} videos={ok_videos} "
            f"avg_per_video≈{avg_per_video:.1f} duration={_format_seconds(materialize_duration)} ({materialize_duration:.2f}s))",
            flush=True,
        )

    target_clips: Optional[int]
    if ds_len is not None:
        resolved_cap = ds_len
        if cap_candidate is not None:
            resolved_cap = min(resolved_cap, cap_candidate)
        target_clips = int(resolved_cap)
    else:
        target_clips = int(cap_candidate) if cap_candidate is not None else None

    if dataset is not None and target_clips is not None:
        try:
            base_len = len(dataset)
        except TypeError:
            base_len = None
        if base_len is not None and target_clips < base_len:
            subset_indices = list(range(target_clips))
            subset_ds = Subset(dataset, subset_indices)
            num_workers = getattr(loader, "num_workers", 0)
            persistent_workers = getattr(loader, "persistent_workers", False)
            if num_workers <= 0:
                persistent_workers = False
            loader_kwargs = {
                "batch_size": getattr(loader, "batch_size", 1),
                "shuffle": False,
                "num_workers": num_workers,
                "pin_memory": getattr(loader, "pin_memory", False),
                "drop_last": getattr(loader, "drop_last", False),
                "collate_fn": getattr(loader, "collate_fn", None),
                "persistent_workers": persistent_workers,
                "timeout": getattr(loader, "timeout", 0),
            }
            prefetch_factor = getattr(loader, "prefetch_factor", None)
            if num_workers > 0 and prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
            pin_memory_device = getattr(loader, "pin_memory_device", None)
            if pin_memory_device:
                loader_kwargs["pin_memory_device"] = pin_memory_device
            worker_init_fn = getattr(loader, "worker_init_fn", None)
            if worker_init_fn is not None:
                loader_kwargs["worker_init_fn"] = worker_init_fn
            generator = getattr(loader, "generator", None)
            if generator is not None:
                loader_kwargs["generator"] = generator
            multiprocessing_context = getattr(loader, "multiprocessing_context", None)
            if multiprocessing_context is not None:
                loader_kwargs["multiprocessing_context"] = multiprocessing_context
            loader = DataLoader(subset_ds, **loader_kwargs)
            dataset = getattr(loader, "dataset", subset_ds)
            target_clips = len(dataset)

    if target_clips is not None and target_clips <= 0:
        raise RuntimeError("target_clips resolved to 0; dataset is empty after materialization.")

    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    timeout_secs = float(args.timeout_mins * 60.0) if args.timeout_mins else 0.0
    t_data0 = time.time()
    (
        onset_logits,
        offset_logits,
        onset_probs,
        offset_probs,
        onset_tgts,
        offset_tgts,
        processed_clips,
        timeout_hit,
        lag_ms_samples,
        lag_source_counts,
    ) = _collect(model, loader, target_clips, timeout_secs)
    data_elapsed = time.time() - t_data0
    stage_durations["data_pass"] = data_elapsed

    if timeout_hit:
        print(f"[calib] timeout reached after {processed_clips} clips", flush=True)

    if avlag_enabled:
        if lag_ms_samples:
            lag_arr = np.asarray(lag_ms_samples, dtype=np.float32)
            lag_mean = float(lag_arr.mean())
            lag_median = float(np.median(lag_arr))
            lag_p95 = float(np.percentile(lag_arr, 95))
            print(
                "[progress] A/V lag ms stats: mean={:.1f} median={:.1f} p95={:.1f} samples={}".format(
                    lag_mean,
                    lag_median,
                    lag_p95,
                    lag_arr.size,
                ),
                flush=True,
            )
        else:
            print("[progress] A/V lag ms stats: no samples collected.", flush=True)
        if lag_source_counts:
            top_sources = ", ".join(f"{src}:{cnt}" for src, cnt in lag_source_counts.most_common(3))
            print(f"[progress] lag sources top: {top_sources}", flush=True)
    else:
        print("[progress] A/V lag ms stats: disabled (all zero).", flush=True)

    onset_stats = _compute_metrics(
        onset_logits,
        onset_probs,
        onset_tgts,
        "onset",
        prob_grid=DEFAULT_PROB_GRID,
        logit_grid=DEFAULT_LOGIT_GRID,
        agg_mode=agg_mode,
        agg_top_k=agg_top_k,
        cap_count=agg_k_onset,
        decoder_kind=decoder_kind,
        decoder_params=decoder_params["onset"],
        hop_seconds=hop_seconds,
        event_tolerance=event_tolerance,
    )
    offset_stats = _compute_metrics(
        offset_logits,
        offset_probs,
        offset_tgts,
        "offset",
        prob_grid=OFFSET_PROB_GRID,
        logit_grid=DEFAULT_LOGIT_GRID,
        agg_mode=agg_mode,
        agg_top_k=agg_top_k,
        cap_count=agg_k_offset,
        decoder_kind=decoder_kind,
        decoder_params=decoder_params["offset"],
        hop_seconds=hop_seconds,
        event_tolerance=event_tolerance,
    )
    onset_platt = _fit_platt_scaling(onset_logits, onset_tgts, head="onset")
    offset_platt = _fit_platt_scaling(offset_logits, offset_tgts, head="offset")
    onset_stats.update(onset_platt)
    offset_stats.update(offset_platt)

    _write_calibration_file(onset_stats, offset_stats)

    total_elapsed = time.time() - t_main_start
    stage_order = [
        ("dataset_init", "dataset"),
        ("materialize", "materialize"),
        ("data_pass", "data_pass"),
    ]
    stage_parts = []
    for key, label in stage_order:
        if key in stage_durations:
            dur_val = stage_durations[key]
            stage_parts.append(f"{label}={_format_seconds(dur_val)}")
    stage_summary = ", ".join(stage_parts) if stage_parts else "n/a"
    print(
        f"[progress] stages: {stage_summary} | total={_format_seconds(total_elapsed)} ({total_elapsed:.2f}s)",
        flush=True,
    )

    for name, stats in [("Onset", onset_stats), ("Offset", offset_stats)]:
        diff = stats["calibrated_pred_rate"] - stats["pos_rate"]
        print(
            f"{name}: pos_rate={stats['pos_rate']:.4f} | "
            f"best_logit={stats['best_logit']:.2f} "
            f"(event_F1={stats['f1_logit']:.3f}, frame_F1={stats['frame_f1_logit']:.3f}, pred_rate={stats['pred_rate_logit']:.4f}) | "
            f"best_prob={stats['best_prob']:.2f} "
            f"(event_F1={stats['f1_prob']:.3f}, frame_F1={stats['frame_f1_prob']:.3f}, pred_rate={stats['pred_rate_prob']:.4f}) | "
            f"temp={stats['temperature']:.3f} bias={stats['logit_bias']:.3f} calibrated_rate={stats['calibrated_pred_rate']:.4f} (Δ={diff:+.4f})"
        )

if __name__ == "__main__":
    main()
