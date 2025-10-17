#!/usr/bin/env python3
"""Purpose:
    Evaluate onset and offset predictions to determine calibrated logit or
    probability thresholds, emit reliability diagnostics, and continuously write
    partial results to ``calibration.json`` during long sweeps.

Key Functions/Classes:
    - _collect(): Runs the model across a dataloader to gather logits,
      probabilities, and aligned targets while checkpointing partial metrics.
    - _compute_metrics(): Sweeps thresholds to compute F1 scores, prediction
      rates, expected calibration error, and Brier scores.
    - main(): Command-line entry point that loads checkpoints, runs evaluation,
      and writes calibration summaries.

CLI:
    Invoke ``python scripts/calib/calibrate_thresholds.py --ckpt <path> --split val``
    with optional ``--max-clips``/``--frames`` overrides or ``--timeout-mins`` to
    stop early while preserving partial statistics.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# -----------------------------------------------------------------------------
# Repo setup so we can import from src/
# -----------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from utils import load_config, align_pitch_dim
from data import make_dataloader
from models import build_model


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

PARTIAL_WRITE_INTERVAL = 32

DEFAULT_LOGIT_GRID = torch.arange(-4.0, 2.0 + 1e-9, 0.05)
DEFAULT_PROB_GRID = torch.arange(0.01, 0.99 + 1e-9, 0.01)
_OFFSET_EXTRA = torch.arange(0.99, 1.001 + 1e-9, 0.002)
OFFSET_PROB_GRID = torch.unique(torch.cat([DEFAULT_PROB_GRID, _OFFSET_EXTRA]))
OFFSET_PROB_GRID = torch.sort(OFFSET_PROB_GRID).values.clamp(max=0.999)


def _print_progress(processed: int, total: Optional[int]) -> None:
    if total is not None and total > 0:
        pct = 100.0 * processed / total
        msg = f"[calib] processed {processed}/{total} clips ({pct:5.1f}%)"
    else:
        msg = f"[calib] processed {processed} clips"
    print(msg, flush=True)


def _write_partial_calibration(
    onset_logits_list,
    offset_logits_list,
    onset_probs_list,
    offset_probs_list,
    onset_tgts_list,
    offset_tgts_list,
) -> None:
    if not onset_logits_list:
        return
    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    onset_probs = torch.cat(onset_probs_list, dim=0)
    offset_probs = torch.cat(offset_probs_list, dim=0)
    onset_tgts = torch.cat(onset_tgts_list, dim=0)
    offset_tgts = torch.cat(offset_tgts_list, dim=0)
    onset_stats = _compute_metrics(
        onset_logits,
        onset_probs,
        onset_tgts,
        "onset",
        prob_grid=DEFAULT_PROB_GRID,
        logit_grid=DEFAULT_LOGIT_GRID,
    )
    offset_stats = _compute_metrics(
        offset_logits,
        offset_probs,
        offset_tgts,
        "offset",
        prob_grid=OFFSET_PROB_GRID,
        logit_grid=DEFAULT_LOGIT_GRID,
    )
    with torch.enable_grad():
        onset_cal = _fit_platt_scaling(onset_logits, onset_tgts)
        offset_cal = _fit_platt_scaling(offset_logits, offset_tgts)
    onset_stats.update(onset_cal)
    offset_stats.update(offset_cal)
    with open("calibration.json", "w") as f:
        json.dump(
            {
                "onset": {
                    "best_logit": onset_stats["best_logit"],
                    "best_prob": onset_stats["best_prob"],
                    "temperature": onset_stats["temperature"],
                    "logit_bias": onset_stats["logit_bias"],
                    "platt_scale": onset_stats["platt_scale"],
                    "platt_bias": onset_stats["platt_bias"],
                    "scale": onset_stats["scale"],
                    "calibrated_pred_rate": onset_stats["calibrated_pred_rate"],
                    "pos_rate": onset_stats["pos_rate"],
                },
                "offset": {
                    "best_logit": offset_stats["best_logit"],
                    "best_prob": offset_stats["best_prob"],
                    "temperature": offset_stats["temperature"],
                    "logit_bias": offset_stats["logit_bias"],
                    "platt_scale": offset_stats["platt_scale"],
                    "platt_bias": offset_stats["platt_bias"],
                    "scale": offset_stats["scale"],
                    "calibrated_pred_rate": offset_stats["calibrated_pred_rate"],
                    "pos_rate": offset_stats["pos_rate"],
                },
                "platt_onset_scale": onset_stats["platt_scale"],
                "platt_onset_bias": onset_stats["platt_bias"],
                "temperature_onset": onset_stats["temperature"],
                "platt_offset_scale": offset_stats["platt_scale"],
                "platt_offset_bias": offset_stats["platt_bias"],
                "temperature_offset": offset_stats["temperature"],
            },
            f,
            indent=2,
        )


def _fit_platt_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_iter: int = 300,
    lr: float = 0.05,
    l2_lambda: float = 1e-4,
) -> dict:
    """Fit Platt scaling (temperature + bias) via logistic regression."""

    torch.set_grad_enabled(True)

    logits = logits.detach()
    targets = targets.detach().float()
    device = logits.device
    logits = logits.to(device)
    targets = targets.to(device)

    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1)

    total = flat_targets.numel()
    pos = flat_targets.sum().item()
    if total == 0:
        return {
            "temperature": 1.0,
            "logit_bias": 0.0,
            "calibrated_pred_rate": 0.0,
            "scale": 1.0,
            "platt_bias": 0.0,
            "platt_scale": 1.0,
        }
    if pos <= 0 or pos >= total:
        print(f"[platt] skipped (degenerate labels): pos={int(pos)} total={int(total)}", flush=True)
        pred_rate_default = float(flat_targets.mean().detach().cpu()) if total > 0 else 0.0
        return {
            "temperature": 1.0,
            "logit_bias": 0.0,
            "calibrated_pred_rate": pred_rate_default,
            "scale": 1.0,
            "platt_bias": 0.0,
            "platt_scale": 1.0,
        }

    scale = torch.nn.Parameter(torch.ones((), device=device))
    bias = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.Adam([scale, bias], lr=lr)

    best_loss = float("inf")
    best_state = (scale.detach().clone(), bias.detach().clone())

    for _ in range(max_iter):
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

    final_scale, final_bias = best_state
    logits_adj = final_scale * flat_logits + final_bias
    probs = torch.sigmoid(logits_adj)

    scale_val = float(final_scale.detach().cpu())
    bias_val = float(final_bias.detach().cpu())
    pred_rate = float(probs.mean().detach().cpu())
    temperature = float(1.0 / max(scale_val, 1e-6))

    return {
        "temperature": temperature,
        "logit_bias": bias_val,
        "calibrated_pred_rate": pred_rate,
        "scale": scale_val,
        "platt_bias": bias_val,
        "platt_scale": scale_val,
    }


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

            if processed % PARTIAL_WRITE_INTERVAL == 0 or (
                target_total is not None and processed >= target_total
            ):
                _write_partial_calibration(
                    onset_logits_list,
                    offset_logits_list,
                    onset_probs_list,
                    offset_probs_list,
                    onset_tgts_list,
                    offset_tgts_list,
                )

            if timeout_secs and time.monotonic() - start_time >= timeout_secs:
                timeout_hit = True
                _write_partial_calibration(
                    onset_logits_list,
                    offset_logits_list,
                    onset_probs_list,
                    offset_probs_list,
                    onset_tgts_list,
                    offset_tgts_list,
                )
                break

    if not onset_logits_list:
        raise RuntimeError("No clips processed during calibration")

    _write_partial_calibration(
        onset_logits_list,
        offset_logits_list,
        onset_probs_list,
        offset_probs_list,
        onset_tgts_list,
        offset_tgts_list,
    )

    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    onset_probs = torch.cat(onset_probs_list, dim=0)
    offset_probs = torch.cat(offset_probs_list, dim=0)
    onset_tgts = torch.cat(onset_tgts_list, dim=0)
    offset_tgts = torch.cat(offset_tgts_list, dim=0)

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
):
    logits_flat = logits.reshape(-1)
    probs_flat = probs.reshape(-1)
    targets_flat = targets.reshape(-1)

    if logit_grid is None:
        logit_grid = DEFAULT_LOGIT_GRID
    logit_grid = logit_grid.to(logits_flat.device)
    best_logit, best_f1_logit, pred_rate_logit = -4.0, -1.0, 0.0
    for thr in logit_grid:
        pred = (logits_flat >= thr).float()
        f1 = _binary_f1(pred, targets_flat)
        if f1 > best_f1_logit:
            best_f1_logit = f1
            best_logit = float(thr.item())
            pred_rate_logit = pred.mean().item()

    if prob_grid is None:
        prob_grid = DEFAULT_PROB_GRID
    prob_grid = prob_grid.to(probs_flat.device)
    best_prob, best_f1_prob, pred_rate_prob = 0.5, -1.0, 0.0
    for thr in prob_grid:
        pred = (probs_flat >= thr).float()
        f1 = _binary_f1(pred, targets_flat)
        if f1 > best_f1_prob:
            best_f1_prob = f1
            best_prob = float(thr.item())
            pred_rate_prob = pred.mean().item()

    pos_rate = targets_flat.mean().item()
    ece, brier = _reliability_curve(probs_flat.numpy(), targets_flat.numpy(), 10, name)

    return {
        "best_logit": best_logit,
        "best_prob": best_prob,
        "f1_logit": best_f1_logit,
        "f1_prob": best_f1_prob,
        "pred_rate_logit": pred_rate_logit,
        "pred_rate_prob": pred_rate_prob,
        "pos_rate": pos_rate,
        "ece": ece,
        "brier": brier,
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
        "--debug",
        action="store_true",
        help="Force single-process dataloading like fast eval (num_workers=0, no pinning)",
    )
    args = ap.parse_args()

    avlag_enabled = not args.no_avlag
    env_disable = str(os.environ.get("AVSYNC_DISABLE", "")).strip().lower()
    if env_disable in {"1", "true", "yes", "on"}:
        avlag_enabled = False

    cfg = dict(load_config("configs/config.yaml"))
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
        print("[debug] num_workers=0, persistent_workers=False, pin_memory=False", flush=True)

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
    loader = make_dataloader(cfg, split=split)
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

    if ds_len is not None:
        if cap_candidate is None:
            target_clips = int(ds_len)
        else:
            target_clips = int(min(ds_len, cap_candidate))
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
    )
    offset_stats = _compute_metrics(
        offset_logits,
        offset_probs,
        offset_tgts,
        "offset",
        prob_grid=OFFSET_PROB_GRID,
        logit_grid=DEFAULT_LOGIT_GRID,
    )
    onset_platt = _fit_platt_scaling(onset_logits, onset_tgts)
    offset_platt = _fit_platt_scaling(offset_logits, offset_tgts)
    onset_stats.update(onset_platt)
    offset_stats.update(offset_platt)

    print(
        "[platt] onset: scale={:.3f} bias={:.3f} temp={:.3f} pos_rate={:.4f}".format(
            onset_stats["platt_scale"],
            onset_stats["platt_bias"],
            onset_stats["temperature"],
            onset_stats["pos_rate"],
        ),
        flush=True,
    )
    print(
        "[platt] offset: scale={:.3f} bias={:.3f} temp={:.3f} pos_rate={:.4f}".format(
            offset_stats["platt_scale"],
            offset_stats["platt_bias"],
            offset_stats["temperature"],
            offset_stats["pos_rate"],
        ),
        flush=True,
    )

    with open("calibration.json", "w") as f:
        json.dump(
            {
                "onset": {
                    "best_logit": onset_stats["best_logit"],
                    "best_prob": onset_stats["best_prob"],
                    "temperature": onset_stats["temperature"],
                    "logit_bias": onset_stats["logit_bias"],
                    "platt_scale": onset_stats["platt_scale"],
                    "platt_bias": onset_stats["platt_bias"],
                    "scale": onset_stats["scale"],
                    "calibrated_pred_rate": onset_stats["calibrated_pred_rate"],
                    "pos_rate": onset_stats["pos_rate"],
                },
                "offset": {
                    "best_logit": offset_stats["best_logit"],
                    "best_prob": offset_stats["best_prob"],
                    "temperature": offset_stats["temperature"],
                    "logit_bias": offset_stats["logit_bias"],
                    "platt_scale": offset_stats["platt_scale"],
                    "platt_bias": offset_stats["platt_bias"],
                    "scale": offset_stats["scale"],
                    "calibrated_pred_rate": offset_stats["calibrated_pred_rate"],
                    "pos_rate": offset_stats["pos_rate"],
                },
                "platt_onset_scale": onset_stats["platt_scale"],
                "platt_onset_bias": onset_stats["platt_bias"],
                "temperature_onset": onset_stats["temperature"],
                "platt_offset_scale": offset_stats["platt_scale"],
                "platt_offset_bias": offset_stats["platt_bias"],
                "temperature_offset": offset_stats["temperature"],
            },
            f,
            indent=2,
        )

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
            f"best_logit={stats['best_logit']:.2f} (pred_rate={stats['pred_rate_logit']:.4f}, F1={stats['f1_logit']:.3f}) | "
            f"best_prob={stats['best_prob']:.2f} (pred_rate={stats['pred_rate_prob']:.4f}, F1={stats['f1_prob']:.3f}) | "
            f"temp={stats['temperature']:.3f} bias={stats['logit_bias']:.3f} calibrated_rate={stats['calibrated_pred_rate']:.4f} (Δ={diff:+.4f})"
        )

if __name__ == "__main__":
    main()
