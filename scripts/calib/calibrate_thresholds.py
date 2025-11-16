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
    - decoder.decode module: Shared decoder utilities (pooling, hysteresis,
      threshold normalization) imported here so this file stays focused on
      experiments.
    - --legacy-calibrate-thresholds: One-flag rollback to the frozen
      ``calibrate_thresholds_legacy.py`` implementation.

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
    --legacy-calibrate-thresholds (default: False)
        Execute the pre-refactor fallback module for quick recovery if needed.

Usage:
    python scripts/calib/calibrate_thresholds.py --ckpt checkpoints/latest.pt --split val
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, List, Dict, Sequence, Tuple

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

from decoder.decode import (
    DECODER_DEFAULTS,
    build_threshold_mask,
    decode_hysteresis,
    decoder_notice_text,
    format_decoder_settings,
    pool_roll_BT,
    resolve_decoder_from_config,
    resolve_decoder_gates,
)
from tivit.decoder.global_fusion import (
    GlobalFusionConfig,
    FusionDebugState,
    resolve_global_fusion_config,
    build_batch_tile_mask,
    fuse_tile_logits,
)
from tivit.decoder.tile_keymap import TileMaskResult
from utils import load_config, align_pitch_dim, configure_verbosity, canonical_video_id
from utils.time_grid import frame_to_sec
from data import make_dataloader
from models import build_model
from utils.determinism import configure_determinism, resolve_deterministic_flag, resolve_seed
from utils.registration_refinement import RegistrationRefiner, resolve_registration_cache_path
from theory.key_prior_runtime import (
    KeyPriorRuntimeSettings,
    resolve_key_prior_settings,
    apply_key_prior_to_logits,
)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

HEARTBEAT_SECONDS = 10.0

DEFAULT_LOGIT_GRID = torch.arange(-4.0, 2.0 + 1e-9, 0.05)
DEFAULT_PROB_GRID = torch.arange(0.01, 0.99 + 1e-9, 0.01)
_OFFSET_EXTRA = torch.arange(0.99, 1.001 + 1e-9, 0.002)
OFFSET_PROB_GRID = torch.unique(torch.cat([DEFAULT_PROB_GRID, _OFFSET_EXTRA]))
OFFSET_PROB_GRID = torch.sort(OFFSET_PROB_GRID).values.clamp(max=0.999)

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


def _load_registration_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[canonical_video_id(str(key))] = value
    return result


def _resolve_clip_ids(paths: Sequence[str], count: int) -> List[Optional[str]]:
    clip_ids: List[Optional[str]] = []
    for idx in range(max(0, int(count))):
        clip_id = None
        if idx < len(paths):
            clip_id = canonical_video_id(Path(paths[idx]).stem)
        clip_ids.append(clip_id)
    return clip_ids

def _collect(
    model,
    loader,
    target_clips: Optional[int],
    timeout_secs: float,
    *,
    key_prior: KeyPriorRuntimeSettings,
    key_prior_fps: float,
    key_prior_midi_low: int | None,
    key_prior_midi_high: int | None,
    return_per_tile: bool,
    fusion_cfg: GlobalFusionConfig,
    model_tiles: int,
    preview_prob_threshold: float,
    registration_refiner: Optional[RegistrationRefiner] = None,
    reg_meta_cache: Optional[Dict[str, Dict[str, Any]]] = None,
):
    onset_logits_list, offset_logits_list = [], []
    onset_probs_list, offset_probs_list = [], []
    onset_tgts_list, offset_tgts_list = [], []
    lag_ms_samples = []
    lag_source_counts: Counter[str] = Counter()
    fusion_enabled = fusion_cfg.enabled
    if fusion_enabled and not return_per_tile:
        raise RuntimeError("Global fusion requires return_per_tile logits during calibration")
    fusion_debug_state = FusionDebugState(model_tiles) if fusion_enabled else None
    comparison_enabled = fusion_cfg.consistency_check and fusion_cfg.consistency_batches > 0
    comparison_batches_used = 0
    tile_mask_cache: Dict[str, TileMaskResult] = {}
    effective_meta_cache: Dict[str, Dict[str, Any]] = dict(reg_meta_cache or {})
    if fusion_enabled and not effective_meta_cache:
        reg_cache_path = resolve_registration_cache_path(os.environ.get("TIVIT_REG_REFINED"))
        effective_meta_cache = _load_registration_metadata(reg_cache_path)

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

            raw_paths = batch.get("path")
            if isinstance(raw_paths, (list, tuple)):
                paths = [str(p) for p in raw_paths]
            elif raw_paths is None:
                paths = []
            else:
                paths = [str(raw_paths)]

            x = batch["video"]
            batch_size = x.shape[0]
            take = batch_size if remaining is None or batch_size <= remaining else max(0, int(remaining))
            if take <= 0:
                break
            idx = slice(None) if take == batch_size else slice(0, take)

            clip_ids: List[Optional[str]] = []
            if fusion_enabled:
                if paths and take < len(paths):
                    paths = paths[:take]
                clip_ids = _resolve_clip_ids(paths, take)

            x = x[idx]
            out = model(x, return_per_tile=return_per_tile)

            comparison_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            if fusion_enabled:
                onset_tile = out.get("onset_tile")
                offset_tile = out.get("offset_tile")
                pitch_tile = out.get("pitch_tile")
                if onset_tile is None or offset_tile is None:
                    raise RuntimeError("global fusion enabled but model did not return per-tile logits")
                tile_mask_batch = build_batch_tile_mask(
                    clip_ids,
                    reg_meta_cache=effective_meta_cache,
                    reg_refiner=registration_refiner,
                    mask_cache=tile_mask_cache,
                    num_tiles=model_tiles,
                    cushion_keys=fusion_cfg.cushion_keys,
                    n_keys=int(onset_tile.shape[-1]),
                )
                tile_mask_tensor = tile_mask_batch.tensor
                mask_device = onset_tile.device
                mask_tensor_device = tile_mask_tensor.to(mask_device, dtype=onset_tile.dtype)
                if fusion_debug_state is not None:
                    for idx, record in enumerate(tile_mask_batch.records):
                        clip_ref = clip_ids[idx] if idx < len(clip_ids) else None
                        fusion_debug_state.record_mask_result(record, clip_id=clip_ref)
                apply_heads = {head.lower() for head in fusion_cfg.apply_to}
                if "onset" in apply_heads:
                    fused_onset = fuse_tile_logits(onset_tile, mask_tensor_device, mode=fusion_cfg.mode)
                    if fusion_debug_state is not None:
                        fusion_debug_state.record_shape("onset", onset_tile.shape, fused_onset.shape)
                    comparison_pairs["onset"] = (onset_entry.detach(), fused_onset.detach())
                    onset_entry = fused_onset
                if "offset" in apply_heads:
                    fused_offset = fuse_tile_logits(offset_tile, mask_tensor_device, mode=fusion_cfg.mode)
                    if fusion_debug_state is not None:
                        fusion_debug_state.record_shape("offset", offset_tile.shape, fused_offset.shape)
                    comparison_pairs["offset"] = (offset_entry.detach(), fused_offset.detach())
                    offset_entry = fused_offset
                if "pitch" in apply_heads and pitch_tile is not None and torch.is_tensor(out.get("pitch_logits")):
                    fused_pitch = fuse_tile_logits(pitch_tile, mask_tensor_device, mode=fusion_cfg.mode)
                    if fusion_debug_state is not None:
                        fusion_debug_state.record_shape("pitch", pitch_tile.shape, fused_pitch.shape)
                    comparison_pairs["pitch"] = (out["pitch_logits"].detach(), fused_pitch.detach())
                    out["pitch_logits"] = fused_pitch

            raw_heads: Dict[str, torch.Tensor] = {}
            onset_entry = out["onset_logits"] if "onset_logits" in out else out.get("onset")
            offset_entry = out["offset_logits"] if "offset_logits" in out else out.get("offset")
            if torch.is_tensor(onset_entry):
                raw_heads["onset"] = onset_entry
            if torch.is_tensor(offset_entry):
                raw_heads["offset"] = offset_entry
            adjusted: Dict[str, torch.Tensor] = {}
            if raw_heads:
                adjusted = apply_key_prior_to_logits(
                    raw_heads,
                    key_prior,
                    fps=key_prior_fps,
                    midi_low=key_prior_midi_low,
                    midi_high=key_prior_midi_high,
                )
            if "onset" in adjusted:
                onset_entry = adjusted["onset"]
            if "offset" in adjusted:
                offset_entry = adjusted["offset"]

            on_logits = onset_entry
            off_logits = offset_entry

            onset_probs = torch.sigmoid(on_logits)
            offset_probs = torch.sigmoid(off_logits)

            onset_roll = batch["onset_roll"][idx].float()
            offset_roll = batch["offset_roll"][idx].float()

            T_logits = on_logits.shape[1]
            if onset_roll.shape[1] != T_logits:
                onset_roll = pool_roll_BT(onset_roll, T_logits)
                offset_roll = pool_roll_BT(offset_roll, T_logits)

            onset_roll = align_pitch_dim(on_logits, onset_roll, "onset")
            offset_roll = align_pitch_dim(off_logits, offset_roll, "offset")

            onset_roll = (onset_roll > 0).float()
            offset_roll = (offset_roll > 0).float()

            if (
                fusion_enabled
                and comparison_enabled
                and fusion_debug_state is not None
                and comparison_batches_used < fusion_cfg.consistency_batches
            ):
                recorded = False
                if "onset" in comparison_pairs:
                    baseline, fused = comparison_pairs["onset"]
                    if fusion_debug_state.record_comparison(
                        "onset",
                        baseline_logits=baseline,
                        fused_logits=fused,
                        targets=onset_roll,
                        prob_threshold=preview_prob_threshold,
                        f1_fn=_binary_f1,
                    ):
                        recorded = True
                if "offset" in comparison_pairs:
                    baseline, fused = comparison_pairs["offset"]
                    if fusion_debug_state.record_comparison(
                        "offset",
                        baseline_logits=baseline,
                        fused_logits=fused,
                        targets=offset_roll,
                        prob_threshold=preview_prob_threshold,
                        f1_fn=_binary_f1,
                    ):
                        recorded = True
                if recorded:
                    comparison_batches_used += 1

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
        fusion_debug_state,
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
        mask_bool = build_threshold_mask(
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
            open_thr, hold_thr = resolve_decoder_gates(
                decoder_params,
                fallback_open=fallback_prob,
                default_hold=DECODER_DEFAULTS[name]["hold"],
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
        mask_bool = build_threshold_mask(
            probs,
            thr_val,
            mode=agg_mode,
            cap_count=cap_count,
            top_k=agg_top_k,
        )
        mask_float = mask_bool.float()
        pred_bin = mask_float
        if decoder_kind == "hysteresis":
            open_thr, hold_thr = resolve_decoder_gates(
                decoder_params,
                fallback_open=thr_val,
                default_hold=DECODER_DEFAULTS[name]["hold"],
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
    ap.add_argument(
        "--legacy-calibrate-thresholds",
        action="store_true",
        help="Route execution through scripts/calib/calibrate_thresholds_legacy.py",
    )
    argv = sys.argv[1:]
    args = ap.parse_args(argv)
    if args.legacy_calibrate_thresholds:
        from scripts.calib import calibrate_thresholds_legacy as legacy_calib

        legacy_argv = [arg for arg in argv if arg != "--legacy-calibrate-thresholds"]
        prev_argv = sys.argv
        try:
            sys.argv = [sys.argv[0], *legacy_argv]
            legacy_calib.main()
        finally:
            sys.argv = prev_argv
        return
    args.verbose = configure_verbosity(args.verbose)

    avlag_enabled = not args.no_avlag
    env_disable = str(os.environ.get("AVSYNC_DISABLE", "")).strip().lower()
    if env_disable in {"1", "true", "yes", "on"}:
        avlag_enabled = False

    cfg = dict(load_config("configs/config.yaml"))
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        cfg["model"] = model_cfg
    decoder_cfg_runtime = cfg.get("decoder", {}) or {}
    fusion_cfg = resolve_global_fusion_config(decoder_cfg_runtime)
    if fusion_cfg.enabled:
        head_desc = ", ".join(fusion_cfg.apply_to)
        print(
            f"[fusion] enabled (mode={fusion_cfg.mode}, heads={head_desc}, cushion={fusion_cfg.cushion_keys})",
            flush=True,
        )
        if fusion_cfg.consistency_check:
            print(
                f"[fusion] consistency check active for first {fusion_cfg.consistency_batches} batches",
                flush=True,
            )
    return_per_tile = bool(model_cfg.get("return_per_tile"))
    if fusion_cfg.needs_per_tile and not return_per_tile:
        model_cfg["return_per_tile"] = True
        return_per_tile = True
        print("[fusion] forcing per-tile logits for calibration", flush=True)
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
    preview_prob_threshold = metrics_cfg.get("prob_threshold_onset", metrics_cfg.get("prob_threshold", 0.5))
    try:
        preview_prob_threshold = float(preview_prob_threshold)
    except (TypeError, ValueError):
        preview_prob_threshold = 0.5
    decoder_kind = "hysteresis"
    decoder_params = resolve_decoder_from_config(metrics_cfg)
    decoder_settings_summary = format_decoder_settings(decoder_kind, decoder_params)
    print(f"[decoder-settings] {decoder_settings_summary}")
    print(f"[decoder] {decoder_notice_text(decoder_kind, decoder_params)}")
    key_prior_settings = resolve_key_prior_settings(decoder_cfg_runtime.get("key_prior"))
    if key_prior_settings.enabled:
        applied = ", ".join(key_prior_settings.apply_to)
        print(
            f"[decoder] key prior enabled (ref_head={key_prior_settings.ref_head}, apply_to={applied})",
            flush=True,
        )

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

    try:
        model_tiles = int(model_cfg.get("tiles", dataset_cfg.get("tiles", 3)))
    except Exception:
        model_tiles = int(dataset_cfg.get("tiles", 3) or 3)

    decode_fps = float(dataset_cfg.get("decode_fps", 0.0) or 0.0)
    hop_seconds = float(dataset_cfg.get("hop_seconds", 0.0) or 0.0)
    if hop_seconds <= 0.0:
        hop_seconds = 1.0 / decode_fps if decode_fps > 0 else 1.0
    if decode_fps <= 0.0 and hop_seconds > 0.0:
        decode_fps = 1.0 / hop_seconds
    if decode_fps <= 0.0:
        decode_fps = 30.0
    frame_targets_cfg = dataset_cfg.get("frame_targets", {}) or {}
    event_tolerance = float(frame_targets_cfg.get("tolerance", hop_seconds))
    midi_low_cfg = frame_targets_cfg.get("note_min")
    key_prior_midi_low = int(midi_low_cfg) if isinstance(midi_low_cfg, (int, float)) else 21

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
    base_dataset_for_reg = dataset
    if isinstance(base_dataset_for_reg, Subset):
        base_dataset_for_reg = base_dataset_for_reg.dataset
    registration_refiner = getattr(base_dataset_for_reg, "registration_refiner", None) if base_dataset_for_reg is not None else None
    reg_meta_cache = {}
    if isinstance(registration_refiner, RegistrationRefiner):
        reg_meta_cache = registration_refiner.export_geometry_cache()
    else:
        registration_refiner = None
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
        fusion_debug_state,
    ) = _collect(
        model,
        loader,
        target_clips,
        timeout_secs,
        key_prior=key_prior_settings,
        key_prior_fps=decode_fps,
        key_prior_midi_low=key_prior_midi_low,
        key_prior_midi_high=None,
        return_per_tile=return_per_tile,
        fusion_cfg=fusion_cfg,
        model_tiles=model_tiles,
        preview_prob_threshold=preview_prob_threshold,
        registration_refiner=registration_refiner,
        reg_meta_cache=reg_meta_cache,
    )
    data_elapsed = time.time() - t_data0
    stage_durations["data_pass"] = data_elapsed

    if fusion_cfg.enabled and fusion_debug_state is not None:
        for line in fusion_debug_state.summary_lines():
            print(line, flush=True)
        if fusion_cfg.consistency_check and not fusion_debug_state.comparison:
            print("[fusion] consistency check enabled but no comparison samples recorded", flush=True)

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
