"""TiViT-Piano evaluation pipeline.

Purpose:
    - Run no-grad evaluation on a configured split using the new training stack.
    - Load checkpoints, decode logits into events, and report aggregated loss + event F1 metrics.
    - Provide a programmable entrypoint for CLI wrappers and autopilot.
Key Functions/Classes:
    - evaluate: compose configs, restore weights, and return averaged metrics.
CLI Arguments:
    - configs: YAML fragments to merge before evaluation.
    - verbose: logging verbosity (quiet|info|debug).
    - split: dataset split to evaluate (default: dataset.split_val/test fallback).
    - checkpoint: checkpoint path to load (default: latest under logging.checkpoint_dir).
    - max_batches: cap evaluation batches for smoke tests.
    - max_clips / frames / seed / deterministic / smoke: optional overrides applied to eval only.
Usage:
    python tivit/pipelines/tivit_eval.py --config tivit/configs/default.yaml
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tivit.data.loaders import make_dataloader
from tivit.decoder.decode import pool_roll_BT
from tivit.models import build_model
from tivit.metrics import event_f1, f1_from_counts
from tivit.postproc import build_decoder
from tivit.pipelines._common import find_checkpoint, prepare_run, resolve_eval_split, load_model_weights, setup_runtime
from tivit.train.loop import PerTileSupport, _prepare_targets
from tivit.losses.multitask_loss import MultitaskLoss
from tivit.utils.amp import autocast
from tivit.utils.logging import log_final_result, log_stage


def _apply_eval_overrides(
    cfg: Mapping[str, Any],
    *,
    split: str,
    max_clips: int | None,
    frames: int | None,
    smoke: bool,
) -> Mapping[str, Any]:
    cfg_copy = copy.deepcopy(cfg)
    dataset_cfg = cfg_copy.setdefault("dataset", {})  # type: ignore[assignment]
    if frames is not None:
        dataset_cfg["frames"] = int(frames)
    if max_clips is not None:
        dataset_cfg["max_clips"] = int(max_clips)
    dataset_cfg["split"] = split
    dataset_cfg.setdefault("split_val", split)
    if smoke:
        dataset_cfg["max_clips"] = min(int(dataset_cfg.get("max_clips", 2) or 2), 2)
        dataset_cfg["batch_size"] = min(int(dataset_cfg.get("batch_size", 1) or 1), 2)
        dataset_cfg["num_workers"] = 0
    return cfg_copy


def _resolve_hop_seconds(dataset_cfg: Mapping[str, Any]) -> float:
    hop_seconds = float(dataset_cfg.get("hop_seconds", 0.0) or 0.0)
    decode_fps = float(dataset_cfg.get("decode_fps", 0.0) or 0.0)
    if hop_seconds <= 0.0 and decode_fps > 0.0:
        hop_seconds = 1.0 / decode_fps
    if hop_seconds <= 0.0:
        hop_seconds = 1.0 / 30.0
    return hop_seconds


def _resolve_event_tolerance(dataset_cfg: Mapping[str, Any]) -> float:
    frame_cfg = dataset_cfg.get("frame_targets")
    tol = None
    if isinstance(frame_cfg, Mapping):
        tol = frame_cfg.get("tolerance")
    try:
        tol_val = float(tol) if tol is not None else 0.05
    except (TypeError, ValueError):
        tol_val = 0.05
    if tol_val < 0.0:
        tol_val = 0.05
    return tol_val


def evaluate(
    configs: Sequence[str | Path] | None = None,
    *,
    verbose: str | None = "quiet",
    split: str | None = None,
    checkpoint: str | Path | None = None,
    max_batches: int | None = None,
    max_clips: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    smoke: bool = False,
) -> Mapping[str, float]:
    cfg, _, _ = prepare_run(configs, stage_name="eval", default_log_file="eval.log", verbose=verbose)
    eval_split = resolve_eval_split(cfg, split_override=split)
    cfg_eval = _apply_eval_overrides(cfg, split=eval_split, max_clips=max_clips, frames=frames, smoke=smoke)
    seed_val, det_flag, device = setup_runtime(cfg_eval, seed=seed, deterministic=deterministic)

    log_stage("eval", f"starting evaluation on split={eval_split} deterministic={det_flag}")

    loader = make_dataloader(cfg_eval, eval_split, drop_last=False, seed=seed_val)
    model = build_model(cfg_eval).to(device)
    loss_fn = MultitaskLoss(cfg_eval)
    training_cfg = cfg_eval.get("training", {}) if isinstance(cfg_eval, Mapping) else {}
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    amp_enabled = bool(training_cfg.get("amp", False)) and torch.cuda.is_available()
    debug_dummy_labels = bool(training_cfg.get("debug_dummy_labels", False))
    per_tile_support = PerTileSupport(cfg_eval, getattr(loader, "dataset", None), phase="eval")
    decoder = build_decoder(cfg_eval)
    dataset_cfg = cfg_eval.get("dataset", {}) if isinstance(cfg_eval, Mapping) else {}
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    hop_seconds = _resolve_hop_seconds(dataset_cfg)
    event_tolerance = _resolve_event_tolerance(dataset_cfg)

    ckpt_path = find_checkpoint(cfg_eval, checkpoint)
    if ckpt_path:
        epoch_loaded = load_model_weights(model, ckpt_path, device)
        epoch_str = f"epoch {epoch_loaded}" if epoch_loaded is not None else "unknown epoch"
        log_stage("eval", f"loaded checkpoint {ckpt_path} ({epoch_str})")
    else:
        log_stage("eval", "no checkpoint found; evaluating randomly initialized weights")

    total_loss = 0.0
    total_batches = 0
    parts_sum: dict[str, float] = {}
    event_counts = {
        "onset": {"tp": 0, "fp": 0, "fn": 0, "clips": 0},
        "offset": {"tp": 0, "fp": 0, "fn": 0, "clips": 0},
    }

    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            video = batch.get("video")
            if not torch.is_tensor(video):
                raise ValueError("Batch is missing tensor key 'video'")
            x = video.to(device=device, non_blocking=True)
            request_per_tile = per_tile_support.request_per_tile_outputs
            with autocast(device, enabled=amp_enabled):
                outputs = model(x, return_per_tile=request_per_tile)
                per_tile_ctx = per_tile_support.build_context(outputs, batch)
                targets = _prepare_targets(outputs, batch, device, debug_dummy_labels=debug_dummy_labels)
                loss, parts = loss_fn(outputs, targets, update_state=False, per_tile=per_tile_ctx)

            loss_val = float(loss.detach().cpu().item())
            total_loss += loss_val
            total_batches += 1
            for key, value in parts.items():
                try:
                    parts_sum[key] = parts_sum.get(key, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue

            logits_map: dict[str, torch.Tensor] = {}
            for head in ("onset", "offset"):
                tensor = outputs.get(f"{head}_logits")
                if torch.is_tensor(tensor) and tensor.dim() == 3:
                    logits_map[f"{head}_logits"] = tensor
            if not logits_map:
                continue

            decoded = decoder(logits_map)
            for head, pred_mask in decoded.items():
                target_roll = targets.get(head)
                if not torch.is_tensor(target_roll):
                    continue
                pred_tensor = pred_mask
                target_tensor = target_roll
                if pred_tensor.dim() == 2:
                    pred_tensor = pred_tensor.unsqueeze(0)
                if target_tensor.dim() == 2:
                    target_tensor = target_tensor.unsqueeze(0)
                if pred_tensor.dim() != 3 or target_tensor.dim() != 3:
                    continue
                if target_tensor.shape[1] != pred_tensor.shape[1]:
                    target_tensor = pool_roll_BT(target_tensor, pred_tensor.shape[1])
                if target_tensor.shape[2] != pred_tensor.shape[2]:
                    continue
                target_mask = target_tensor > 0.5
                result = event_f1(pred_tensor, target_mask, hop_seconds=hop_seconds, tolerance=event_tolerance)
                counts = event_counts.get(head)
                if counts is None:
                    continue
                counts["tp"] += int(result.true_positives)
                counts["fp"] += int(result.false_positives)
                counts["fn"] += int(result.false_negatives)
                counts["clips"] += int(result.clips_evaluated)

    metrics: dict[str, float] = {}
    if total_batches > 0:
        metrics["loss"] = total_loss / total_batches
        for key, value in parts_sum.items():
            metrics[key] = value / total_batches

    onset_counts = event_counts["onset"]
    offset_counts = event_counts["offset"]
    onset_summary = f1_from_counts(onset_counts["tp"], onset_counts["fp"], onset_counts["fn"])
    offset_summary = f1_from_counts(offset_counts["tp"], offset_counts["fp"], offset_counts["fn"])
    if onset_counts["clips"] > 0 or offset_counts["clips"] > 0:
        metrics["onset_event_f1"] = onset_summary.f1
        metrics["offset_event_f1"] = offset_summary.f1
        metrics["ev_f1_mean"] = 0.5 * (onset_summary.f1 + offset_summary.f1)

    log_stage("eval", f"evaluation finished on split={eval_split} metrics={metrics}")
    log_final_result("eval", f"metrics={metrics}")
    return metrics


__all__ = ["evaluate"]
