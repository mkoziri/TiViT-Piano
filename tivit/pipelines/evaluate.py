"""TiViT-Piano evaluation pipeline.

Purpose:
    - Run no-grad evaluation on a configured split using the new training stack.
    - Load checkpoints and report aggregated loss/part metrics without legacy glue.
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
from torch.cuda.amp import autocast

from tivit.data.loaders import make_dataloader
from tivit.models import build_model
from tivit.pipelines._common import find_checkpoint, prepare_run, resolve_eval_split, load_model_weights, setup_runtime
from tivit.train.eval_loop import run_evaluation
from tivit.train.loop import PerTileSupport, _prepare_targets
from tivit.losses.multitask_loss import MultitaskLoss
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

    ckpt_path = find_checkpoint(cfg_eval, checkpoint)
    if ckpt_path:
        epoch_loaded = load_model_weights(model, ckpt_path, device)
        epoch_str = f"epoch {epoch_loaded}" if epoch_loaded is not None else "unknown epoch"
        log_stage("eval", f"loaded checkpoint {ckpt_path} ({epoch_str})")
    else:
        log_stage("eval", "no checkpoint found; evaluating randomly initialized weights")

    def _step(batch: Mapping[str, object]):
        video = batch.get("video")
        if not torch.is_tensor(video):
            raise ValueError("Batch is missing tensor key 'video'")
        x = video.to(device=device, non_blocking=True)
        request_per_tile = per_tile_support.request_per_tile_outputs
        with autocast(enabled=amp_enabled):
            outputs = model(x, return_per_tile=request_per_tile)
            per_tile_ctx = per_tile_support.build_context(outputs, batch)
            targets = _prepare_targets(outputs, batch, device, debug_dummy_labels=debug_dummy_labels)
            loss, parts = loss_fn(outputs, targets, update_state=False, per_tile=per_tile_ctx)
        return loss, parts

    metrics = run_evaluation(_step, loader, max_batches=max_batches)
    log_stage("eval", f"evaluation finished on split={eval_split} metrics={metrics}")
    log_final_result("eval", f"metrics={metrics}")
    return metrics


__all__ = ["evaluate"]
