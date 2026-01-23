"""Threshold sweep calibration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from tivit.calibration.thresholds import (
    build_sweep_values,
    load_threshold_priors,
    resolve_hold,
    resolve_threshold_center,
    summarize_sweep,
)
from tivit.data.loaders import make_dataloader
from tivit.decoder.decode import DECODER_DEFAULTS, decode_hysteresis, pool_roll_BT, resolve_decoder_from_config
from tivit.metrics import event_f1
from tivit.models import build_model
from tivit.pipelines._common import find_checkpoint, load_model_weights, setup_runtime
from tivit.pipelines.evaluate import _apply_eval_overrides, _resolve_event_tolerance, _resolve_hop_seconds
from tivit.postproc.hand_gate_runtime import apply_hand_gate_from_config
from tivit.postproc.key_prior_runtime import apply_key_prior_from_config
from tivit.train.loop import _prepare_targets
from tivit.utils.amp import autocast
from tivit.utils.logging import log_stage


def resolve_calibration_split(cfg: Mapping[str, Any]) -> str:
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    split = dataset_cfg.get("split_val") or dataset_cfg.get("split_train") or dataset_cfg.get("split") or "train"
    return str(split)




def run_threshold_sweep(
    cfg: Mapping[str, Any],
    *,
    split: str,
    checkpoint: str | Path | None,
    max_batches: int | None,
    max_clips: int | None,
    frames: int | None,
    seed: int | None,
    deterministic: bool | None,
    smoke: bool,
) -> Mapping[str, Any]:
    cfg_eval = _apply_eval_overrides(cfg, split=split, max_clips=max_clips, frames=frames, smoke=smoke)
    seed_val, det_flag, device = setup_runtime(cfg_eval, seed=seed, deterministic=deterministic)
    log_stage("calibration", f"starting sweep on split={split} deterministic={det_flag}")

    calibration_cfg = cfg_eval.get("calibration", {}) if isinstance(cfg_eval, Mapping) else {}
    if not isinstance(calibration_cfg, Mapping):
        calibration_cfg = {}
    sweep_cfg = calibration_cfg.get("sweep", {}) if isinstance(calibration_cfg, Mapping) else {}
    if not isinstance(sweep_cfg, Mapping):
        sweep_cfg = {}
    delta = float(sweep_cfg.get("delta", 0.05) or 0.05)
    steps = int(sweep_cfg.get("steps", 5) or 5)
    min_prob = float(sweep_cfg.get("min_prob", 0.02) or 0.02)
    max_prob = float(sweep_cfg.get("max_prob", 0.98) or 0.98)

    priors_path = calibration_cfg.get("threshold_priors_path")
    recommendations = load_threshold_priors(priors_path)
    centers = {}
    sources = {}
    for head in ("onset", "offset"):
        center, source = resolve_threshold_center(cfg_eval, recommendations, head)
        centers[head] = max(min(center, max_prob), min_prob)
        sources[head] = source

    threshold_values: dict[str, list[float]] = {}
    threshold_holds: dict[str, list[float]] = {}
    threshold_counts: dict[str, list[dict[str, int]]] = {}

    training_cfg = cfg_eval.get("training", {}) if isinstance(cfg_eval, Mapping) else {}
    metrics_cfg = training_cfg.get("metrics", {}) if isinstance(training_cfg, Mapping) else {}
    decoder_templates = resolve_decoder_from_config(metrics_cfg if isinstance(metrics_cfg, Mapping) else {})

    for head in ("onset", "offset"):
        values = build_sweep_values(centers[head], delta, steps, min_prob=min_prob, max_prob=max_prob)
        template = decoder_templates.get(head, {})
        holds = [resolve_hold(val, template, head) for val in values]
        threshold_values[head] = values
        threshold_holds[head] = holds
        threshold_counts[head] = [{"tp": 0, "fp": 0, "fn": 0, "clips": 0} for _ in values]

    loader = make_dataloader(cfg_eval, split, drop_last=False, seed=seed_val)
    model = build_model(cfg_eval).to(device)
    training_cfg = cfg_eval.get("training", {}) if isinstance(cfg_eval, Mapping) else {}
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    amp_enabled = bool(training_cfg.get("amp", False)) and torch.cuda.is_available()
    debug_dummy_labels = bool(training_cfg.get("debug_dummy_labels", False))

    ckpt_path = find_checkpoint(cfg_eval, checkpoint)
    if ckpt_path:
        epoch_loaded = load_model_weights(model, ckpt_path, device)
        epoch_str = f"epoch {epoch_loaded}" if epoch_loaded is not None else "unknown epoch"
        log_stage("calibration", f"loaded checkpoint {ckpt_path} ({epoch_str})")
    else:
        log_stage("calibration", "no checkpoint found; calibrating randomly initialized weights")

    dataset_cfg = cfg_eval.get("dataset", {}) if isinstance(cfg_eval, Mapping) else {}
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    hop_seconds = _resolve_hop_seconds(dataset_cfg)
    event_tolerance = _resolve_event_tolerance(dataset_cfg)

    model.eval()
    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            video = batch.get("video")
            if not torch.is_tensor(video):
                raise ValueError("Batch is missing tensor key 'video'")
            x = video.to(device=device, non_blocking=True)
            with autocast(device, enabled=amp_enabled):
                outputs = model(x, return_per_tile=False)
                targets = _prepare_targets(outputs, batch, device, debug_dummy_labels=debug_dummy_labels)

            logits_map: dict[str, torch.Tensor] = {}
            for head in ("onset", "offset"):
                tensor = outputs.get(f"{head}_logits")
                if torch.is_tensor(tensor):
                    logits_map[head] = tensor
            if not logits_map:
                continue

            adjusted = apply_key_prior_from_config(logits_map, cfg_eval)
            if adjusted:
                logits_map.update(adjusted)
            probs = {head: torch.sigmoid(tensor) for head, tensor in logits_map.items()}
            gated = apply_hand_gate_from_config(probs, outputs, cfg_eval, input_is_logits=True)
            if gated:
                probs.update(gated)

            for head in ("onset", "offset"):
                pred = probs.get(head)
                target_roll = targets.get(head)
                if pred is None or not torch.is_tensor(target_roll):
                    continue
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)
                if target_roll.dim() == 2:
                    target_roll = target_roll.unsqueeze(0)
                if pred.dim() != 3 or target_roll.dim() != 3:
                    continue
                if target_roll.shape[1] != pred.shape[1]:
                    target_roll = pool_roll_BT(target_roll, pred.shape[1])
                if target_roll.shape[2] != pred.shape[2]:
                    continue

                pred_cpu = pred.detach().float().cpu()
                target_mask = (target_roll > 0.5).detach().cpu()

                template = decoder_templates.get(head, {})
                min_on = template.get("min_on", DECODER_DEFAULTS[head]["min_on"])
                min_off = template.get("min_off", DECODER_DEFAULTS[head]["min_off"])
                merge_gap = template.get("merge_gap", DECODER_DEFAULTS[head]["merge_gap"])
                median = template.get("median", DECODER_DEFAULTS[head]["median"])

                for sweep_idx, (thr, hold) in enumerate(zip(threshold_values[head], threshold_holds[head])):
                    decoded = decode_hysteresis(
                        pred_cpu,
                        open_thr=thr,
                        hold_thr=hold,
                        min_on=min_on,
                        min_off=min_off,
                        merge_gap=merge_gap,
                        median=median,
                    )
                    result = event_f1(decoded, target_mask, hop_seconds=hop_seconds, tolerance=event_tolerance)
                    counts = threshold_counts[head][sweep_idx]
                    counts["tp"] += result.true_positives
                    counts["fp"] += result.false_positives
                    counts["fn"] += result.false_negatives
                    counts["clips"] += result.clips_evaluated

    summary: dict[str, Any] = {"method": "threshold_sweep", "split": split}
    sweep_summary: dict[str, Any] = {}
    best_thresholds: dict[str, float] = {}
    decoder_payload: dict[str, Any] = {}
    metrics_payload: dict[str, float] = {}

    for head in ("onset", "offset"):
        results, best_idx = summarize_sweep(
            threshold_values[head],
            threshold_holds[head],
            threshold_counts[head],
            center=centers[head],
        )
        sweep_summary[head] = {
            "center": centers[head],
            "source": sources[head],
            "values": threshold_values[head],
            "results": results,
        }
        best = results[best_idx]
        best_thresholds[head] = best["threshold"]
        metrics_payload[f"{head}_event_f1"] = float(best["f1"])
        decoder_payload[head] = {
            "open": best["threshold"],
            "hold": best["hold"],
            "min_on": int(decoder_templates.get(head, {}).get("min_on", DECODER_DEFAULTS[head]["min_on"])),
            "min_off": int(decoder_templates.get(head, {}).get("min_off", DECODER_DEFAULTS[head]["min_off"])),
            "merge_gap": int(decoder_templates.get(head, {}).get("merge_gap", DECODER_DEFAULTS[head]["merge_gap"])),
            "median": int(decoder_templates.get(head, {}).get("median", DECODER_DEFAULTS[head]["median"])),
        }

    if "onset_event_f1" in metrics_payload and "offset_event_f1" in metrics_payload:
        metrics_payload["ev_f1_mean"] = 0.5 * (
            metrics_payload["onset_event_f1"] + metrics_payload["offset_event_f1"]
        )

    summary["thresholds"] = best_thresholds
    summary["decoder"] = decoder_payload
    summary["metrics"] = metrics_payload
    summary["sweep"] = {
        "delta": delta,
        "steps": steps,
        "min_prob": min_prob,
        "max_prob": max_prob,
        "per_head": sweep_summary,
    }
    if priors_path is not None:
        summary["threshold_priors_path"] = str(priors_path)

    return summary


__all__ = ["resolve_calibration_split", "run_threshold_sweep"]
