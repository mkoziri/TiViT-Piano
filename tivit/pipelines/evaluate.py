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

from tivit.calibration.io import read_calibration
from tivit.data.loaders import make_dataloader
from tivit.decoder.decode import pool_roll_BT
from tivit.models import build_model
from tivit.metrics import event_f1, f1_from_counts
from tivit.metrics.patk_metrics import frame_counts, note_event_counts, onset_event_counts
from tivit.postproc.patk_decode import (
    PatkDecodeConfig,
    build_notes_from_peaks,
    build_notes_from_rolls,
    clamp_probs,
    compute_target_length,
    extract_onset_peaks,
    resolve_note_range,
    resample_probs_btP,
    resample_roll_btP,
    smooth_time_probs,
)
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


def _resolve_eval_mode(metrics_cfg: Mapping[str, Any]) -> str:
    mode = "native"
    if isinstance(metrics_cfg, Mapping):
        mode = str(metrics_cfg.get("eval_mode", mode) or mode)
    mode = mode.strip().lower()
    if mode not in {"native", "patk", "both"}:
        mode = "native"
    return mode


def _resolve_calibration_path(cfg: Mapping[str, Any]) -> Path | None:
    calibration_cfg = cfg.get("calibration", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(calibration_cfg, Mapping):
        return None
    raw_path = calibration_cfg.get("output_path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path if path.exists() else None
    candidates = []
    if path.exists():
        candidates.append(path)
    log_dir = Path(cfg.get("logging", {}).get("log_dir", "logs")).expanduser()
    candidates.append(log_dir / path)
    repo_root = Path(__file__).resolve().parents[2]
    if path.parent == Path("."):
        candidates.append(repo_root / "tivit" / "logs" / path.name)
    else:
        candidates.append(repo_root / "tivit" / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _apply_calibration_decoder_overrides(
    metrics_cfg: dict[str, Any],
    calibration: Mapping[str, Any],
) -> bool:
    decoder_payload = calibration.get("decoder")
    if not isinstance(decoder_payload, Mapping):
        return False
    decoder_cfg = metrics_cfg.get("decoder")
    if not isinstance(decoder_cfg, dict):
        decoder_cfg = {}
        metrics_cfg["decoder"] = decoder_cfg
    updated = False
    for head in ("onset", "offset"):
        entry = decoder_payload.get(head)
        if not isinstance(entry, Mapping):
            continue
        head_cfg = decoder_cfg.get(head)
        if not isinstance(head_cfg, dict):
            head_cfg = {}
            decoder_cfg[head] = head_cfg
        for key in ("open", "hold", "min_on", "min_off", "merge_gap", "median"):
            if key in entry and entry[key] is not None:
                head_cfg[key] = entry[key]
                updated = True
    return updated


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
    metrics_cfg = training_cfg.get("metrics", {}) if isinstance(training_cfg, Mapping) else {}
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = dict(metrics_cfg) if isinstance(metrics_cfg, Mapping) else {}
        if isinstance(training_cfg, dict):
            training_cfg["metrics"] = metrics_cfg
    calib_path = _resolve_calibration_path(cfg_eval)
    if calib_path is not None:
        try:
            calibration_payload = read_calibration(calib_path)
        except Exception as exc:
            log_stage("eval", f"failed to load calibration {calib_path}: {exc}")
        else:
            if _apply_calibration_decoder_overrides(metrics_cfg, calibration_payload):
                log_stage("eval", f"using calibration overrides from {calib_path}")
    decoder = build_decoder(cfg_eval)
    eval_mode = _resolve_eval_mode(metrics_cfg)
    patk_cfg = PatkDecodeConfig.from_config(metrics_cfg, cfg_eval.get("dataset", {}) if isinstance(cfg_eval, Mapping) else {})
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
    debug_logged = False
    event_counts = None
    if eval_mode in {"native", "both"}:
        event_counts = {
            "onset": {"tp": 0, "fp": 0, "fn": 0, "clips": 0},
            "offset": {"tp": 0, "fp": 0, "fn": 0, "clips": 0},
        }

    patk_counts = None
    if eval_mode in {"patk", "both"}:
        patk_counts = {
            "frame": {"tp": 0, "fp": 0, "fn": 0},
            "onset": {"tp": 0, "fp": 0, "fn": 0},
            "note": {"tp": 0, "fp": 0, "fn": 0},
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

            if not debug_logged:
                # TEMP_DEBUG: log sigmoid stats for onset/offset to verify logit scale.
                onset_logits = outputs.get("onset_logits")
                offset_logits = outputs.get("offset_logits")
                if torch.is_tensor(onset_logits) and torch.is_tensor(offset_logits):
                    onset_probs = torch.sigmoid(onset_logits.detach())
                    offset_probs = torch.sigmoid(offset_logits.detach())
                    log_stage(
                        "eval",
                        (
                            "TEMP_DEBUG logits: onset(mean={:.6f} max={:.6f}) "
                            "offset(mean={:.6f} max={:.6f})"
                        ).format(
                            float(onset_probs.mean().item()),
                            float(onset_probs.max().item()),
                            float(offset_probs.mean().item()),
                            float(offset_probs.max().item()),
                        ),
                    )
                    debug_logged = True


            if eval_mode in {"native", "both"} and event_counts is not None:
                logits_map: dict[str, torch.Tensor] = {}
                for head in ("onset", "offset"):
                    tensor = outputs.get(f"{head}_logits")
                    if torch.is_tensor(tensor) and tensor.dim() == 3:
                        logits_map[f"{head}_logits"] = tensor
                if logits_map:
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

            if eval_mode in {"patk", "both"} and patk_counts is not None:
                onset_logits = outputs.get("onset_logits")
                offset_logits = outputs.get("offset_logits")
                pitch_logits = outputs.get("pitch_logits")
                if not (torch.is_tensor(onset_logits) and torch.is_tensor(offset_logits) and torch.is_tensor(pitch_logits)):
                    continue
                if onset_logits.dim() != 3 or offset_logits.dim() != 3 or pitch_logits.dim() != 3:
                    continue

                note_min, _note_max = resolve_note_range(dataset_cfg)
                t_in = int(onset_logits.shape[1])
                t_patk = compute_target_length(t_in, hop_seconds, fps=patk_cfg.fps)
                hop_patk = 1.0 / max(patk_cfg.fps, 1e-6)

                onset_probs = torch.sigmoid(onset_logits).detach()
                frame_probs = torch.sigmoid(pitch_logits).detach()
                onset_probs = resample_probs_btP(onset_probs, t_patk)
                frame_probs = resample_probs_btP(frame_probs, t_patk)

                onset_probs = smooth_time_probs(onset_probs, sigma=patk_cfg.onset_sigma, radius=patk_cfg.onset_radius)
                frame_probs = smooth_time_probs(frame_probs, sigma=patk_cfg.frame_sigma, radius=patk_cfg.frame_radius)

                onset_mask = clamp_probs(onset_probs, patk_cfg.threshold)
                frame_mask = clamp_probs(frame_probs, patk_cfg.threshold)

                onset_peaks = extract_onset_peaks(onset_probs, onset_mask)
                pred_notes = build_notes_from_peaks(
                    onset_peaks,
                    frame_mask,
                    hop_seconds=hop_patk,
                    note_min=note_min,
                    ignore_tail=patk_cfg.ignore_tail,
                )

                target_onset = targets.get("onset")
                target_pitch = targets.get("pitch")
                if not (torch.is_tensor(target_onset) and torch.is_tensor(target_pitch)):
                    continue
                target_onset = resample_roll_btP(target_onset, t_patk)
                target_pitch = resample_roll_btP(target_pitch, t_patk)
                ref_notes = build_notes_from_rolls(
                    target_onset,
                    target_pitch,
                    hop_seconds=hop_patk,
                    note_min=note_min,
                )

                pred_frame = frame_mask.detach().cpu()
                target_frame = (target_pitch > 0.5).detach().cpu()
                tp, fp, fn = frame_counts(pred_frame, target_frame)
                patk_counts["frame"]["tp"] += tp
                patk_counts["frame"]["fp"] += fp
                patk_counts["frame"]["fn"] += fn

                for notes_pred, notes_ref in zip(pred_notes, ref_notes):
                    tp, fp, fn = onset_event_counts(
                        notes_pred,
                        notes_ref,
                        onset_tolerance=patk_cfg.onset_tolerance,
                    )
                    patk_counts["onset"]["tp"] += tp
                    patk_counts["onset"]["fp"] += fp
                    patk_counts["onset"]["fn"] += fn

                    tp, fp, fn = note_event_counts(
                        notes_pred,
                        notes_ref,
                        onset_tolerance=patk_cfg.onset_tolerance,
                        offset_ratio=patk_cfg.offset_ratio,
                        offset_min_tolerance=patk_cfg.offset_min_tolerance,
                    )
                    patk_counts["note"]["tp"] += tp
                    patk_counts["note"]["fp"] += fp
                    patk_counts["note"]["fn"] += fn

    metrics: dict[str, float] = {}
    if total_batches > 0:
        metrics["loss"] = total_loss / total_batches
        for key, value in parts_sum.items():
            metrics[key] = value / total_batches

    if event_counts is not None:
        onset_counts = event_counts["onset"]
        offset_counts = event_counts["offset"]
        onset_summary = f1_from_counts(onset_counts["tp"], onset_counts["fp"], onset_counts["fn"])
        offset_summary = f1_from_counts(offset_counts["tp"], offset_counts["fp"], offset_counts["fn"])
        if onset_counts["clips"] > 0 or offset_counts["clips"] > 0:
            metrics["onset_event_f1"] = onset_summary.f1
            metrics["offset_event_f1"] = offset_summary.f1
            metrics["ev_f1_mean"] = 0.5 * (onset_summary.f1 + offset_summary.f1)

    if patk_counts is not None:
        frame_summary = f1_from_counts(
            patk_counts["frame"]["tp"], patk_counts["frame"]["fp"], patk_counts["frame"]["fn"]
        )
        onset_summary = f1_from_counts(
            patk_counts["onset"]["tp"], patk_counts["onset"]["fp"], patk_counts["onset"]["fn"]
        )
        note_summary = f1_from_counts(
            patk_counts["note"]["tp"], patk_counts["note"]["fp"], patk_counts["note"]["fn"]
        )
        metrics["patk_frame_f1"] = frame_summary.f1
        metrics["patk_frame_precision"] = frame_summary.precision
        metrics["patk_frame_recall"] = frame_summary.recall
        metrics["patk_onset_f1"] = onset_summary.f1
        metrics["patk_onset_precision"] = onset_summary.precision
        metrics["patk_onset_recall"] = onset_summary.recall
        metrics["patk_note_f1"] = note_summary.f1
        metrics["patk_note_precision"] = note_summary.precision
        metrics["patk_note_recall"] = note_summary.recall

    log_stage("eval", f"evaluation finished on split={eval_split} metrics={metrics}")
    log_final_result("eval", f"metrics={metrics}")
    return metrics


__all__ = ["evaluate"]
