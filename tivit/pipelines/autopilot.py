"""TiViT-Piano autopilot orchestration.

Purpose:
    - Run a training → evaluation → export cycle using the new implementation.
    - Provide a one-call automation entrypoint without relying on legacy drivers.
    - Surface checkpoint, metrics, and export paths for downstream tooling.
Key Functions/Classes:
    - autopilot: orchestrate train_single, evaluate, and export_model.
CLI Arguments:
    - configs: YAML fragments to merge.
    - verbose: logging verbosity (quiet|info|debug).
    - train_split / val_split / eval_split: dataset split overrides.
    - max_clips / frames / seed / deterministic / smoke: shared runtime overrides.
    - checkpoint: optional checkpoint hint (default: latest after training).
    - max_eval_batches: limit eval batches; skip_eval / skip_export toggles.
    - export_path: explicit TorchScript destination.
Usage:
    python tivit/pipelines/tivit_autopilot.py --config tivit/configs/default.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from tivit.pipelines._common import find_checkpoint, prepare_run
from tivit.pipelines.evaluate import evaluate
from tivit.pipelines.export import export_model
from tivit.pipelines.train_single import train_single
from tivit.utils.logging import configure_logging, log_final_result, log_stage


def autopilot(
    configs: Sequence[str | Path] | None = None,
    *,
    verbose: str | None = "quiet",
    train_split: str | None = None,
    val_split: str | None = None,
    eval_split: str | None = None,
    max_clips: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    smoke: bool = False,
    checkpoint: str | Path | None = None,
    max_eval_batches: int | None = None,
    skip_eval: bool = False,
    skip_export: bool = False,
    export_path: str | Path | None = None,
) -> Mapping[str, object]:
    cfg, log_dir, _ = prepare_run(configs, stage_name="autopilot", default_log_file="autopilot.log", verbose=verbose)
    log_stage("autopilot", "starting autopilot pipeline (train → eval → export)")

    autopilot_cfg = cfg.get("autopilot", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(autopilot_cfg, Mapping):
        autopilot_cfg = {}
    enable_training = bool(autopilot_cfg.get("enable_training", True))
    if enable_training:
        train_single(
            configs=configs,
            verbose=verbose,
            train_split=train_split,
            val_split=val_split,
            max_clips=max_clips,
            frames=frames,
            seed=seed,
            deterministic=deterministic,
            smoke=smoke,
        )
    else:
        log_stage("autopilot", "skipping training stage (autopilot.enable_training=false)")

    resolved_ckpt = find_checkpoint(cfg, checkpoint)
    metrics: Mapping[str, float] | None = None
    if not skip_eval:
        metrics = evaluate(
            configs,
            verbose=verbose,
            split=eval_split,
            checkpoint=resolved_ckpt,
            max_batches=max_eval_batches,
            max_clips=max_clips,
            frames=frames,
            seed=seed,
            deterministic=deterministic,
            smoke=smoke,
        )

    export_location: Path | None = None
    if not skip_export:
        export_location = export_model(
            configs,
            verbose=verbose,
            checkpoint=resolved_ckpt,
            output_path=export_path,
            seed=seed,
            deterministic=deterministic,
        )

    configure_logging(verbose, log_dir=log_dir, log_file="autopilot.log", stage_only_console=True)
    summary_parts = []
    if resolved_ckpt is not None:
        summary_parts.append(f"checkpoint={resolved_ckpt}")
    if metrics is not None:
        summary_parts.append(f"metrics={metrics}")
    if export_location is not None:
        summary_parts.append(f"export={export_location}")
    summary = "; ".join(summary_parts) if summary_parts else "autopilot run completed"
    log_stage("autopilot", summary)
    log_final_result("autopilot", summary)

    return {"checkpoint": resolved_ckpt, "metrics": metrics, "export": export_location}


__all__ = ["autopilot"]
