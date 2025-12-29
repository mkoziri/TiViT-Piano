"""High-level training entrypoint used by CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence

from tivit.core.config import load_experiment_config, write_run_artifacts
from tivit.train.loop import run_training
from tivit.utils.logging import configure_logging, log_final_result, log_stage


def train_single(
    configs: Sequence[str | Path] | None = None,
    *,
    verbose: str | None = "quiet",
    train_split: str | None = None,
    val_split: str | None = None,
    max_clips: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    smoke: bool = False,
) -> None:
    cfg = load_experiment_config(configs)
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, Mapping) else {}
    log_dir = log_cfg.get("log_dir", "logs")
    log_file = log_cfg.get("train_log", "train.log")
    configure_logging(verbose, log_dir=log_dir, log_file=log_file, stage_only_console=True)
    write_run_artifacts(cfg, log_dir=log_dir, command=sys.argv, configs=configs)

    log_stage("train", "building dataloaders and model")
    run_training(
        configs,
        verbose=verbose,
        train_split=train_split,
        val_split=val_split,
        max_clips=max_clips,
        frames=frames,
        seed=seed,
        deterministic=deterministic,
        smoke=smoke,
    )
    log_final_result("train", "training completed")


__all__ = ["train_single"]
