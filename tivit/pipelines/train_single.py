"""TiViT-Piano single-run training pipeline.

Purpose:
    - Compose configs, configure logging, and delegate to the new training loop.
    - Expose CLI-friendly overrides for splits, frames, seeds, and smoke tests.
    - Persist resolved configs/commands for reproducibility.
Key Functions/Classes:
    - train_single: wrapper around ``tivit.train.loop.run_training`` for one experiment.
CLI Arguments:
    - configs: YAML fragments to merge before training.
    - verbose: logging verbosity (quiet|info|debug).
    - train_split / val_split: dataset split overrides for train/validation loaders.
    - max_clips: cap clips per split for quick iterations.
    - frames: override clip length in frames.
    - seed / deterministic: runtime overrides for determinism.
    - smoke: 1-epoch tiny run for fast sanity checks.
Usage:
    python scripts/tivit_train.py --config tivit/configs/default.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from tivit.pipelines._common import prepare_run
from tivit.train.loop import run_training
from tivit.utils.logging import log_final_result, log_stage


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
    prepare_run(configs, stage_name="train", default_log_file="train.log", verbose=verbose)

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
