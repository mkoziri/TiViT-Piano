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
    python -m tivit.pipelines.train_single --config tivit/configs/default.yaml
"""

from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TiViT-Piano training (new stack, no legacy)")
    ap.add_argument("--config", action="append", default=None, help="One or more config fragments to merge")
    ap.add_argument("--train-split", choices=["train", "val", "test"])
    ap.add_argument("--val-split", choices=["train", "val", "test"])
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="quiet")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _main() -> None:
    args = _parse_args()
    configs = args.config or [Path("tivit/configs/default.yaml")]
    train_single(
        configs=configs,
        verbose=args.verbose,
        train_split=args.train_split,
        val_split=args.val_split,
        max_clips=args.max_clips,
        frames=args.frames,
        seed=args.seed,
        deterministic=args.deterministic,
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    _main()


__all__ = ["train_single"]
