#!/usr/bin/env python3
"""Autopilot CLI helper for the new TiViT stack.

Purpose:
    - Provide a simple top-level entrypoint for running the autopilot pipeline.
    - Compose configs and forward CLI overrides into the orchestration call.
Key Functions/Classes:
    - main: parse CLI flags and invoke ``tivit.pipelines.autopilot.autopilot``.
CLI Arguments:
    - config: YAML fragments to merge.
    - verbose: logging verbosity (quiet|info|debug).
    - train-split / val-split / eval-split: dataset split overrides.
    - max-clips / frames / seed / deterministic / smoke: runtime overrides.
    - checkpoint: checkpoint hint (default: latest after training).
    - max-eval-batches: limit evaluation batches.
    - skip-eval / skip-export: toggle evaluation/export stages.
    - export-path: explicit TorchScript output path.
Usage:
    python run_autopilot.py --config tivit/configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tivit.pipelines.autopilot import autopilot


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TiViT autopilot (new stack)")
    ap.add_argument("--config", action="append", default=None, help="Config fragments to merge")
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="quiet")
    ap.add_argument("--train-split", choices=["train", "val", "test"])
    ap.add_argument("--val-split", choices=["train", "val", "test"])
    ap.add_argument("--eval-split", choices=["train", "val", "test"])
    ap.add_argument("--max-clips", type=int, help="Override dataset.max_clips for train/eval")
    ap.add_argument("--frames", type=int, help="Override dataset.frames")
    ap.add_argument("--seed", type=int, help="Seed override")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deterministic flag override (default: config/True)",
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny train/eval cycle for sanity checks")
    ap.add_argument("--checkpoint", help="Checkpoint hint to use instead of latest after training")
    ap.add_argument("--max-eval-batches", type=int, help="Limit evaluation batches")
    ap.add_argument("--skip-eval", action="store_true", help="Skip evaluation stage")
    ap.add_argument("--skip-export", action="store_true", help="Skip TorchScript export stage")
    ap.add_argument("--export-path", help="Explicit TorchScript output path")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    configs = args.config or [Path("tivit/configs/default.yaml")]
    autopilot(
        configs=configs,
        verbose=args.verbose,
        train_split=args.train_split,
        val_split=args.val_split,
        eval_split=args.eval_split,
        max_clips=args.max_clips,
        frames=args.frames,
        seed=args.seed,
        deterministic=args.deterministic,
        smoke=bool(args.smoke),
        checkpoint=args.checkpoint,
        max_eval_batches=args.max_eval_batches,
        skip_eval=bool(args.skip_eval),
        skip_export=bool(args.skip_export),
        export_path=args.export_path,
    )


if __name__ == "__main__":
    main()
