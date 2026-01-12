#!/usr/bin/env python3
"""Evaluation CLI stub."""

from __future__ import annotations

import argparse
from pathlib import Path

from tivit.pipelines.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None, help="Config fragments to merge")
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="quiet")
    ap.add_argument("--split", help="Dataset split to evaluate (defaults to dataset.split_val/test)")
    ap.add_argument("--checkpoint", help="Checkpoint path to load (default: latest in checkpoint_dir)")
    ap.add_argument("--max-batches", dest="max_batches", type=int, help="Limit evaluation batches")
    ap.add_argument("--max-clips", type=int, help="Override dataset.max_clips for eval")
    ap.add_argument("--frames", type=int, help="Override dataset.frames for eval")
    ap.add_argument("--seed", type=int, help="Seed override")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deterministic flag override (default: config/True)",
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny eval run with small batch/clips")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configs = args.config or [Path("tivit/configs/default.yaml")]
    evaluate(
        configs=configs,
        verbose=args.verbose,
        split=args.split,
        checkpoint=args.checkpoint,
        max_batches=args.max_batches,
        max_clips=args.max_clips,
        frames=args.frames,
        seed=args.seed,
        deterministic=args.deterministic,
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()
