#!/usr/bin/env python3
"""Thin CLI that routes to the new training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from tivit.pipelines.train_single import train_single


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
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


def main() -> None:
    args = parse_args()
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
    main()
