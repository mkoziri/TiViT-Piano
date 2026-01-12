#!/usr/bin/env python3
"""Export CLI stub."""

from __future__ import annotations

import argparse
from pathlib import Path

from tivit.pipelines.export import export_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None, help="Config fragments to merge")
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="quiet")
    ap.add_argument("--checkpoint", help="Checkpoint path to export (default: latest in checkpoint_dir)")
    ap.add_argument("--output", dest="output_path", help="TorchScript output path")
    ap.add_argument("--seed", type=int, help="Seed override before export")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deterministic flag override (default: config/True)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configs = args.config or [Path("tivit/configs/default.yaml")]
    export_model(
        configs=configs,
        verbose=args.verbose,
        checkpoint=args.checkpoint,
        output_path=args.output_path,
        seed=args.seed,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
