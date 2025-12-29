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
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configs = args.config or [Path("tivit/configs/default.yaml")]
    export_model(configs=configs, verbose=args.verbose)


if __name__ == "__main__":
    main()
