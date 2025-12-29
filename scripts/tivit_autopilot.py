#!/usr/bin/env python3
"""Autopilot CLI stub."""

from __future__ import annotations

import argparse
from pathlib import Path

from tivit.pipelines.autopilot import autopilot


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=None, help="Config fragments to merge")
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="quiet")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configs = args.config or [Path("tivit/configs/default.yaml")]
    autopilot(configs=configs, verbose=args.verbose)


if __name__ == "__main__":
    main()
