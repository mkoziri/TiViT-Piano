#!/usr/bin/env python3
"""
Purpose:
    Lightweight OMAPS dataset smoke test (no pytest needed).

Key Functions/Classes:
    - main(): Instantiate dataset and print keys.

CLI Arguments:
    (none)

Usage:
    python tivit/tests/test_dataset_omaps.py
"""

from __future__ import annotations

from pathlib import Path

from tivit.core.config import load_yaml_file
from tivit.data.datasets.omaps_impl import OMAPSDataset


def main() -> None:
    """Instantiate dataset and print basic keys."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "dataset" / "omaps.yaml"
    cfg = dict(load_yaml_file(cfg_path))
    dataset_cfg = dict(cfg.get("dataset", {}))
    dataset_cfg.update({"frames": 2, "max_clips": 8, "require_labels": False})
    cfg["dataset"] = dataset_cfg
    ds = OMAPSDataset(cfg, split=dataset_cfg.get("split_test", "test"), full_cfg=cfg)
    print("len", len(ds))
    if len(ds) > 0:
        sample = ds[0]
        print("keys", list(sample.keys()))


if __name__ == "__main__":
    main()
