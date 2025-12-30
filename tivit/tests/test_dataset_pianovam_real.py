#!/usr/bin/env python3
"""
Purpose:
    Integration-style PianoVAM dataset test that reads actual sample clips/labels
    from the dataset's test split.

Usage:
    python tivit/tests/test_dataset_pianovam_real.py
"""

from __future__ import annotations

from pathlib import Path

from tivit.core.config import load_yaml_file
from tivit.data.datasets.pianovam_impl import PianoVAMDataset


def _resolve_cfg() -> tuple[dict, str]:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "dataset" / "pianovam.yaml"
    cfg = dict(load_yaml_file(cfg_path))
    dataset_cfg = dict(cfg.get("dataset", {}))

    root_dir = dataset_cfg.get("root_dir", "data/PianoVAM")
    root_path = (repo_root / ".." / root_dir).resolve()
    split = dataset_cfg.get("split_test") or dataset_cfg.get("split") or "test"
    split_path = root_path / split

    return {
        "dataset": {
            **dataset_cfg,
            "root_dir": str(root_path),
            "annotations_root": str(dataset_cfg.get("annotations_root", root_path)),
            "max_clips": min(2, len(list(split_path.rglob("*.mp4"))) or 2),
            "num_workers": 0,
            "require_labels": bool(list(split_path.rglob("*.mid")) or list(split_path.rglob("*.midi"))),
        }
    }, split


def main() -> None:
    """Instantiate the dataset using real test data and print sample keys."""
    cfg, split = _resolve_cfg()
    split_dir = Path(cfg["dataset"]["root_dir"]) / split
    videos = sorted(split_dir.rglob("*.mp4"))
    if not videos:
        print(f"[skip] no videos found under {split_dir}; place a couple of clips to run this test")
        return

    ds = PianoVAMDataset(cfg, split=split, full_cfg=cfg)
    print("len", len(ds))
    if len(ds) > 0:
        sample = ds[0]
        print("keys", list(sample.keys()))


if __name__ == "__main__":
    main()
