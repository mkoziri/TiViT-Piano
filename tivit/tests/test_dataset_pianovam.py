#!/usr/bin/env python3
"""
Purpose:
    Lightweight PianoVAM dataset smoke test (no pytest needed).

Key Functions/Classes:
    - main(): Instantiate dataset and print keys.

CLI Arguments:
    (none)

Usage:
    python tivit/tests/test_dataset_pianovam.py
"""

from __future__ import annotations

from tivit.data.datasets.pianovam_impl import PianoVAMDataset


def main() -> None:
    """Instantiate dataset and print basic keys."""
    ds = PianoVAMDataset({"dataset": {"frames": 2}}, split="test", full_cfg={"dataset": {"frames": 2}})
    print("len", len(ds))
    if len(ds) > 0:
        sample = ds[0]
        print("keys", list(sample.keys()))


if __name__ == "__main__":
    main()
