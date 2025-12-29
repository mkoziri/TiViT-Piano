#!/usr/bin/env python3
"""
Purpose:
    Onset-balanced sampler smoke test.

Key Functions/Classes:
    - main(): Build sampler and print sample counts.

CLI Arguments:
    (none)

Usage:
    python tivit/tests/test_sampler.py
"""

from __future__ import annotations

import collections

from tivit.data.sampler import build_onset_balanced_sampler


class DummyDataset:
    def __init__(self) -> None:
        self.onset_sampler_metadata = {
            "onset": [0, 1, 2],
            "background": [3, 4, 5, 6, 7, 8, 9],
            "start_frames": {0: 0, 1: 10, 2: 20},
        }

    def __len__(self) -> int:
        return 10


def main() -> None:
    """Build sampler and print sample counts."""
    sampler = build_onset_balanced_sampler(
        DummyDataset(),
        {"mode": "onset_balanced", "onset_frac": 0.6, "nearmiss_frac": 0.2, "bg_frac": 0.2},
        base_seed=123,
    )
    assert sampler is not None, "Expected onset-balanced sampler to be constructed"
    counts = collections.Counter()
    for idx in sampler:
        counts[int(idx)] += 1
    print(dict(counts))


if __name__ == "__main__":
    main()
