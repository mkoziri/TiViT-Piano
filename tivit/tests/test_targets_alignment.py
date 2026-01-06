#!/usr/bin/env python3
"""
Purpose:
    Smoke test for frame target alignment (no pytest needed).

Key Functions/Classes:
    - main(): Build frame targets and print keys/shapes.

CLI Arguments:
    (none)

Usage:
    python tivit/tests/test_targets_alignment.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from tivit.data.targets.frame_targets import (
    resolve_frame_target_spec,
    resolve_soft_target_config,
    prepare_frame_targets,
)
from tivit.data.cache.frame_target_cache import FrameTargetCache


def main() -> None:
    """Build frame targets and print keys/shapes."""
    events = [(0.1, 0.5, 60), (0.6, 1.0, 64)]
    labels = torch.tensor(events, dtype=torch.float32)
    spec = resolve_frame_target_spec(
        {"enable": True, "tolerance": 0.03, "note_min": 21, "note_max": 108},
        frames=16,
        stride=1,
        fps=30.0,
        canonical_hw=(145, 1024),
    )
    cache = FrameTargetCache(Path(".cache/test_targets"))
    result = prepare_frame_targets(
        labels=labels,
        lag_result=None,
        spec=spec,  # type: ignore[arg-type]
        cache=cache,
        split="test",
        video_id="video_000",
        clip_start=0.0,
        soft_targets=resolve_soft_target_config({"enable": False}),
    )
    payload = result.payload or {}
    print("keys", list(payload.keys()))
    if payload.get("onset_roll") is not None:
        print("onset shape", tuple(payload["onset_roll"].shape))


if __name__ == "__main__":
    main()
