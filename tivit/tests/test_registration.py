"""
Purpose:
    Smoke test for registration refinement warp (no pytest needed).

Key Functions/Classes:
    - main(): Run refiner on a dummy clip and report homography.

CLI Arguments:
    (none)

Usage:
    python tivit/tests/test_registration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tivit.data.roi.keyboard_roi import RegistrationRefiner


def main() -> None:
    """Run refiner on synthetic frames and print homography."""
    frames = torch.rand(4, 3, 145, 300)
    refiner = RegistrationRefiner((145, 300))
    frames_np = [f.permute(1, 2, 0).numpy() for f in frames]

    # Monkey-patch frame sampling to avoid actual video I/O in the smoke test.
    def _fake_sample(video_path, crop_meta, debug=None):  # type: ignore[override]
        return frames_np

    refiner._sample_video_frames = _fake_sample  # type: ignore[attr-defined]

    result = refiner.refine(video_id="video_dummy", video_path=Path("dummy.mp4"), crop_meta=None)
    print("status", result.status)
    print("homography", result.homography)


if __name__ == "__main__":
    main()
