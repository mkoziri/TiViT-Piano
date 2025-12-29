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

import numpy as np
import torch

from tivit.data.roi.keyboard_roi import RegistrationRefiner


def main() -> None:
    """Run refiner on synthetic frames and print homography."""
    frames = torch.rand(4, 3, 145, 300)
    refiner = RegistrationRefiner((145, 300))
    result = refiner.refine("video_dummy", [f.permute(1, 2, 0).numpy() for f in frames])
    print("status", result.status)
    print("homography", result.homography)


if __name__ == "__main__":
    main()

