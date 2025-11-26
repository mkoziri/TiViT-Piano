"""Sanity check for adaptive_band positive weights."""

from __future__ import annotations

import torch

from scripts.train import _banded_pos_weight_from_roll


def _build_roll(length: int, positives: torch.Tensor) -> torch.Tensor:
    """Expand per-pitch positive counts into a binary roll tensor."""
    P = positives.numel()
    roll = torch.zeros((length, P), dtype=torch.float32)
    for pitch, count in enumerate(positives.tolist()):
        count = int(count)
        if count <= 0:
            continue
        count = min(count, length)
        roll[:count, pitch] = 1.0
    return roll


def _assert_band(weights: torch.Tensor, band: tuple[float, float]) -> None:
    low, high = band if band[0] <= band[1] else (band[1], band[0])
    if not ((weights >= low - 1e-6).all() and (weights <= high + 1e-6).all()):
        raise AssertionError(f"Weights {weights.tolist()} outside band {band}")


def main() -> None:
    # Case 1: onset-like distribution with ratios mapped to [3,5]
    onset_band = (3.0, 5.0)
    onset_roll = _build_roll(10, torch.tensor([2, 5, 8]))
    onset_weights = _banded_pos_weight_from_roll(onset_roll, onset_band)
    _assert_band(onset_weights, onset_band)

    # Case 2: frame/pitch distribution mapped to [2,4]
    pitch_band = (2.0, 4.0)
    pitch_roll = _build_roll(12, torch.tensor([1, 6, 10, 3]))
    pitch_weights = _banded_pos_weight_from_roll(pitch_roll, pitch_band)
    _assert_band(pitch_weights, pitch_band)

    print("Adaptive band pos_weight checks passed.")


if __name__ == "__main__":
    main()
