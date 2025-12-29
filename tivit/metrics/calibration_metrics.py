"""Calibration metric helpers."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, *, bins: int = 10) -> float:
    probs = probs.reshape(-1)
    targets = targets.reshape(-1)
    if probs.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = targets[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


__all__ = ["expected_calibration_error"]

