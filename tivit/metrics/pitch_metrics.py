"""Pitch-level metrics."""

from __future__ import annotations

import numpy as np


def frame_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape or pred.size == 0:
        return 0.0
    matches = (pred == target).sum()
    return float(matches) / float(pred.size)


__all__ = ["frame_accuracy"]

