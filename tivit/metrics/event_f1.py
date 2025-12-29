"""Event-level F1 helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def event_f1(pred: Sequence[int], target: Sequence[int]) -> float:
    pred_set = set(int(p) for p in pred)
    tgt_set = set(int(t) for t in target)
    tp = len(pred_set & tgt_set)
    fp = len(pred_set - tgt_set)
    fn = len(tgt_set - pred_set)
    denom = (2 * tp + fp + fn)
    return float(0.0 if denom == 0 else (2 * tp) / denom)


def strict_loose_f1(results: Mapping[str, Sequence[int]]) -> Mapping[str, float]:
    return {
        "strict": event_f1(results.get("pred_strict", ()), results.get("target_strict", ())),
        "loose": event_f1(results.get("pred_loose", ()), results.get("target_loose", ())),
    }


__all__ = ["event_f1", "strict_loose_f1"]

