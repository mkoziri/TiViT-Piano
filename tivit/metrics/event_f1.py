"""Event-level F1 computation helpers.

Purpose:
    - Provide tolerance-aware matching between predicted and reference piano-roll events for consistent evaluation.
    - Return structured precision/recall/F1 plus TP/FP/FN counts for logging and sweeps.
Key Functions/Classes:
    - ``EventF1Result``: Container for the aggregated F1 triple and supporting counts.
    - ``event_f1``: Compute tolerance-aware event F1 for 2D/3D pianoroll masks.
    - ``f1_from_counts``: Convert TP/FP/FN totals into precision/recall/F1.
CLI Arguments:
    (none)
Usage:
    from tivit.metrics import event_f1
    result = event_f1(pred_mask, target_mask, hop_seconds=0.02, tolerance=0.05)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    import torch
    MaskLike: TypeAlias = np.ndarray | torch.Tensor
else:
    MaskLike: TypeAlias = np.ndarray


@dataclass
class EventF1Result:
    """Structured event-level F1 statistics."""

    f1: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    clips_evaluated: int


def _to_bool_mask(mask: MaskLike) -> np.ndarray:
    """Convert supported mask inputs to a boolean numpy array."""

    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim not in (2, 3):
        raise ValueError("Event masks must be 2D (T, P) or 3D (B, T, P)")
    return arr.astype(bool, copy=False)


def _event_counts_for_clip(pred_mask: np.ndarray, target_mask: np.ndarray, hop_seconds: float, tolerance: float) -> tuple[int, int, int] | None:
    pred_idx = np.argwhere(pred_mask)
    target_idx = np.argwhere(target_mask)
    if pred_idx.size == 0 and target_idx.size == 0:
        return None

    pred_times = pred_idx[:, 0].astype(np.float64) * float(hop_seconds)
    target_times = target_idx[:, 0].astype(np.float64) * float(hop_seconds)
    pred_pitch = pred_idx[:, 1]
    target_pitch = target_idx[:, 1]

    used = np.zeros(target_idx.shape[0], dtype=bool)
    tp = 0
    for i in range(pred_idx.shape[0]):
        pitch = pred_pitch[i]
        time_val = pred_times[i]
        mask = (target_pitch == pitch) & (~used)
        if not mask.any():
            continue
        cand_idx = np.flatnonzero(mask)
        diffs = np.abs(target_times[cand_idx] - time_val)
        rel = int(np.argmin(diffs))
        if diffs[rel] <= tolerance:
            tp += 1
            used[cand_idx[rel]] = True

    fp = int(pred_idx.shape[0] - tp)
    fn = int(target_idx.shape[0] - tp)
    return tp, fp, fn


def event_f1(pred: MaskLike, target: MaskLike, *, hop_seconds: float, tolerance: float, eps: float = 1e-8) -> EventF1Result:
    """Compute event-level F1 with a temporal tolerance window per clip."""

    pred_mask = _to_bool_mask(pred)
    target_mask = _to_bool_mask(target)
    if pred_mask.shape != target_mask.shape:
        raise ValueError("Prediction and target masks must share shape for event F1 computation")

    hop = float(hop_seconds)
    tol = float(tolerance)
    if not np.isfinite(hop) or hop <= 0:
        raise ValueError("hop_seconds must be positive and finite")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tolerance must be non-negative and finite")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    evaluated = 0

    if pred_mask.ndim == 2:
        counts = _event_counts_for_clip(pred_mask, target_mask, hop, tol)
        if counts is not None:
            tp, fp, fn = counts
            total_tp += tp
            total_fp += fp
            total_fn += fn
            evaluated = 1
    else:
        for clip_pred, clip_target in zip(pred_mask, target_mask):
            counts = _event_counts_for_clip(clip_pred, clip_target, hop, tol)
            if counts is None:
                continue
            tp, fp, fn = counts
            total_tp += tp
            total_fp += fp
            total_fn += fn
            evaluated += 1

    precision = total_tp / (total_tp + total_fp + eps) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn + eps) if (total_tp + total_fn) > 0 else 0.0
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall + eps)

    return EventF1Result(
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        true_positives=int(total_tp),
        false_positives=int(total_fp),
        false_negatives=int(total_fn),
        clips_evaluated=int(evaluated),
    )


def f1_from_counts(tp: int, fp: int, fn: int, *, eps: float = 1e-8) -> EventF1Result:
    """Compute precision/recall/F1 from aggregated counts."""
    total_tp = int(tp)
    total_fp = int(fp)
    total_fn = int(fn)
    precision = total_tp / (total_tp + total_fp + eps) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn + eps) if (total_tp + total_fn) > 0 else 0.0
    if precision == 0.0 and recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall + eps)
    return EventF1Result(
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        clips_evaluated=0,
    )


__all__ = ["EventF1Result", "event_f1", "f1_from_counts"]
