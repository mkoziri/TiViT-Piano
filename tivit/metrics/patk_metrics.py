"""PATK-style evaluation helpers.

Purpose:
    - Compute framewise, onsetwise, and notewise F1 metrics from note events.
    - Provide greedy matching for pitch-aligned notes with tolerance windows.
Key Functions/Classes:
    - frame_counts(): Count TP/FP/FN for binary frame rolls.
    - onset_event_counts(): Count TP/FP/FN for onsetwise note events.
    - note_event_counts(): Count TP/FP/FN for notewise note events (onset+offset).
CLI Arguments:
    (none)
Usage:
    tp, fp, fn = onset_event_counts(pred_notes, ref_notes, onset_tolerance=0.1)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


NoteEvent = Tuple[float, float, int]


def frame_counts(pred_mask, target_mask) -> tuple[int, int, int]:
    """Compute TP/FP/FN over binary frame masks."""
    pred = pred_mask.bool()
    target = target_mask.bool()
    tp = int((pred & target).sum().item())
    fp = int((pred & (~target)).sum().item())
    fn = int(((~pred) & target).sum().item())
    return tp, fp, fn


def _group_by_pitch(notes: Sequence[NoteEvent]) -> dict[int, List[NoteEvent]]:
    groups: dict[int, List[NoteEvent]] = {}
    for onset, offset, pitch in notes:
        groups.setdefault(int(pitch), []).append((float(onset), float(offset), int(pitch)))
    for pitch in groups:
        groups[pitch].sort(key=lambda x: x[0])
    return groups


def onset_event_counts(
    pred_notes: Sequence[NoteEvent],
    ref_notes: Sequence[NoteEvent],
    *,
    onset_tolerance: float,
) -> tuple[int, int, int]:
    """Compute TP/FP/FN for onsetwise note matching."""
    ref_by_pitch = _group_by_pitch(ref_notes)
    pred_by_pitch = _group_by_pitch(pred_notes)
    tp = fp = fn = 0
    for pitch, ref_list in ref_by_pitch.items():
        used = [False] * len(ref_list)
        pred_list = pred_by_pitch.get(pitch, [])
        for onset_p, _offset_p, _ in pred_list:
            best_idx = None
            best_diff = None
            for idx, (onset_r, _offset_r, _) in enumerate(ref_list):
                if used[idx]:
                    continue
                diff = abs(onset_r - onset_p)
                if diff <= onset_tolerance and (best_diff is None or diff < best_diff):
                    best_diff = diff
                    best_idx = idx
            if best_idx is not None:
                used[best_idx] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for flag in used if not flag)
    for pitch, pred_list in pred_by_pitch.items():
        if pitch not in ref_by_pitch:
            fp += len(pred_list)
    return tp, fp, fn


def note_event_counts(
    pred_notes: Sequence[NoteEvent],
    ref_notes: Sequence[NoteEvent],
    *,
    onset_tolerance: float,
    offset_ratio: float,
    offset_min_tolerance: float,
) -> tuple[int, int, int]:
    """Compute TP/FP/FN for notewise note matching (onset + offset)."""
    ref_by_pitch = _group_by_pitch(ref_notes)
    pred_by_pitch = _group_by_pitch(pred_notes)
    tp = fp = fn = 0
    for pitch, ref_list in ref_by_pitch.items():
        used = [False] * len(ref_list)
        pred_list = pred_by_pitch.get(pitch, [])
        for onset_p, offset_p, _ in pred_list:
            best_idx = None
            best_diff = None
            for idx, (onset_r, offset_r, _) in enumerate(ref_list):
                if used[idx]:
                    continue
                onset_diff = abs(onset_r - onset_p)
                if onset_diff > onset_tolerance:
                    continue
                duration = max(offset_r - onset_r, 0.0)
                offset_tol = max(float(offset_min_tolerance), float(offset_ratio) * duration)
                if abs(offset_r - offset_p) > offset_tol:
                    continue
                if best_diff is None or onset_diff < best_diff:
                    best_diff = onset_diff
                    best_idx = idx
            if best_idx is not None:
                used[best_idx] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for flag in used if not flag)
    for pitch, pred_list in pred_by_pitch.items():
        if pitch not in ref_by_pitch:
            fp += len(pred_list)
    return tp, fp, fn


__all__ = [
    "NoteEvent",
    "frame_counts",
    "onset_event_counts",
    "note_event_counts",
]
