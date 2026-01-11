"""Event F1 metric smoke tests.

Purpose:
    - Verify tolerance-aware event matching produces expected counts across common scenarios.
    - Provide quick validation without pytest or unittest dependencies.
Key Functions/Classes:
    - ``run_checks()``: Execute all assertion-based scenarios.
    - ``_assert_close()``: Lightweight float comparison helper.
CLI Arguments:
    (none)
Usage:
    python tivit/tests/test_event_f1.py
"""

from __future__ import annotations

import math
import numpy as np

from tivit.metrics import EventF1Result, event_f1


def _assert_close(val: float, target: float, *, tol: float = 1e-6) -> None:
    assert math.isclose(val, target, rel_tol=tol, abs_tol=tol), f"{val} != {target}"


def test_matches_events_within_tolerance() -> None:
    pred = np.zeros((15, 1), dtype=bool)
    target = np.zeros_like(pred)
    pred[10, 0] = True
    target[13, 0] = True  # 3 frames apart @ 0.01s hop -> 0.03s diff

    result = event_f1(pred, target, hop_seconds=0.01, tolerance=0.05)

    assert isinstance(result, EventF1Result)
    _assert_close(result.f1, 1.0)
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 0, 0)
    assert result.clips_evaluated == 1


def test_counts_pitch_mismatches() -> None:
    pred = np.zeros((5, 2), dtype=bool)
    target = np.zeros_like(pred)
    pred[2, 0] = True
    target[2, 1] = True

    result = event_f1(pred, target, hop_seconds=0.02, tolerance=0.05)

    assert result.f1 == 0.0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 1, 1)


def test_handles_empty_masks() -> None:
    pred = np.zeros((4, 2), dtype=bool)
    target = np.zeros_like(pred)

    result = event_f1(pred, target, hop_seconds=0.01, tolerance=0.05)

    assert result.f1 == 0.0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 0, 0)
    assert result.clips_evaluated == 0


def test_batches_accumulate_counts() -> None:
    pred = np.zeros((2, 4, 1), dtype=bool)
    target = np.zeros_like(pred)
    pred[0, 0, 0] = True
    target[0, 0, 0] = True
    pred[1, 1, 0] = True
    target[1, 2, 0] = True  # falls outside tolerance for second clip

    result = event_f1(pred, target, hop_seconds=0.02, tolerance=0.01)

    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 1, 1)
    assert result.clips_evaluated == 2
    _assert_close(result.precision, 0.5)
    _assert_close(result.recall, 0.5)
    _assert_close(result.f1, 0.5)


def run_checks() -> None:
    test_matches_events_within_tolerance()
    test_counts_pitch_mismatches()
    test_handles_empty_masks()
    test_batches_accumulate_counts()
    print("event_f1 smoke tests passed")


if __name__ == "__main__":
    run_checks()
