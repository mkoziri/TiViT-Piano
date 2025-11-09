from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from utils.tie_break import OnsetTieBreakContext, select_best_onset_row


def _row(
    *,
    thr: float,
    f1: float = 0.9,
    pred_rate: float = 0.2,
    pos_rate: float = 0.2,
) -> dict:
    return {
        "onset_thr": thr,
        "onset_event_f1": f1,
        "onset_pred_rate": pred_rate,
        "onset_pos_rate": pos_rate,
    }


def test_tie_break_prefers_balanced_rate() -> None:
    ctx = OnsetTieBreakContext(anchor_used=0.3, prev_threshold=None)
    noisy = _row(thr=0.31, pred_rate=0.35, pos_rate=0.2)
    balanced = _row(thr=0.32, pred_rate=0.21, pos_rate=0.2)
    best, reason = select_best_onset_row([noisy, balanced], ctx)
    assert best is balanced
    assert reason == "TB2"


def test_tie_break_prefers_anchor_exact_match() -> None:
    ctx = OnsetTieBreakContext(anchor_used=0.4, prev_threshold=None)
    exact = _row(thr=0.4)
    offset = _row(thr=0.43)
    best, reason = select_best_onset_row([offset, exact], ctx)
    assert best is exact
    assert reason == "TB3"


def test_tie_break_prefers_anchor_distance() -> None:
    ctx = OnsetTieBreakContext(anchor_used=0.5, prev_threshold=None)
    closer = _row(thr=0.48)
    farther = _row(thr=0.55)
    best, reason = select_best_onset_row([farther, closer], ctx)
    assert best is closer
    assert reason == "TB3"


def test_tie_break_prefers_previous_threshold() -> None:
    ctx = OnsetTieBreakContext(anchor_used=None, prev_threshold=0.6)
    previous = _row(thr=0.6)
    other = _row(thr=0.58)
    best, reason = select_best_onset_row([other, previous], ctx)
    assert best is previous
    assert reason == "TB4"


def test_tie_break_prefers_higher_threshold_as_final_fallback() -> None:
    ctx = OnsetTieBreakContext(anchor_used=None, prev_threshold=None)
    higher = _row(thr=0.62)
    lower = _row(thr=0.58)
    best, reason = select_best_onset_row([lower, higher], ctx)
    assert best is higher
    assert reason == "TB5"


def test_tie_break_is_deterministic_across_orderings() -> None:
    ctx = OnsetTieBreakContext(anchor_used=0.45, prev_threshold=0.5)
    rows = [
        _row(thr=0.45, pred_rate=0.25, pos_rate=0.2),
        _row(thr=0.5, pred_rate=0.25, pos_rate=0.2),
        _row(thr=0.55, pred_rate=0.25, pos_rate=0.2),
    ]
    forward_best, forward_reason = select_best_onset_row(rows, ctx)
    reverse_best, reverse_reason = select_best_onset_row(list(reversed(rows)), ctx)
    assert forward_best is reverse_best
    assert forward_reason == reverse_reason == "TB4"
