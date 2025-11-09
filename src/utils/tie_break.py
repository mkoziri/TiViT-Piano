from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

TIE_BREAK_EPS_F1 = 1e-4
TIE_BREAK_EPS_RATE = 1e-4
TIE_BREAK_EPS_THR = 1e-6

_TIE_BREAK_EPS_LOGGED = False


def log_tie_break_eps() -> None:
    """Emit the tolerance line (once per process) to make logs auditable."""
    global _TIE_BREAK_EPS_LOGGED
    if _TIE_BREAK_EPS_LOGGED:
        return
    print(
        "[tie-break] eps_f1={:.1e} eps_rate={:.1e} eps_thr={:.1e}".format(
            TIE_BREAK_EPS_F1,
            TIE_BREAK_EPS_RATE,
            TIE_BREAK_EPS_THR,
        )
    )
    _TIE_BREAK_EPS_LOGGED = True


@dataclass
class OnsetTieBreakContext:
    anchor_used: Optional[float]
    prev_threshold: Optional[float]
    eps_f1: float = TIE_BREAK_EPS_F1
    eps_rate: float = TIE_BREAK_EPS_RATE
    eps_thr: float = TIE_BREAK_EPS_THR


def _abs_rate_delta(row: Mapping[str, float]) -> float:
    pred = float(row["onset_pred_rate"])
    pos = float(row["onset_pos_rate"])
    return abs(pred - pos)


def _compare_with_tol(lhs: float, rhs: float, tol: float) -> int:
    """Return 1 if lhs>rhs beyond tol, -1 if lhs<rhs beyond tol, else 0."""
    delta = lhs - rhs
    if delta > tol:
        return 1
    if delta < -tol:
        return -1
    return 0


def select_best_onset_row(
    rows: Sequence[Mapping[str, float]],
    ctx: OnsetTieBreakContext,
) -> Tuple[Mapping[str, float], str]:
    """Select the best onset row using the Stage-B tie-break ladder.

    The returned reason is ``metric`` when the primary score (F1) suffices,
    otherwise ``TB2`` through ``TB5`` to denote which ladder rung resolved
    the tie.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    best = rows[0]
    reason = "metric"
    for candidate in rows[1:]:
        winner, winner_reason = _resolve_pair(best, candidate, ctx)
        if winner is candidate:
            best = candidate
            reason = winner_reason or reason
    return best, reason


def _resolve_pair(
    current: Mapping[str, float],
    challenger: Mapping[str, float],
    ctx: OnsetTieBreakContext,
) -> Tuple[Mapping[str, float], Optional[str]]:
    """Return the preferred row along with the rung that resolved the tie."""
    cur_f1 = float(current["onset_event_f1"])
    new_f1 = float(challenger["onset_event_f1"])
    cmp_f1 = _compare_with_tol(new_f1, cur_f1, ctx.eps_f1)
    if cmp_f1 > 0:
        return challenger, "metric"
    if cmp_f1 < 0:
        return current, None

    cur_diff = _abs_rate_delta(current)
    new_diff = _abs_rate_delta(challenger)
    cmp_rate = _compare_with_tol(cur_diff, new_diff, ctx.eps_rate)
    if cmp_rate > 0:
        return challenger, "TB2"
    if cmp_rate < 0:
        return current, None

    anchor = ctx.anchor_used
    if anchor is not None:
        cur_thr = float(current["onset_thr"])
        new_thr = float(challenger["onset_thr"])
        cur_exact = abs(cur_thr - anchor) <= ctx.eps_thr
        new_exact = abs(new_thr - anchor) <= ctx.eps_thr
        if new_exact and not cur_exact:
            return challenger, "TB3"
        if cur_exact and not new_exact:
            return current, None
        cur_gap = abs(cur_thr - anchor)
        new_gap = abs(new_thr - anchor)
        cmp_anchor = _compare_with_tol(cur_gap, new_gap, ctx.eps_thr)
        if cmp_anchor > 0:
            return challenger, "TB3"
        if cmp_anchor < 0:
            return current, None
    prev_thr = ctx.prev_threshold
    if prev_thr is not None:
        cur_prev = abs(float(current["onset_thr"]) - prev_thr) <= ctx.eps_thr
        new_prev = abs(float(challenger["onset_thr"]) - prev_thr) <= ctx.eps_thr
        if new_prev and not cur_prev:
            return challenger, "TB4"
        if cur_prev and not new_prev:
            return current, None
    cur_thr_val = float(current["onset_thr"])
    new_thr_val = float(challenger["onset_thr"])
    cmp_thr = _compare_with_tol(new_thr_val, cur_thr_val, ctx.eps_thr)
    if cmp_thr > 0:
        return challenger, "TB5"
    if cmp_thr < 0:
        return current, None
    return current, None
