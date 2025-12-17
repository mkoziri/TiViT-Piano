from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple

import torch

from decoder.decode import build_threshold_mask

LogFn = Callable[[str, bool], None]


def unique_sorted_thresholds(values: Iterable[float]) -> List[float]:
    """Return sorted, deduped probability thresholds clipped to [0, 1]."""

    uniq = {}
    for val in values:
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fval):
            continue
        fval = max(0.0, min(1.0, fval))
        key = int(round(fval * 1e6))
        if key not in uniq:
            uniq[key] = fval
    return sorted(uniq.values())


def coerce_quantiles(values: Sequence[float] | None) -> List[float]:
    """Normalize percentile inputs (0-100 or 0-1) into quantile fractions."""

    if not values:
        return []
    fracs: List[float] = []
    for raw in values:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        if abs(val) > 1.0:
            val = val / 100.0
        val = max(0.0, min(1.0, val))
        fracs.append(val)
    return unique_sorted_thresholds(fracs)


def summarize_scores(name: str, scores: torch.Tensor, log_fn: LogFn | None = None) -> dict:
    flat = scores.reshape(-1).float()
    flat = flat[torch.isfinite(flat)]
    stats = {
        "count": int(flat.numel()),
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "p50": float(torch.quantile(flat, 0.50).item()) if flat.numel() else 0.0,
        "p90": float(torch.quantile(flat, 0.90).item()) if flat.numel() else 0.0,
        "p95": float(torch.quantile(flat, 0.95).item()) if flat.numel() else 0.0,
        "p99": float(torch.quantile(flat, 0.99).item()) if flat.numel() else 0.0,
    }
    if log_fn:
        log_fn(
            "[sweep][scores] {name} count={count} min={min:.4f} mean={mean:.4f} "
            "max={max:.4f} p50={p50:.4f} p90={p90:.4f} p95={p95:.4f} p99={p99:.4f}".format(
                name=name, **stats
            ),
            True,
        )
    return stats


def _pred_rate_range(
    probs: torch.Tensor,
    thresholds: Sequence[float],
    *,
    mode: str,
    cap_count: int,
    top_k: int,
) -> Tuple[float, float]:
    rates: List[float] = []
    for thr in thresholds:
        mask = build_threshold_mask(
            probs,
            float(thr),
            mode=mode,
            cap_count=cap_count,
            top_k=top_k,
        )
        rates.append(float(mask.float().mean().item()))
    if not rates:
        return 0.0, 0.0
    return min(rates), max(rates)


def build_probability_thresholds(
    name: str,
    scores: torch.Tensor,
    *,
    mode: str,
    default_grid: Sequence[float],
    quantiles: Sequence[float],
    floor_band: Sequence[float],
    include_max_quantile: bool,
    explicit: Sequence[float] | None,
    agg_mode: str,
    cap_count: int,
    top_k: int,
    log_fn: LogFn | None = None,
) -> Tuple[List[float], str, str, Tuple[float, float]]:
    """Generate probability thresholds that align with the score distribution."""

    flat = scores.reshape(-1).float()
    flat = flat[torch.isfinite(flat)]
    max_score = float(flat.max().item()) if flat.numel() else 0.0

    if explicit is not None:
        thresholds = unique_sorted_thresholds(explicit)
        reason = "explicit list provided"
        mode_used = "explicit"
        rate_range = _pred_rate_range(scores, thresholds, mode=agg_mode, cap_count=cap_count, top_k=top_k)
        if log_fn:
            log_fn(
                f"[sweep] {name} thresholds mode={mode_used} reason={reason} n={len(thresholds)}",
                True,
            )
        return thresholds, mode_used, reason, rate_range

    floor_in_range = [v for v in floor_band if v <= max_score + 1e-9]
    if not floor_in_range and floor_band and max_score > 0.0:
        floor_in_range = [max_score * v for v in floor_band if v >= 0.0]

    base_grid = [float(v) for v in default_grid if float(v) <= max_score + 1e-9 or max_score == 0.0]
    if not base_grid and max_score > 0.0:
        base_grid = torch.linspace(0.0, max_score, steps=4).tolist()

    mode_used = mode
    reason = "quantile" if mode in {"quantile", "hybrid"} else "absolute"

    if mode in {"quantile", "hybrid"}:
        quantile_vals: List[float] = []
        if flat.numel():
            for q in quantiles:
                try:
                    quantile_vals.append(float(torch.quantile(flat, float(q)).item()))
                except Exception:
                    continue
        if include_max_quantile and max_score > 0.0:
            quantile_vals.append(max_score)
        base = [0.0, *quantile_vals, *floor_in_range]
        if max_score > 0.0 and len(unique_sorted_thresholds(base)) < 3:
            base.extend(torch.linspace(0.0, max_score, steps=4).tolist())
        thresholds = unique_sorted_thresholds(base)
        if mode == "hybrid":
            thresholds = unique_sorted_thresholds([*thresholds, *base_grid])
    else:
        thresholds = unique_sorted_thresholds([*base_grid, *floor_in_range])
        if thresholds and thresholds[0] > 0.0:
            thresholds = [0.0, *thresholds]

    rate_range = _pred_rate_range(scores, thresholds, mode=agg_mode, cap_count=cap_count, top_k=top_k)
    if log_fn:
        preview = "[" + ",".join(f"{v:.3f}" for v in thresholds[:3])
        if len(thresholds) > 3:
            preview += f" ... {thresholds[-1]:.3f}"
        preview += "]"
        log_fn(
            f"[sweep] {name} thresholds mode={mode_used} reason={reason} n={len(thresholds)} {preview} "
            f"pred_rate=[{rate_range[0]:.4f},{rate_range[1]:.4f}]",
            True,
        )
    return thresholds, mode_used, reason, rate_range
