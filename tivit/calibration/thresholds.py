"""Threshold sweep helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from tivit.decoder.decode import DECODER_DEFAULTS
from tivit.metrics import f1_from_counts


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_threshold_priors(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    priors_path = Path(path).expanduser()
    if not priors_path.exists():
        return None
    with priors_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        return None
    recommendations = payload.get("recommendations")
    if isinstance(recommendations, Mapping):
        return recommendations
    return None


def resolve_threshold_center(
    cfg: Mapping[str, Any],
    recommendations: Mapping[str, Any] | None,
    head: str,
) -> tuple[float, str]:
    if recommendations is not None:
        thresholds = recommendations.get("thresholds")
        if isinstance(thresholds, Mapping):
            entry = thresholds.get(head)
            if isinstance(entry, Mapping):
                default_val = _coerce_optional_float(entry.get("default"))
                if default_val is None:
                    raw_range = entry.get("range")
                    if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2:
                        low = _coerce_optional_float(raw_range[0])
                        high = _coerce_optional_float(raw_range[1])
                        if low is not None and high is not None:
                            return 0.5 * (low + high), "threshold_priors"
                else:
                    return default_val, "threshold_priors"
    training_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    metrics_cfg = training_cfg.get("metrics", {}) if isinstance(training_cfg, Mapping) else {}
    base = _coerce_optional_float(metrics_cfg.get("prob_threshold"))
    if base is None:
        base = 0.2
    head_key = f"prob_threshold_{head}"
    head_val = _coerce_optional_float(metrics_cfg.get(head_key))
    return (head_val if head_val is not None else base), "training.metrics"


def build_sweep_values(
    center: float,
    delta: float,
    steps: int,
    *,
    min_prob: float,
    max_prob: float,
) -> list[float]:
    steps = max(int(steps), 1)
    if steps % 2 == 0:
        steps += 1
    mid = steps // 2
    values = []
    for idx in range(steps):
        val = center + float(delta) * (idx - mid)
        val = max(float(min_prob), min(float(max_prob), val))
        values.append(val)
    unique = sorted({round(v, 6) for v in values})
    return unique


def resolve_hold(open_thr: float, template: Mapping[str, Any], head: str) -> float:
    ratio = _coerce_optional_float(template.get("low_ratio"))
    hold = None
    if ratio is not None:
        hold = open_thr * max(0.0, ratio)
    if hold is None:
        hold = _coerce_optional_float(template.get("hold"))
        if hold is None:
            hold = DECODER_DEFAULTS.get(head, {}).get("hold", open_thr)
    if not math.isfinite(hold):
        hold = open_thr
    hold = max(0.0, min(float(hold), float(open_thr)))
    return hold


def summarize_sweep(
    thresholds: list[float],
    holds: list[float],
    counts: list[dict[str, int]],
    *,
    center: float,
) -> tuple[list[dict[str, Any]], int]:
    results: list[dict[str, Any]] = []
    best_idx = 0
    best_key = (-1.0, float("-inf"))
    for idx, (thr, hold, count) in enumerate(zip(thresholds, holds, counts)):
        summary = f1_from_counts(count["tp"], count["fp"], count["fn"])
        result = {
            "threshold": thr,
            "hold": hold,
            "precision": summary.precision,
            "recall": summary.recall,
            "f1": summary.f1,
            "tp": count["tp"],
            "fp": count["fp"],
            "fn": count["fn"],
            "clips": count["clips"],
        }
        results.append(result)
        key = (summary.f1, -abs(thr - center))
        if key > best_key:
            best_key = key
            best_idx = idx
    return results, best_idx


__all__ = [
    "build_sweep_values",
    "load_threshold_priors",
    "resolve_hold",
    "resolve_threshold_center",
    "summarize_sweep",
]
