import torch

from scripts.calib.threshold_utils import (
    build_probability_thresholds,
    coerce_quantiles,
    unique_sorted_thresholds,
)


def _noop_log(_msg: str, _force: bool = True) -> None:
    return


def test_quantile_handles_small_max():
    probs = torch.full((10,), 0.01)
    thresholds, mode_used, _reason, rate_range = build_probability_thresholds(
        "onset",
        probs,
        mode="quantile",
        default_grid=[0.2, 0.3, 0.4],
        quantiles=coerce_quantiles([0, 50, 90]),
        floor_band=[0.2, 0.3],
        include_max_quantile=True,
        explicit=None,
        agg_mode="any",
        cap_count=1,
        top_k=0,
        log_fn=_noop_log,
    )
    assert mode_used == "quantile"
    assert thresholds[0] == 0.0
    assert max(thresholds) <= 0.0101
    assert rate_range[1] > 0.0


def test_explicit_overrides_quantile():
    probs = torch.linspace(0.0, 0.5, steps=10)
    explicit = [0.25, 0.35]
    thresholds, mode_used, reason, _ = build_probability_thresholds(
        "onset",
        probs,
        mode="quantile",
        default_grid=[0.1, 0.2, 0.3],
        quantiles=coerce_quantiles([0, 50, 90]),
        floor_band=[],
        include_max_quantile=True,
        explicit=explicit,
        agg_mode="any",
        cap_count=1,
        top_k=0,
        log_fn=_noop_log,
    )
    assert mode_used == "explicit"
    assert "explicit" in reason
    assert thresholds == unique_sorted_thresholds(explicit)


def test_thresholds_with_typical_range():
    probs = torch.linspace(0.1, 0.5, steps=20)
    thresholds, mode_used, _reason, rate_range = build_probability_thresholds(
        "offset",
        probs,
        mode="quantile",
        default_grid=[0.1, 0.2, 0.3, 0.4],
        quantiles=coerce_quantiles([0, 50, 90, 99]),
        floor_band=[0.2],
        include_max_quantile=True,
        explicit=None,
        agg_mode="any",
        cap_count=1,
        top_k=0,
        log_fn=_noop_log,
    )
    assert mode_used == "quantile"
    assert max(thresholds) <= 0.5 + 1e-6
    assert rate_range[1] > rate_range[0]


def test_pred_rate_range_nonzero():
    probs = torch.linspace(0.0, 0.5, steps=6)
    thresholds, _mode_used, _reason, rate_range = build_probability_thresholds(
        "onset",
        probs,
        mode="absolute",
        default_grid=[0.0, 0.1, 0.2, 0.3],
        quantiles=[],
        floor_band=[],
        include_max_quantile=True,
        explicit=None,
        agg_mode="any",
        cap_count=1,
        top_k=0,
        log_fn=_noop_log,
    )
    assert len(thresholds) > 1
    assert rate_range[1] > 0.0
    assert rate_range[1] > rate_range[0]
