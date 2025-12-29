"""Thresholding policies."""

from tivit.decoder.decode import build_threshold_mask, median_filter_time, pool_roll_BT, topk_mask


def build_postproc(*_: object, **__: object):
    return {
        "build_threshold_mask": build_threshold_mask,
        "median_filter_time": median_filter_time,
        "pool_roll_BT": pool_roll_BT,
        "topk_mask": topk_mask,
    }


__all__ = ["build_threshold_mask", "median_filter_time", "pool_roll_BT", "topk_mask", "build_postproc"]
