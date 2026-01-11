#!/usr/bin/env python3
"""Multitask loss config schema smoke test.

Purpose:
    - Ensure the explicit training.loss schema is accepted without hidden defaults.
    - Catch missing-required-key failures early without relying on pytest.
Key Functions/Classes:
    - main(): run the schema checks.
CLI Arguments:
    (none)
Usage:
    python tivit/tests/test_loss_config_schema.py
"""

from __future__ import annotations

import copy
import sys

try:
    import torch  # noqa: F401
except ImportError:
    print("torch not available; skipping loss config schema smoke test", file=sys.stderr)
    sys.exit(0)

from tivit.losses.multitask_loss import MultitaskLoss


def _base_cfg() -> dict:
    return {
        "training": {
            "loss": {
                "head_weights": {"pitch": 1.0, "onset": 2.0, "offset": 3.0, "hand": 0.4, "clef": 0.5},
                "ema_alpha": 0.25,
                "neg_smooth_onoff": 0.1,
                "per_tile": {
                    "enabled": False,
                    "heads": ["pitch", "onset", "offset"],
                    "mask_cushion_keys": None,
                    "debug": {"enabled": False, "interval": 0},
                },
                "heads": {
                    "pitch": {
                        "loss": "bce",
                        "pos_weight_mode": "sqrt",
                        "pos_weight": None,
                        "pos_weight_band": [1.0, 3.0],
                    },
                    "onset": {
                        "loss": "bce",
                        "pos_weight_mode": "fixed",
                        "pos_weight": 3.0,
                        "pos_weight_band": None,
                        "focal_gamma": 2.0,
                        "focal_alpha": 0.25,
                        "prior_mean": 0.02,
                        "prior_weight": 0.5,
                    },
                    "offset": {
                        "loss": "focal",
                        "pos_weight_mode": "adaptive",
                        "pos_weight": None,
                        "pos_weight_band": [0.5, 2.0],
                        "focal_gamma": 2.5,
                        "focal_alpha": 0.1,
                        "prior_mean": 0.01,
                        "prior_weight": 0.25,
                    },
                    "hand": {"loss": "ce"},
                    "clef": {"loss": "ce"},
                },
            }
        }
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_value_error(cfg: dict) -> None:
    try:
        MultitaskLoss(cfg)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid loss config")


def main() -> None:
    cfg = _base_cfg()
    loss = MultitaskLoss(cfg)

    _assert(loss.head_weights == {"pitch": 1.0, "onset": 2.0, "offset": 3.0, "hand": 0.4, "clef": 0.5}, "head_weights mismatch")
    _assert(loss.neg_smooth == 0.1, "neg_smooth mismatch")
    _assert(loss.per_tile_defaults["enabled"] is False, "per_tile enabled flag mismatch")
    _assert(loss.pitch_cfg["pos_weight_mode"] == "sqrt", "pitch pos_weight_mode mismatch")
    _assert(loss.onset_cfg["pos_weight"] == 3.0, "onset pos_weight mismatch")
    _assert(loss.offset_cfg["pos_weight_mode"] == "adaptive", "offset pos_weight_mode mismatch")
    _assert(loss.offset_cfg["pos_weight_band"] == (0.5, 2.0), "offset pos_weight_band mismatch")

    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["training"]["loss"].pop("head_weights")
    _expect_value_error(cfg_bad)

    print("loss config schema checks passed")


if __name__ == "__main__":
    main()
