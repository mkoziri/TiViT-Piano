"""Validate the explicit training.loss schema for MultitaskLoss."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")  # noqa: F401

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


def test_loss_schema_reads_explicit_config():
    cfg = _base_cfg()
    loss = MultitaskLoss(cfg)

    assert loss.head_weights == {"pitch": 1.0, "onset": 2.0, "offset": 3.0, "hand": 0.4, "clef": 0.5}
    assert loss.neg_smooth == 0.1
    assert loss.per_tile_defaults["enabled"] is False
    assert loss.pitch_cfg["pos_weight_mode"] == "sqrt"
    assert loss.onset_cfg["pos_weight"] == 3.0
    assert loss.offset_cfg["pos_weight_mode"] == "adaptive"
    assert loss.offset_cfg["pos_weight_band"] == (0.5, 2.0)


def test_loss_schema_missing_required_key_raises():
    cfg = _base_cfg()
    cfg_bad = copy.deepcopy(cfg)
    cfg_bad["training"]["loss"].pop("head_weights")
    with pytest.raises(ValueError):
        MultitaskLoss(cfg_bad)

