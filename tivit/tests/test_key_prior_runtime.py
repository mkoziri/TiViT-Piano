"""Key-prior runtime tests.

Purpose:
    - Verify runtime settings resolve from config with the decoder toggle.
    - Ensure the key prior returns correctly shaped logits when enabled.

Key Functions/Classes:
    - test_key_prior_runtime_disabled(): Confirms disabled toggle returns None.
    - test_key_prior_runtime_apply(): Confirms enabled toggle applies prior.

CLI Arguments:
    (none)

Usage:
    pytest tivit/tests/test_key_prior_runtime.py
"""

from __future__ import annotations

import torch

from tivit.postproc.key_prior_runtime import apply_key_prior_from_config, resolve_key_prior_runtime


def _base_cfg(enabled: bool) -> dict:
    return {
        "decoder": {"post": {"key_prior": {"enabled": enabled}}},
        "priors": {
            "key_signature": {
                "ref_head": "onset",
                "apply_to": ["onset"],
                "window_sec": 1.0,
            }
        },
        "dataset": {"decode_fps": 30.0, "frame_targets": {"note_min": 21, "note_max": 108}},
    }


def test_key_prior_runtime_disabled():
    runtime = resolve_key_prior_runtime(_base_cfg(False))
    assert runtime is None


def test_key_prior_runtime_apply():
    cfg = _base_cfg(True)
    runtime = resolve_key_prior_runtime(cfg)
    assert runtime is not None
    settings, fps, midi_low, midi_high = runtime
    assert settings.enabled
    assert fps == 30.0
    assert midi_low == 21
    assert midi_high == 108

    logits = torch.zeros((4, 88), dtype=torch.float32)
    out = apply_key_prior_from_config({"onset": logits}, cfg)
    assert "onset" in out
    assert out["onset"].shape == logits.shape
