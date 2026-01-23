#!/usr/bin/env python3
"""Decode-time hand gate sanity check (standalone script).

Purpose:
    - Verify hand gating applies when enabled and suppresses the left-hand region.
Key Functions/Classes:
    - main: Runs the sanity check and prints a success message.
CLI Arguments:
    (none)
Usage:
    python tivit/tests/test_hand_gate_decode.py
"""

from __future__ import annotations

import torch

from tivit.postproc.hand_gate_runtime import apply_hand_gate_from_config


def _make_cfg(enabled: bool) -> dict:
    return {
        "decoder": {
            "post": {
                "hand_gate": {
                    "enabled": enabled,
                }
            }
        },
        "priors": {
            "hand_gating": {
                "strength": 1.0,
                "apply_to": ["onset"],
                "decode_mode": "multiply",
            }
        },
        "dataset": {"frame_targets": {"note_min": 21, "clef_thresholds": [60, 64]}},
    }


def main() -> None:
    T, P = 2, 88
    probs = {"onset": torch.ones((T, P))}
    hand_logits = torch.tensor([[-5.0, 5.0], [-5.0, 5.0]])
    cfg = _make_cfg(enabled=True)

    gated = apply_hand_gate_from_config(probs, {"hand_logits": hand_logits}, cfg, input_is_logits=True)
    assert "onset" in gated, "hand gate should update onset probabilities"
    gated_onset = gated["onset"]

    split_idx = 60 - 21
    left_mean = float(gated_onset[:, :split_idx].mean().item())
    right_mean = float(gated_onset[:, split_idx:].mean().item())
    assert left_mean < 0.1, f"expected left side to be suppressed, got {left_mean:.3f}"
    assert right_mean > 0.9, f"expected right side to stay high, got {right_mean:.3f}"

    cfg_off = _make_cfg(enabled=False)
    gated_off = apply_hand_gate_from_config(probs, {"hand_logits": hand_logits}, cfg_off, input_is_logits=True)
    assert not gated_off, "hand gate should be disabled when enabled=false"

    print("ok: hand gate decode sanity check")


if __name__ == "__main__":
    main()
