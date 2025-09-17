"""Purpose:
    Provide unit-style tests that exercise the key-aware prior to ensure key
    estimation and rescoring behave as expected.

Key Functions/Classes:
    - _build_c_major_sequence(): Constructs deterministic logits representing a
      C-major progression for repeatable testing.
    - test_estimate_key_posteriors_prefers_c_major(): Asserts that the prior
      favors the correct key when evaluated on the synthetic sequence.
    - test_rescore_logits_boosts_in_key_classes(): Checks that rescoring
      increases in-key pitch classes relative to out-of-key ones.

CLI:
    Designed for ``pytest``.  Run ``pytest scripts/test_key_prior.py`` or the
    full suite to execute these tests.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from theory.key_prior import KeyAwarePrior, KeyPriorConfig
else:
    try:
        from tivit.theory import KeyAwarePrior, KeyPriorConfig
    except ModuleNotFoundError:  # pragma: no cover - environment guard
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
        from tivit.theory import KeyAwarePrior, KeyPriorConfig
        
        
def _build_c_major_sequence() -> tuple[KeyAwarePrior, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a deterministic C-major logit sequence."""

    midi_low = 60
    midi_high = 71
    config = KeyPriorConfig(window_sec=1.5, fps=4.0, midi_low=midi_low, midi_high=midi_high)
    prior = KeyAwarePrior(config)

    num_frames = 32
    num_pitches = midi_high - midi_low + 1
    logits = np.full((num_frames, num_pitches), -3.0, dtype=np.float64)

    pitch_class_indices = np.mod(np.arange(midi_low, midi_high + 1), 12)
    pc_to_index = {pc: idx for idx, pc in enumerate(pitch_class_indices)}

    scale_pcs = np.array([0, 2, 4, 5, 7, 9, 11])
    triad_pcs = [0, 4, 7]

    for frame in range(num_frames):
        active_pcs = triad_pcs + [scale_pcs[(frame + i) % len(scale_pcs)] for i in range(2)]
        for pc in active_pcs:
            logits[frame, pc_to_index[pc]] = 5.0

    return prior, logits, pitch_class_indices, scale_pcs


def test_estimate_key_posteriors_prefers_c_major() -> None:
    """The key estimator should strongly favor C major for a C-major sequence."""

    prior, logits, _, _ = _build_c_major_sequence()
    posteriors, key_names = prior.estimate_key_posteriors(logits)
    top_indices = np.argmax(posteriors, axis=1)
    c_major_index = key_names.index("C:maj")
    ratio = np.mean(top_indices == c_major_index)
    assert ratio > 0.9, f"C:maj dominance expected, got ratio={ratio:.2f}"


def test_rescore_logits_boosts_in_key_classes() -> None:
    """Rescoring should boost in-key pitch classes relative to out-of-key ones."""

    prior, logits, pitch_class_indices, scale_pcs = _build_c_major_sequence()
    posteriors, key_names = prior.estimate_key_posteriors(logits)
    pc_prior = prior.pc_prior_from_keys(posteriors, key_names)
    rescored = prior.rescore_logits_with_pc_prior(logits, pc_prior)

    deltas = rescored - logits
    in_key_mask = np.isin(pitch_class_indices, scale_pcs)
    in_key_delta = float(deltas[:, in_key_mask].mean())
    out_key_delta = float(deltas[:, ~in_key_mask].mean())

    assert in_key_delta > 0.0
    assert in_key_delta > out_key_delta

