"""Purpose:
    Implement pitch-class key priors that estimate musical keys and rescore
    model logits with theory-aware adjustments.

Key Functions/Classes:
    - build_key_profiles(): Constructs normalized Krumhansl--Schmuckler key
      templates for all major and minor keys.
    - KeyAwarePrior: Estimates frame-level key posteriors and derives
      pitch-class priors that rescore onset/pitch logits.
    - KeyPriorConfig: Dataclass collecting hyper-parameters for the prior
      estimator, including windowing and MIDI ranges.

CLI:
    None.  Import these utilities when post-processing logits or running unit
    tests such as ``scripts/test_key_prior.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "build_key_profiles",
    "KeyPriorConfig",
    "KeyAwarePrior",
]

_PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)


def build_key_profiles() -> Dict[str, np.ndarray]:
    """Return L2-normalized Krumhansl--Schmuckler key profiles.

    The profiles are ordered as 12 major keys followed by 12 minor keys. Values
    are arranged using pitch-class order ``[C, C#, ..., B]``.

    Returns
    -------
    Dict[str, np.ndarray]
        A mapping from key label (e.g. ``"C:maj"``) to a 12-dimensional
        ``float64`` vector with unit L2 norm.
    """

    major_profile = np.array(
        [
            6.35,
            2.23,
            3.48,
            2.33,
            4.38,
            4.09,
            2.52,
            5.19,
            2.39,
            3.66,
            2.29,
            2.88,
        ],
        dtype=np.float64,
    )
    minor_profile = np.array(
        [
            6.33,
            2.68,
            3.52,
            5.38,
            2.6,
            3.53,
            2.54,
            4.75,
            3.98,
            2.69,
            3.34,
            3.17,
        ],
        dtype=np.float64,
    )

    profiles: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(_PITCH_CLASS_NAMES):
        rolled = np.roll(major_profile, idx)
        norm = np.linalg.norm(rolled)
        if norm == 0:
            raise ValueError("Major key profile norm is zero, cannot normalize.")
        profiles[f"{name}:maj"] = rolled / norm

    for idx, name in enumerate(_PITCH_CLASS_NAMES):
        rolled = np.roll(minor_profile, idx)
        norm = np.linalg.norm(rolled)
        if norm == 0:
            raise ValueError("Minor key profile norm is zero, cannot normalize.")
        profiles[f"{name}:min"] = rolled / norm

    return profiles


@dataclass(slots=True)
class KeyPriorConfig:
    """Configuration for :class:`KeyAwarePrior`.

    Parameters
    ----------
    window_sec
        Length of the centered moving-average window in seconds.
    beta
        Temperature used when computing the softmax over keys.
    rho_uniform
        Mixing coefficient for the uniform pitch-class prior.
    prior_strength
        Strength of the logarithmic prior adjustment applied to logits.
    epsilon
        Numerical constant used to avoid divisions by zero and logarithms of
        zero.
    fps
        Frame rate for the logits (frames per second).
    midi_low
        Lowest MIDI pitch (inclusive) covered by the logits.
    midi_high
        Highest MIDI pitch (inclusive) covered by the logits.
    """

    window_sec: float = 3.0
    beta: float = 4.0
    rho_uniform: float = 0.8
    prior_strength: float = 0.5
    epsilon: float = 1e-8
    fps: float = 30.0
    midi_low: int = 21
    midi_high: int = 108

    def __post_init__(self) -> None:
        if self.window_sec < 0:
            raise ValueError("window_sec must be non-negative.")
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if not 0 <= self.rho_uniform <= 1:
            raise ValueError("rho_uniform must lie in the interval [0, 1].")
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be non-negative.")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if self.midi_high < self.midi_low:
            raise ValueError("midi_high must be greater than or equal to midi_low.")


class KeyAwarePrior:
    """Estimate musical keys and apply pitch-class priors to logits.

    The typical workflow is:

    1. Call :meth:`estimate_key_posteriors` with ``(T, P)`` logits to obtain a
       per-frame distribution over the 24 candidate keys and the associated key
       labels.
    2. Pass the resulting posterior to :meth:`pc_prior_from_keys` to convert it
       into a ``(T, 12)`` pitch-class prior with rows that sum to one.
    3. Supply the original logits and the pitch-class prior to
       :meth:`rescore_logits_with_pc_prior` to obtain soft-rescored logits that
       remain finite.

    Each method accepts and returns NumPy arrays so the utilities integrate into
    both offline scripts and JIT-free research workflows.
    """

    def __init__(self, config: KeyPriorConfig | None = None) -> None:
        """Initialize the prior estimator.

        Parameters
        ----------
        config
            Optional configuration. If ``None``, default values are used.
        """

        self.config = config or KeyPriorConfig()
        self._epsilon = self.config.epsilon
        self._debug = os.getenv("TIVIT_THEORY_DEBUG") == "1"

        profiles = build_key_profiles()
        self.key_names: List[str] = list(profiles.keys())
        self._key_profile_matrix = np.vstack([profiles[name] for name in self.key_names])
        profile_sums = np.sum(self._key_profile_matrix, axis=1, keepdims=True)
        self._pc_distributions = self._key_profile_matrix / np.maximum(
            profile_sums, self._epsilon
        )

        self._midi_numbers = np.arange(
            self.config.midi_low, self.config.midi_high + 1, dtype=np.int64
        )
        self._expected_pitches = self._midi_numbers.size
        pitch_class_indices = np.mod(self._midi_numbers, len(_PITCH_CLASS_NAMES))
        self._pitch_class_indices = pitch_class_indices
        self._pitch_class_matrix = np.eye(len(_PITCH_CLASS_NAMES), dtype=np.float64)[
            pitch_class_indices
        ]

        window_frames = max(int(round(self.config.window_sec * self.config.fps)), 1)
        if window_frames % 2 == 0:
            window_frames += 1
        self._window_size = window_frames
        self._log_uniform = -np.log(len(_PITCH_CLASS_NAMES))

    def estimate_key_posteriors(
        self, logits_t_by_m: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Estimate per-frame key posteriors from pitch logits.

        Parameters
        ----------
        logits_t_by_m
            Array of shape ``(T, P)`` with frame-level logits for pitches from
            ``midi_low`` to ``midi_high``.

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Posterior probabilities for each key (shape ``(T, 24)``) and the
            corresponding key labels.

        Raises
        ------
        ValueError
            If the pitch dimension does not match the configured MIDI range.
        """

        logits = np.asarray(logits_t_by_m, dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError("logits_t_by_m must be a 2D array of shape (T, P).")

        num_frames, num_pitches = logits.shape
        if num_pitches != self._expected_pitches:
            raise ValueError(
                "Number of pitch logits does not match configured MIDI range."
            )

        probs = self._sigmoid(logits)
        pc_probs = probs @ self._pitch_class_matrix
        smoothed = self._moving_average(pc_probs)
        norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
        norms = np.maximum(norms, self._epsilon)
        normalized = smoothed / norms

        similarities = normalized @ self._key_profile_matrix.T
        scaled = similarities * self.config.beta
        if num_frames:
            max_per_frame = np.max(scaled, axis=1, keepdims=True)
        else:
            max_per_frame = np.zeros((0, 1), dtype=np.float64)
        stabilized = scaled - max_per_frame
        exp_scores = np.exp(stabilized)
        sums = np.sum(exp_scores, axis=1, keepdims=True)
        sums = np.maximum(sums, self._epsilon)
        key_posteriors = exp_scores / sums

        if self._debug:
            print(f"[KeyAwarePrior] window_size_frames={self._window_size}")
            if num_frames:
                top_indices = np.argmax(key_posteriors, axis=1)
                counts = np.bincount(top_indices, minlength=len(self.key_names))
                top_order = np.argsort(counts)[::-1]
                snapshot = {
                    self.key_names[i]: int(counts[i])
                    for i in top_order[:3]
                    if counts[i] > 0
                }
                print(f"[KeyAwarePrior] top1_key_counts={snapshot}")

        return key_posteriors, self.key_names.copy()

    def pc_prior_from_keys(
        self, P_keys: np.ndarray, key_names: List[str]
    ) -> np.ndarray:
        """Convert key posteriors into per-frame pitch-class priors.

        Parameters
        ----------
        P_keys
            Array of shape ``(T, 24)`` containing key posterior probabilities.
        key_names
            Key labels corresponding to the columns of ``P_keys``.

        Returns
        -------
        np.ndarray
            Pitch-class priors of shape ``(T, 12)`` with rows summing to one.
        """

        posteriors = np.asarray(P_keys, dtype=np.float64)
        if posteriors.ndim != 2:
            raise ValueError("P_keys must be a 2D array of shape (T, 24).")
        if posteriors.shape[1] != self._key_profile_matrix.shape[0]:
            raise ValueError("P_keys must have 24 columns corresponding to keys.")
        if key_names != self.key_names:
            raise ValueError("key_names must match the estimator's key order.")

        row_sums = np.sum(posteriors, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, self._epsilon)
        normalized_posteriors = posteriors / row_sums

        pc_marginal = normalized_posteriors @ self._pc_distributions
        rho = self.config.rho_uniform
        uniform = np.full((1, len(_PITCH_CLASS_NAMES)), 1.0 / len(_PITCH_CLASS_NAMES))
        mixed = rho * uniform + (1.0 - rho) * pc_marginal
        sums = np.sum(mixed, axis=1, keepdims=True)
        sums = np.maximum(sums, self._epsilon)
        return mixed / sums

    def rescore_logits_with_pc_prior(
        self, logits_t_by_m: np.ndarray, Pc_t: np.ndarray
    ) -> np.ndarray:
        """Adjust logits using a pitch-class prior.

        Parameters
        ----------
        logits_t_by_m
            Array of shape ``(T, P)`` with logits for each MIDI pitch.
        Pc_t
            Array of shape ``(T, 12)`` with pitch-class priors per frame.

        Returns
        -------
        np.ndarray
            Rescored logits with the same shape as ``logits_t_by_m``.
        """

        logits = np.asarray(logits_t_by_m, dtype=np.float64)
        if logits.ndim != 2:
            raise ValueError("logits_t_by_m must be a 2D array of shape (T, P).")

        priors = np.asarray(Pc_t, dtype=np.float64)
        if priors.ndim != 2 or priors.shape[1] != len(_PITCH_CLASS_NAMES):
            raise ValueError("Pc_t must have shape (T, 12).")
        if logits.shape[0] != priors.shape[0]:
            raise ValueError("logits_t_by_m and Pc_t must share the same frame count.")
        if logits.shape[1] != self._expected_pitches:
            raise ValueError(
                "Number of pitch logits does not match configured MIDI range."
            )

        log_prior = np.log(priors + self._epsilon)
        offsets = log_prior[:, self._pitch_class_indices] - self._log_uniform
        adjusted = logits + self.config.prior_strength * offsets

        if self._debug:
            print(f"[KeyAwarePrior] prior_strength={self.config.prior_strength}")

        return adjusted

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable logistic sigmoid."""

        positive_mask = x >= 0
        negative_mask = ~positive_mask
        z = np.empty_like(x, dtype=np.float64)
        z[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))
        exp_x = np.exp(x[negative_mask])
        z[negative_mask] = exp_x / (1.0 + exp_x)
        return z

    def _moving_average(self, values: np.ndarray) -> np.ndarray:
        """Compute a centered moving average with truncated edges."""

        if values.ndim != 2:
            raise ValueError("values must be a 2D array.")
        num_frames = values.shape[0]
        if num_frames == 0:
            return values.copy()
        window = self._window_size
        if window <= 1:
            return values.copy()

        radius = window // 2
        indices = np.arange(num_frames)
        start_indices = np.clip(indices - radius, 0, num_frames)
        end_indices = np.clip(indices + radius + 1, 0, num_frames)

        cumsum = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0)])
        window_sums = cumsum[end_indices] - cumsum[start_indices]
        window_lengths = (end_indices - start_indices).reshape(-1, 1)
        window_lengths = np.maximum(window_lengths, 1)
        return window_sums / window_lengths
