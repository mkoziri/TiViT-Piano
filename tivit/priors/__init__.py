"""Training-time prior interfaces and factories.

Purpose:
    - Expose Prior interfaces and training-time prior builders.
    - Keep decoding-only priors in tivit.postproc to avoid cross-layer coupling.

Key Functions/Classes:
    - Prior: Minimal interface for target/logit adjustments.
    - build_hand_gating(): Training-time hand-gating prior builder.
    - build_chord_smoothness(): Placeholder smoothing prior builder.

CLI Arguments:
    (none)

Usage:
    from tivit.priors import Prior, build_hand_gating
"""

from .base import Prior, null_prior
from .hand_gating import build_prior as build_hand_gating
from .chord_smoothness import build_prior as build_chord_smoothness

__all__ = [
    "Prior",
    "null_prior",
    "build_hand_gating",
    "build_chord_smoothness",
]
