"""Purpose:
    Provide functions for aligning pitch dimensions between model outputs and
    dataset targets while logging helpful diagnostics.

Key Functions/Classes:
    - align_pitch_dim(): Slices or validates pitch dimensions so predictions and
      labels match, with logging describing the applied mapping.

CLI:
    None.  Import this helper within training and evaluation code.
"""

import torch
from typing import Optional

from .logging import get_logger

logger = get_logger(__name__)

def align_pitch_dim(pred: torch.Tensor, target: torch.Tensor, label: str = "", start: int = 21) -> torch.Tensor:
    """
    Align ``target`` pitch dimension to match ``pred``.

    If ``target`` has 128 pitches and ``pred`` has ``P`` (e.g., 88), slice the
    MIDI range ``start``..``start+P-1``.  Raises ``ValueError`` if shapes cannot
    be aligned.  Logs a short a short log message describing the mapping used.
    """
    P_pred = pred.shape[-1]
    P_tgt = target.shape[-1]
    if P_pred == P_tgt:
        print(f"[PITCH] {label}: using {P_pred} pitch bins (no remap)")
        logger.info("%s pitch: using %s pitch bins (no remap)", label, P_pred)
        return target
    if P_tgt == 128 and start + P_pred <= 128:
        end = start + P_pred
        print(f"[PITCH] {label}: slicing MIDI {start}-{end-1} -> {P_pred} bins")
        logger.info("%s pitch: slicing MIDI %s-%s -> %s bins", label, start, end - 1, P_pred)
        return target[..., start:end]
    raise ValueError(
        f"{label} pitch dim mismatch: model={P_pred}, target={P_tgt}; "
        f"expected target 128 or {P_pred} with MIDI {start}-{start+P_pred-1}"
    )

