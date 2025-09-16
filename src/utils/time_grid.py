"""Purpose:
    Convert between seconds and frame indices on a uniform time grid using
    tensor-friendly helper functions.

Key Functions/Classes:
    - sec_to_frame(): Map seconds to frame indices with optional bounds.
    - frame_to_sec(): Convert frame indices back to seconds while supporting
      tensor inputs.

CLI:
    None.  Import these helpers wherever timing conversions are needed.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


Number = Union[int, float]
TensorLike = Union[Number, torch.Tensor]


def sec_to_frame(t: TensorLike, hop_seconds: float, *, offset: int = 0,
                 max_idx: Optional[int] = None) -> TensorLike:
    """Convert time in seconds to a frame index.

    Args:
        t: Time value(s) in seconds.
        hop_seconds: Duration of one frame in seconds.
        offset: Optional frame offset to add after conversion.
        max_idx: If given, clamp the result to ``[0, max_idx]``.

    Returns:
        Frame index as ``int`` or ``torch.Tensor`` matching the input type.
    """
    x = torch.as_tensor(t, dtype=torch.float32)
    k = torch.round(x / float(hop_seconds)).to(torch.int64) + int(offset)
    if max_idx is not None:
        k = torch.clamp(k, 0, int(max_idx))
    return k.item() if not torch.is_tensor(t) else k


def frame_to_sec(k: TensorLike, hop_seconds: float, *, offset: int = 0) -> TensorLike:
    """Convert frame index to time in seconds."""
    x = torch.as_tensor(k, dtype=torch.float32)
    s = (x - int(offset)) * float(hop_seconds)
    return s.item() if not torch.is_tensor(k) else s


__all__ = ["sec_to_frame", "frame_to_sec"]

