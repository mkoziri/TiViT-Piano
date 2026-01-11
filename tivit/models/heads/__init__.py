"""Head builders.

Purpose:
    - Re-export task-specific head constructors for registry and factory use.
Key Functions/Classes:
    - ``build_onset_head`` / ``build_offset_head`` / ``build_pitch_head`` / ``build_hand_head`` / ``build_clef_head``: Builders for task heads.
CLI Arguments:
    (none)
Usage:
    from tivit.models.heads import build_onset_head
"""

from .common import MultiLayerHead
from .onset import build_head as build_onset_head
from .offset import build_head as build_offset_head
from .pitch import build_head as build_pitch_head
from .hand import build_head as build_hand_head
from .clef import build_head as build_clef_head

__all__ = [
    "MultiLayerHead",
    "build_onset_head",
    "build_offset_head",
    "build_pitch_head",
    "build_hand_head",
    "build_clef_head",
]
