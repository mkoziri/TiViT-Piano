"""Hand label utilities for TiViT.

New functionality is organized per-source so training/eval code can depend on
small, focused APIs.  PianoVAM hand landmarks are parsed via
``load_pianovam_hand_landmarks`` and aligned to clip frame grids.
"""

from .pianovam_loader import (
    AlignedHandLandmarks,
    load_pianovam_hand_landmarks,
)
from .coordinate_transforms import CanonicalLandmarks, map_landmarks_to_canonical
from .hand_label_builder import (
    EventHandLabelConfig,
    EventHandLabels,
    build_event_hand_labels,
    key_centers_from_geometry,
)
from .hand_reach import HandReachResult, compute_hand_reach

__all__ = [
    "AlignedHandLandmarks",
    "CanonicalLandmarks",
    "EventHandLabelConfig",
    "EventHandLabels",
    "key_centers_from_geometry",
    "build_event_hand_labels",
    "HandReachResult",
    "compute_hand_reach",
    "load_pianovam_hand_landmarks",
    "map_landmarks_to_canonical",
]
