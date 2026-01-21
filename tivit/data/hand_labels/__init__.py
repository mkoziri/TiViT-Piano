"""Hand label utilities for dataset supervision.

Purpose:
    - Provide loaders and transforms for PianoVAM hand landmarks.
    - Build hand reach masks and event-level hand labels for supervision.

Key Functions/Classes:
    - load_pianovam_hand_landmarks(): Align handskeleton JSON to clip frames.
    - map_landmarks_to_canonical(): Project landmarks into canonical space.
    - compute_hand_reach(): Soft reach mask over keys.

CLI Arguments:
    (none)

Usage:
    from tivit.data.hand_labels import load_pianovam_hand_landmarks
"""

from .pianovam_loader import AlignedHandLandmarks, load_pianovam_hand_landmarks
from .coordinate_transforms import CanonicalLandmarks, map_landmarks_to_canonical
from .hand_label_builder import EventHandLabelConfig, EventHandLabels, build_event_hand_labels, key_centers_from_geometry
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
