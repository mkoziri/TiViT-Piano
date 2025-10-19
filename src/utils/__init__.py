"""Purpose:
    Surface commonly used utility helpers for configuration loading, logging,
    pitch alignment, and time-grid conversions.

Key Functions/Classes:
    - load_config(): Read YAML configuration files with path expansion.
    - align_pitch_dim(): Align target pitch dimensions to match model outputs.
    - setup_logging()/get_logger(): Configure consistent logging namespaces.
    - sec_to_frame()/frame_to_sec(): Convert between seconds and frame indices.

CLI:
    None.  Import the desired helper functions from this namespace.
"""

from .config import load_config
from .identifiers import canonical_video_id
from .pitch import align_pitch_dim
from .logging import setup_logging, get_logger
from .logging_utils import configure_verbosity
from .time_grid import sec_to_frame, frame_to_sec

__all__ = [
    "load_config",
    "canonical_video_id",
    "align_pitch_dim",
    "setup_logging",
    "configure_verbosity",
    "get_logger",
    "sec_to_frame",
    "frame_to_sec",
]
