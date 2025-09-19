"""Purpose:
    Provide convenient imports for dataset factories used throughout the
    project.  ``make_dataloader`` now routes to either the OMAPS or PianoYT
    backend depending on configuration, while dataset classes remain available
    for advanced tooling.

Key Functions/Classes:
    - OMAPSDataset: Loads tiled piano video clips and aligned label data.
    - PianoYTDataset: Equivalent loader for the PianoYT corpus.
    - make_dataloader(): Builds PyTorch dataloaders configured for clip or
      frame-level supervision.

CLI:
    None.  These helpers are consumed by scripts such as :mod:`scripts.train`.
"""

from .loader import make_dataloader, is_pipeline_v2_enabled
from .omaps_dataset import OMAPSDataset
from .pianoyt_dataset import PianoYTDataset

__all__ = ["make_dataloader", "is_pipeline_v2_enabled", "OMAPSDataset", "PianoYTDataset"]

