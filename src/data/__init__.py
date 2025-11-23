"""Purpose:
    Provide convenient imports for dataset factories used throughout the
    project.  ``make_dataloader`` now routes to either the OMAPS or PianoYT
    backend depending on configuration, while dataset classes remain available
    for advanced tooling.

Key Functions/Classes:
    - OMAPSDataset: Loads tiled piano video clips and aligned label data.
    - PianoYTDataset: Equivalent loader for the PianoYT corpus.
    - PianoVAMDataset: Dataset that mirrors PianoYTDataset but reads PianoVAM assets.
    - make_dataloader(): Builds PyTorch dataloaders configured for clip or
      frame-level supervision.

CLI:
    None.  These helpers are consumed by scripts such as :mod:`scripts.train`.
"""

from .loader import make_dataloader
from .omaps_dataset import OMAPSDataset
from .pianoyt_dataset import PianoYTDataset
from .pianovam_dataset import PianoVAMDataset

__all__ = ["make_dataloader", "OMAPSDataset", "PianoYTDataset"]

