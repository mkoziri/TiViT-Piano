"""Purpose:
    Provide convenient imports for dataset factories used throughout the
    project.

Key Functions/Classes:
    - OMAPSDataset: Loads tiled piano video clips and aligned label data.
    - make_dataloader(): Builds PyTorch dataloaders configured for clip or
      frame-level supervision.

CLI:
    None.  Use the dataset helpers from other modules such as
    :mod:`scripts.train` or :mod:`scripts.calibrate`.
"""

from .omaps_dataset import make_dataloader, OMAPSDataset

__all__ = ["make_dataloader", "OMAPSDataset"]

