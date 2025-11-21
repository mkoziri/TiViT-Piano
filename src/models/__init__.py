"""Purpose:
    Provide canonical entry points for constructing TiViT-Piano models.

Key Functions/Classes:
    - TiViTPiano: Core ViViT-inspired architecture with tiled video encoding
      and multi-task heads.
    - ViTSTilePiano: Alternative backend powered by a pretrained ViT-S tile
      encoder plus temporal/tile mixing.
    - build_model(): Factory that builds models from configuration dictionaries.

CLI:
    Not a CLI module.  Import from scripts such as :mod:`scripts.train` or
    :mod:`scripts.eval_thresholds`.
"""

from .tivit_piano import TiViTPiano
from .vits_tile import ViTSTilePiano
from .factory import build_model

__all__ = ["TiViTPiano", "ViTSTilePiano", "build_model"]
