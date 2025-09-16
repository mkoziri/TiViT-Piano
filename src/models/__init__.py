"""Purpose:
    Provide canonical entry points for constructing TiViT-Piano models.

Key Functions/Classes:
    - TiViTPiano: Core ViViT-inspired architecture with tiled video encoding
      and multi-task heads.
    - build_model(): Factory that builds models from configuration dictionaries.

CLI:
    Not a CLI module.  Import from scripts such as :mod:`scripts.train` or
    :mod:`scripts.eval_thresholds`.
"""

from .tivit_piano import TiViTPiano
from .factory import build_model

__all__ = ["TiViTPiano", "build_model"]

