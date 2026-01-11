"""Model builders exposed under the new layout.

Purpose:
    - Provide a stable import path for constructing TiViT models from the new package structure.
Key Functions/Classes:
    - ``build_model``: Delegates to :mod:`tivit.models.factory` for registry resolution and legacy-compatible instantiation.
CLI Arguments:
    (none)
Usage:
    from tivit.models import build_model
"""

from .factory import build_model

__all__ = ["build_model"]
