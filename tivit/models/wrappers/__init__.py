"""Wrapper entrypoints.

Purpose:
    - Re-export multitask wrapper builders so callers can import from a single
      namespace while the new layout solidifies.
Key Functions/Classes:
    - ``build_model``: Legacy-compatible multitask model constructor.
CLI Arguments:
    (none)
Usage:
    from tivit.models.wrappers import build_model
"""

from .multitask import build_model

__all__ = ["build_model"]
