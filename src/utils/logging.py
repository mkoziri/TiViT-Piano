"""Purpose:
    Provide consistent logging helpers for TiViT-Piano scripts and libraries.

Key Functions/Classes:
    - setup_logging(): Configure the root logger with an optional debug level.
    - get_logger(): Convenience wrapper returning namespaced loggers.

CLI:
    None.  Import these helpers from ``src.utils.logging``.
"""

import logging
from typing import Optional

from .logging_utils import configure_verbosity


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging using the legacy debug toggle."""

    level = "debug" if debug else "info"
    configure_verbosity(level)
    return logging.getLogger("tivit")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name if name else "tivit")
