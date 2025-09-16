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


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure and return a root logger.

    Parameters
    ----------
    debug: bool, optional
        If True, set the log level to ``DEBUG``. Otherwise ``INFO``.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    return logging.getLogger("tivit")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name if name else "tivit")
