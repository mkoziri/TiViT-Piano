"""Logging helpers for configuring TiViT verbosity levels.

This module centralises the root logging configuration so CLI entry points can
depend on a single ``configure_verbosity`` helper.  The handler uses
``tqdm.contrib.logging.TqdmLoggingHandler`` when available so log messages do
not break active progress bars.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:  # pragma: no cover - optional dependency guard
    from tqdm.contrib.logging import TqdmLoggingHandler
except Exception:  # pragma: no cover - fallback when tqdm.contrib is missing

    class TqdmLoggingHandler(logging.StreamHandler):
        """Fallback stream handler when tqdm's logging bridge is unavailable."""

        pass


VERBOSITY_LEVELS = {
    "quiet": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

_DEFAULT_FORMAT = "%(asctime)s %(levelname).1s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def _coerce_level_name(level: Optional[str], env_var: str) -> str:
    """Normalize a verbosity string using environment fallback."""

    candidate = (level or "").strip().lower()
    if not candidate:
        candidate = os.environ.get(env_var, "").strip().lower()
    if candidate not in VERBOSITY_LEVELS:
        return "quiet"
    return candidate


def configure_verbosity(level: Optional[str], env_var: str = "TIVIT_VERBOSE") -> str:
    """Configure root logging for the requested verbosity.

    Parameters
    ----------
    level:
        Verbosity name (``quiet``, ``info``, ``debug``).  ``None`` defers to the
        ``env_var`` value.  Invalid strings fall back to ``quiet``.
    env_var:
        Environment variable consulted when ``level`` is ``None``.

    Returns
    -------
    str
        The resolved verbosity token.
    """

    resolved = _coerce_level_name(level, env_var)
    numeric_level = VERBOSITY_LEVELS[resolved]

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = TqdmLoggingHandler()
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)

    root.setLevel(numeric_level)
    logging.captureWarnings(True)

    os.environ[env_var] = resolved

    return resolved


__all__ = ["configure_verbosity", "VERBOSITY_LEVELS"]
