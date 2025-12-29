"""Logging helpers for configuring TiViT verbosity levels.

This module centralises the root logging configuration so CLI entry points can
depend on a single ``configure_verbosity`` helper.  The handler uses
``tqdm.contrib.logging.TqdmLoggingHandler`` when available so log messages do
not break active progress bars.
"""

from __future__ import annotations

from typing import Optional

from tivit.utils.logging import QUIET_INFO_FLAG, configure_logging

VERBOSITY_LEVELS = {"quiet", "info", "debug"}


def configure_verbosity(level: Optional[str], env_var: str = "TIVIT_VERBOSE") -> str:
    """Delegate to the shared logging setup while preserving the old API."""

    return configure_logging(level, env_var=env_var, stage_only_console=True)


__all__ = ["configure_verbosity", "VERBOSITY_LEVELS", "QUIET_INFO_FLAG"]
