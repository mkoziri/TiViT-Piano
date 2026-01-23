"""File-first logging with quiet console output.

Purpose:
    - Configure a consistent logging policy for TiViT pipelines.
    - Keep file logs verbose while keeping console logs quiet and structured.

Key Functions/Classes:
    - configure_logging(): Configure root handlers and per-module levels.
    - get_logger(): Return a namespaced logger.
    - log_stage()/log_final_result(): Emit stage and final markers.
    - StageFilter: Permit only stage/progress logs on the console.

CLI:
    None. Import from pipeline entrypoints or training loops.
"""

from __future__ import annotations

import logging
import os
from importlib import import_module
from pathlib import Path
from typing import Optional

QUIET_INFO_FLAG = "tivit_force_info"

try:  # pragma: no cover - optional dependency guard
    _tqdm_logging = import_module("tqdm.contrib.logging")
    _TQDM_HANDLER = getattr(_tqdm_logging, "TqdmLoggingHandler", None)
except Exception:  # pragma: no cover
    _TQDM_HANDLER = None


class StageFilter(logging.Filter):
    """Allow only stage/final/progress logs through the console handler."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple predicate
        if record.levelno >= logging.WARNING:
            return True
        if getattr(record, "stage", False) or getattr(record, "is_final", False) or getattr(record, "progress_only", False):
            return True
        return bool(getattr(record, QUIET_INFO_FLAG, False))


def _coerce_level_name(level: Optional[str], env_var: str) -> str:
    candidate = (level or "").strip().lower()
    if not candidate:
        candidate = os.environ.get(env_var, "").strip().lower()
    if candidate not in {"quiet", "info", "debug"}:
        return "quiet"
    return candidate


def configure_logging(
    level: Optional[str],
    *,
    env_var: str = "TIVIT_VERBOSE",
    log_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    stage_only_console: bool = True,
) -> str:
    resolved = _coerce_level_name(level, env_var)
    numeric_level = {"quiet": logging.INFO, "info": logging.INFO, "debug": logging.DEBUG}[resolved]

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    log_dir_path = Path(log_dir or os.environ.get("TIVIT_LOG_DIR", "logs")).expanduser()
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / (Path(log_file).name if log_file else "tivit.log")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(file_handler)

    handler_cls = _TQDM_HANDLER or logging.StreamHandler
    console_handler = handler_cls()
    console_handler.setLevel(logging.INFO if resolved == "quiet" else numeric_level)
    console_handler.setFormatter(logging.Formatter("%(levelname).1s %(name)s: %(message)s"))
    if stage_only_console:
        console_handler.addFilter(StageFilter())
        root.setLevel(logging.DEBUG)
    else:
        root.setLevel(numeric_level)
    root.addHandler(console_handler)

    logging.captureWarnings(True)
    os.environ[env_var] = resolved
    # Silence heavy ROI debug blocks unless explicitly running in debug mode.
    logging.getLogger("tivit.data.roi.keyboard_roi").setLevel(
        logging.DEBUG if resolved == "debug" else logging.INFO
    )
    return resolved


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else "tivit")


def log_stage(stage: str, message: str) -> None:
    logger = get_logger(f"tivit.stage.{stage}")
    logger.info(message, extra={"stage": True})


def log_final_result(stage: str, message: str) -> None:
    logger = get_logger(f"tivit.final.{stage}")
    logger.info(message, extra={"is_final": True})


__all__ = ["configure_logging", "get_logger", "log_stage", "log_final_result", "QUIET_INFO_FLAG", "StageFilter"]
