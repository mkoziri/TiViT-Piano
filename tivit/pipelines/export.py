"""Model export placeholder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from tivit.core.config import load_experiment_config, write_run_artifacts
from tivit.utils.logging import configure_logging, log_final_result, log_stage


def export_model(configs: Sequence[str | Path] | None = None, *, verbose: str | None = "quiet") -> None:
    cfg = load_experiment_config(configs)
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    log_dir = log_cfg.get("log_dir", "logs")
    configure_logging(verbose, log_dir=log_dir, log_file="export.log", stage_only_console=True)
    write_run_artifacts(cfg, log_dir=log_dir, command=sys.argv, configs=configs)
    log_stage("export", "model export stub; integrate with your exporter if needed")
    log_final_result("export", "no-op export finished")


__all__ = ["export_model"]
