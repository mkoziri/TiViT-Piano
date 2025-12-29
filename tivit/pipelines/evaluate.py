"""Evaluation pipeline placeholder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from tivit.core.config import load_experiment_config, write_run_artifacts
from tivit.utils.logging import configure_logging, log_final_result, log_stage


def evaluate(configs: Sequence[str | Path] | None = None, *, verbose: str | None = "quiet") -> None:
    cfg = load_experiment_config(configs)
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    log_dir = log_cfg.get("log_dir", "logs")
    configure_logging(verbose, log_dir=log_dir, log_file="eval.log", stage_only_console=True)
    write_run_artifacts(cfg, log_dir=log_dir, command=sys.argv, configs=configs)
    log_stage("eval", "evaluation pipeline stub (reuse calibration/eval scripts for full metrics)")
    log_final_result("eval", "no-op evaluate finished")


__all__ = ["evaluate"]
