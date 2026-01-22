"""Shared helpers for pipeline entrypoints.

Purpose:
    - Centralize config/logging setup for pipeline wrappers.
    - Resolve checkpoints and runtime seeds without touching legacy code.
    - Provide lightweight utilities reused across train/eval/export/autopilot.
Key Functions/Classes:
    - prepare_run: load configs, configure logging, and write run artifacts.
    - resolve_eval_split: choose an eval split from config with sensible fallbacks.
    - find_checkpoint: locate an explicit or latest checkpoint path.
    - load_model_weights: load a model state dict from a checkpoint payload.
    - setup_runtime: apply determinism settings and return a device handle.
CLI Arguments:
    (none)
Usage:
    from tivit.pipelines._common import prepare_run, find_checkpoint
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging
from typing import Any, Mapping, MutableMapping, Sequence

import torch

from tivit.core.config import load_experiment_config, write_run_artifacts
from tivit.core.determinism import configure_determinism, resolve_deterministic_flag, resolve_seed
from tivit.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def prepare_run(
    configs: Sequence[str | Path] | None,
    *,
    stage_name: str,
    default_log_file: str,
    verbose: str | None = "quiet",
) -> tuple[MutableMapping[str, Any], Path, Path]:
    """Load config, configure logging, and persist run artifacts."""
    cfg = dict(load_experiment_config(configs))
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, Mapping) else {}
    log_dir = Path(log_cfg.get("log_dir", "logs")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_name = log_cfg.get(f"{stage_name}_log", default_log_file)
    configure_logging(verbose, log_dir=log_dir, log_file=log_file_name, stage_only_console=True)
    write_run_artifacts(cfg, log_dir=log_dir, command=sys.argv, configs=configs)
    return cfg, log_dir, log_dir / log_file_name


def resolve_eval_split(cfg: Mapping[str, Any], split_override: str | None = None) -> str:
    """Choose an evaluation split with fallbacks."""
    if split_override:
        return str(split_override)
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    for key in ("split_val", "split_eval", "split_test", "split"):
        candidate = dataset_cfg.get(key)
        if candidate:
            return str(candidate)
    return "val"


def find_checkpoint(cfg: Mapping[str, Any], checkpoint: str | Path | None = None) -> Path | None:
    """Return an explicit checkpoint or the latest epoch_* file under checkpoint_dir."""
    if checkpoint:
        resolved = Path(checkpoint).expanduser()
        return resolved if resolved.exists() else None
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, Mapping) else {}
    ckpt_dir = Path(log_cfg.get("checkpoint_dir", "./checkpoints")).expanduser()
    if not ckpt_dir.exists():
        return None
    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    candidates = list(ckpt_dir.glob("epoch_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _maybe_init_lazy_encoder(
    model: torch.nn.Module,
    state: Mapping[str, Any],
    device: torch.device,
) -> None:
    if getattr(model, "encoder", None) is not None:
        return
    init_fn = getattr(model, "_init_encoder_if_needed", None)
    if not callable(init_fn):
        return
    if not any(str(key).startswith("encoder.") for key in state.keys()):
        return
    try:
        init_fn(t_tokens=1, s_tokens=1)
    except Exception:
        return
    LOGGER.info("Initialized lazy encoder before checkpoint load.")
    model.to(device)


def load_model_weights(
    model: torch.nn.Module,
    checkpoint: Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> int | None:
    """Load model weights from a checkpoint payload; return epoch if present."""
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model", payload)
    if isinstance(state, Mapping):
        _maybe_init_lazy_encoder(model, state, device)
    model.load_state_dict(state, strict=strict)
    epoch_val = payload.get("epoch")
    try:
        return int(epoch_val)
    except Exception:
        return None


def setup_runtime(
    cfg: Mapping[str, Any],
    *,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> tuple[int, bool, torch.device]:
    """Apply determinism settings and return (seed, deterministic_flag, device)."""
    seed_val = resolve_seed(seed, cfg)
    det_flag = resolve_deterministic_flag(deterministic, cfg, default=True)
    configure_determinism(seed_val, deterministic=det_flag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return seed_val, det_flag, device


__all__ = ["prepare_run", "resolve_eval_split", "find_checkpoint", "load_model_weights", "setup_runtime"]
