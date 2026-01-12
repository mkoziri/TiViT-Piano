"""TiViT-Piano TorchScript exporter.

Purpose:
    - Load a configured model checkpoint from the new training stack.
    - Trace a TorchScript module with representative input shapes for deployment.
    - Persist the exported artifact alongside run provenance without legacy hooks.
Key Functions/Classes:
    - export_model: build the model, restore weights, and save TorchScript.
CLI Arguments:
    - configs: YAML fragments to merge before export.
    - verbose: logging verbosity (quiet|info|debug).
    - checkpoint: checkpoint path to export (default: latest under logging.checkpoint_dir).
    - output_path: explicit TorchScript destination (default: inference.output_dir/<name>_ts.pt).
    - seed / deterministic: optional runtime overrides applied before export.
Usage:
    python tivit/pipelines/tivit_export.py --config tivit/configs/default.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch

from tivit.models import build_model
from tivit.pipelines._common import find_checkpoint, load_model_weights, prepare_run, setup_runtime
from tivit.utils.logging import log_final_result, log_stage


def _safe_stem(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "tivit_model"


def _build_example_input(cfg: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(dataset_cfg, Mapping):
        dataset_cfg = {}
    frames = int(dataset_cfg.get("frames", 1) or 1)
    channels = int(dataset_cfg.get("channels", 3) or 3)
    resize = dataset_cfg.get("resize") or dataset_cfg.get("canonical_hw") or (224, 224)
    try:
        height, width = int(resize[0]), int(resize[1])
    except Exception:
        height, width = 224, 224
    return torch.randn(1, frames, channels, height, width, device=device, dtype=torch.float32)


def export_model(
    configs: Sequence[str | Path] | None = None,
    *,
    verbose: str | None = "quiet",
    checkpoint: str | Path | None = None,
    output_path: str | Path | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> Path:
    cfg, _, _ = prepare_run(configs, stage_name="export", default_log_file="export.log", verbose=verbose)
    _, det_flag, device = setup_runtime(cfg, seed=seed, deterministic=deterministic)
    log_stage("export", f"building model on device={device} deterministic={det_flag}")

    model = build_model(cfg).to(device)
    ckpt_path = find_checkpoint(cfg, checkpoint)
    if ckpt_path:
        epoch_loaded = load_model_weights(model, ckpt_path, device)
        epoch_str = f"epoch {epoch_loaded}" if epoch_loaded is not None else "unknown epoch"
        log_stage("export", f"loaded checkpoint {ckpt_path} ({epoch_str})")
    else:
        log_stage("export", "no checkpoint found; exporting randomly initialized weights")

    model.eval()
    example_input = _build_example_input(cfg, device)
    with torch.inference_mode():
        scripted = cast(
            torch.jit.ScriptModule | torch.jit.ScriptFunction,
            torch.jit.trace(lambda x: model(x, return_per_tile=False), example_input),
        )

    if output_path is not None:
        export_path = Path(output_path).expanduser()
    else:
        infer_cfg = cfg.get("inference", {}) if isinstance(cfg, Mapping) else {}
        if not isinstance(infer_cfg, Mapping):
            infer_cfg = {}
        log_cfg = cfg.get("logging", {}) if isinstance(cfg, Mapping) else {}
        if not isinstance(log_cfg, Mapping):
            log_cfg = {}
        base_dir = Path(infer_cfg.get("output_dir", Path(log_cfg.get("log_dir", "logs")))).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        experiment_cfg = cfg.get("experiment", {}) if isinstance(cfg, Mapping) else {}
        if not isinstance(experiment_cfg, Mapping):
            experiment_cfg = {}
        run_name = _safe_stem(str(experiment_cfg.get("name", "tivit_model")))
        export_path = base_dir / f"{run_name}_ts.pt"

    export_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(export_path), _extra_files={})
    log_final_result("export", f"TorchScript saved to {export_path}")
    return export_path


__all__ = ["export_model"]
