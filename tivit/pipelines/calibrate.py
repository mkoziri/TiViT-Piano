"""TiViT-Piano calibration pipeline (new stack).

Purpose:
    - Run a fast threshold sweep on the validation split (train fallback).
    - Use dataset-only threshold priors as sweep centers when available.
    - Persist a calibration JSON payload for downstream runs.

Key Functions/Classes:
    - calibrate: orchestrates config loading, checkpoint restore, sweep evaluation.
    - run_threshold_sweep: evaluate per-head threshold sweeps with event F1.

CLI Arguments:
    --config PATH (repeatable): config fragments to merge.
    --checkpoint PATH: explicit checkpoint (default: latest in checkpoint_dir).
    --max-batches INT / --max-clips INT / --frames INT: evaluation caps.
    --seed INT / --deterministic[/-no-deterministic]: runtime overrides.
    --verbose {quiet,info,debug}: logging verbosity.
    --smoke: tiny run for quick sanity checks.

Usage:
    python -m tivit.pipelines.calibrate --config tivit/configs/default.yaml --config tivit/configs/calib/threshold_sweep.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from tivit.calibration.sweep import resolve_calibration_split, run_threshold_sweep
from tivit.calibration.io import write_calibration
from tivit.pipelines._common import prepare_run
from tivit.utils.logging import log_final_result


def calibrate(
    configs: Sequence[str | Path] | None = None,
    *,
    verbose: str | None = "quiet",
    checkpoint: str | Path | None = None,
    max_batches: int | None = None,
    max_clips: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    smoke: bool = False,
) -> Mapping[str, Any]:
    cfg, log_dir, _ = prepare_run(configs, stage_name="calibration", default_log_file="calibration.log", verbose=verbose)
    split = resolve_calibration_split(cfg)
    calibration_cfg = cfg.get("calibration", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(calibration_cfg, Mapping):
        calibration_cfg = {}
    method = str(calibration_cfg.get("method", "threshold_sweep")).lower()
    if method not in {"threshold_sweep", "basic"}:
        raise ValueError(f"Unsupported calibration.method '{method}' (expected threshold_sweep/basic).")

    payload = run_threshold_sweep(
        cfg,
        split=split,
        checkpoint=checkpoint,
        max_batches=max_batches,
        max_clips=max_clips,
        frames=frames,
        seed=seed,
        deterministic=deterministic,
        smoke=smoke,
    )

    output_path = calibration_cfg.get("output_path", "calibration.json")
    out_path = Path(output_path).expanduser()
    if not out_path.is_absolute():
        out_path = log_dir / out_path
    saved_path = write_calibration(payload, out_path)
    log_final_result("calibration", f"calibration saved to {saved_path}")
    return payload


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TiViT-Piano calibration (new stack, no legacy)")
    ap.add_argument("--config", action="append", default=None, help="One or more config fragments to merge")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-batches", type=int)
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="quiet")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _main() -> None:
    args = _parse_args()
    configs = args.config or [Path("tivit/configs/default.yaml")]
    calibrate(
        configs=configs,
        verbose=args.verbose,
        checkpoint=args.checkpoint,
        max_batches=args.max_batches,
        max_clips=args.max_clips,
        frames=args.frames,
        seed=args.seed,
        deterministic=args.deterministic,
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    _main()


__all__ = ["calibrate"]
