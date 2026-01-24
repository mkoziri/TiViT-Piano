"""TiViT-Piano single-run evaluation pipeline.

Purpose:
    - Compose configs, configure logging, and delegate to the new eval loop.
    - Expose CLI-friendly overrides for splits, frames, seeds, and smoke tests.
Key Functions/Classes:
    - eval_single: wrapper around ``tivit.pipelines.evaluate.evaluate``.
CLI Arguments:
    - configs: YAML fragments to merge before evaluation.
    - verbose: logging verbosity (quiet|info|debug).
    - split: dataset split override for evaluation.
    - checkpoint: optional checkpoint hint (default: latest).
    - max_batches / max_clips / frames: evaluation caps.
    - seed / deterministic: runtime overrides.
    - smoke: tiny run for quick sanity checks.
Usage:
    python -m tivit.pipelines.eval_single --config tivit/configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

from tivit.pipelines._common import prepare_run
from tivit.pipelines.evaluate import evaluate
from tivit.utils.logging import log_final_result, log_stage


def eval_single(
    configs: Sequence[str | Path] | None = None,
    *,
    verbose: str | None = "quiet",
    split: str | None = None,
    checkpoint: str | Path | None = None,
    max_batches: int | None = None,
    max_clips: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    smoke: bool = False,
) -> Mapping[str, float]:
    """Evaluate TiViT models using the new layout (no legacy dependencies)."""
    prepare_run(configs, stage_name="eval", default_log_file="eval.log", verbose=verbose)
    log_stage("eval", "starting evaluation")
    metrics = evaluate(
        configs=configs,
        verbose=verbose,
        split=split,
        checkpoint=checkpoint,
        max_batches=max_batches,
        max_clips=max_clips,
        frames=frames,
        seed=seed,
        deterministic=deterministic,
        smoke=smoke,
    )
    log_final_result("eval", f"metrics={metrics}")
    return metrics


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run TiViT-Piano evaluation (new stack, no legacy)")
    ap.add_argument("--config", action="append", default=None, help="One or more config fragments to merge")
    ap.add_argument("--verbose", choices=["quiet", "info", "debug"], default="quiet")
    ap.add_argument("--split", help="Dataset split to evaluate (defaults to dataset.split_val/test)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-batches", dest="max_batches", type=int)
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _main() -> None:
    args = _parse_args()
    configs = args.config or [Path("tivit/configs/default.yaml")]
    eval_single(
        configs=configs,
        verbose=args.verbose,
        split=args.split,
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


__all__ = ["eval_single"]
