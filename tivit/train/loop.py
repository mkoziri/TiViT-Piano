"""Programmatic training entrypoint built on the legacy script."""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from scripts import train as legacy_train
from tivit.core.config import DEFAULT_CONFIG_PATH, load_experiment_config
from tivit.utils.logging import log_final_result, log_stage


@contextmanager
def _argv(args: Sequence[str]):
    original = list(sys.argv)
    sys.argv = [sys.argv[0]] + list(args)
    try:
        yield
    finally:
        sys.argv = original


def _materialize_config(cfg: Mapping[str, object]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(dict(cfg), tmp, sort_keys=False)
    return Path(tmp.name)


def run_training(
    configs: Sequence[str | Path] | None = None,
    *,
    verbose: str | None = "quiet",
    train_split: str | None = None,
    val_split: str | None = None,
    max_clips: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    smoke: bool = False,
) -> None:
    """Run the existing training loop with merged configs."""

    cfg = load_experiment_config(configs)
    cfg_path = _materialize_config(cfg)
    args: list[str] = ["--config", str(cfg_path)]
    if verbose:
        args.extend(["--verbose", verbose])
    if train_split:
        args.extend(["--train-split", train_split])
    if val_split:
        args.extend(["--val-split", val_split])
    if max_clips is not None:
        args.extend(["--max-clips", str(int(max_clips))])
    if frames is not None:
        args.extend(["--frames", str(int(frames))])
    if seed is not None:
        args.extend(["--seed", str(int(seed))])
    if deterministic is not None:
        args.extend(["--deterministic", "true" if deterministic else "false"])
    if smoke:
        args.append("--smoke")

    log_stage("train", f"starting training with configs={configs or [str(DEFAULT_CONFIG_PATH)]}")
    try:
        with _argv(args):
            legacy_train.main()
    except Exception:
        log_final_result("train", "training run failed")
        raise
    finally:
        try:
            cfg_path.unlink()
        except OSError:
            pass
    log_final_result("train", "training run finished")


__all__ = ["run_training"]
