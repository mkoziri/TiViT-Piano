"""Deterministic runtime helpers for training, evaluation, and calibration.

Purpose:
- Resolve seeds/deterministic flags from CLI or config and apply backend settings.
- Produce reproducible DataLoader seeds/generators scoped by namespace.
- Build metadata fingerprints for caching eval windows.

Key Functions/Classes:
- configure_determinism: Seed Python/NumPy/PyTorch and toggle deterministic backends.
- make_loader_components: Create DataLoader generator/worker_init_fn pairs.
- build_snapshot_metadata: Normalise metadata used for cache fingerprints.

CLI Arguments:
- (none; import-only utilities).

Usage:
- Import in entrypoints: ``from tivit.core.determinism import configure_determinism``.
"""

from __future__ import annotations

import os
import random
import time
import zlib
from functools import partial
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch

DEFAULT_SEED = 1337
_UINT32_MAX = 2 ** 32


def _normalise_metadata_value(value: Any) -> Any:
    """Convert nested metadata into JSON-serialisable primitives."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_normalise_metadata_value(v) for v in value]
    if isinstance(value, Mapping):
        return {str(key): _normalise_metadata_value(val) for key, val in value.items()}
    return str(value)


def resolve_seed(seed_arg: Optional[int], cfg: Optional[Mapping[str, Any]]) -> int:
    """Return the seed resolved via CLI override → config → default."""

    if seed_arg is not None:
        return int(seed_arg)
    if cfg:
        exp_cfg = cfg.get("experiment")
        if isinstance(exp_cfg, Mapping):
            exp_seed = exp_cfg.get("seed")
            if isinstance(exp_seed, (int, float)):
                return int(exp_seed)
        dataset_cfg = cfg.get("dataset")
        if isinstance(dataset_cfg, Mapping):
            dataset_seed = dataset_cfg.get("seed")
            if isinstance(dataset_seed, (int, float)):
                return int(dataset_seed)
        data_cfg = cfg.get("data")
        if isinstance(data_cfg, Mapping):
            data_seed = data_cfg.get("seed")
            if isinstance(data_seed, (int, float)):
                return int(data_seed)
    return DEFAULT_SEED


def resolve_deterministic_flag(
    deterministic_arg: Optional[bool],
    cfg: Optional[Mapping[str, Any]],
    default: bool = True,
) -> bool:
    """Return the deterministic toggle resolved via CLI → config → default."""

    if deterministic_arg is not None:
        return bool(deterministic_arg)
    if cfg:
        exp_cfg = cfg.get("experiment")
        if isinstance(exp_cfg, Mapping):
            exp_det = exp_cfg.get("deterministic")
            if isinstance(exp_det, bool):
                return exp_det
    return bool(default)


def configure_determinism(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU/CUDA) and toggle deterministic backends."""

    seed = int(seed) % _UINT32_MAX
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _derive_namespaced_seed(base_seed: int, namespace: Optional[str]) -> int:
    """Derive a deterministic seed offset using CRC32 of the namespace."""
    if not namespace:
        return base_seed % _UINT32_MAX
    crc = zlib.crc32(namespace.encode("utf-8"), base_seed % _UINT32_MAX)
    return (base_seed + crc) % _UINT32_MAX


def make_torch_generator(base_seed: int, *, namespace: Optional[str] = None) -> torch.Generator:
    """Create a torch.Generator seeded with a namespace-specific derivation."""

    generator = torch.Generator()
    generator.manual_seed(_derive_namespaced_seed(base_seed, namespace))
    return generator


def _seed_worker(worker_id: int, *, base_seed: int) -> None:
    """Seed a DataLoader worker for deterministic shuffling."""
    worker_seed = (base_seed + worker_id) % _UINT32_MAX
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init_fn(base_seed: int, *, namespace: Optional[str] = None) -> Callable[[int], None]:
    """Return a DataLoader worker init function that seeds Python/NumPy/PyTorch."""

    derived = _derive_namespaced_seed(base_seed, namespace)
    return partial(_seed_worker, base_seed=derived)


def make_loader_components(
    base_seed: int,
    *,
    namespace: Optional[str] = None,
) -> tuple[torch.Generator, Callable[[int], None]]:
    """Create generator/worker_init_fn pair for a DataLoader."""

    return (
        make_torch_generator(base_seed, namespace=namespace),
        make_worker_init_fn(base_seed, namespace=namespace),
    )


def build_snapshot_metadata(
    *,
    dataset_name: str,
    split: str,
    seed: int,
    frames: int,
    stride: int,
    max_clips: Optional[int],
    extras: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Assemble normalised metadata used to fingerprint eval-window caches."""

    payload: dict[str, Any] = {
        "dataset": dataset_name,
        "split": split,
        "seed": int(seed),
        "frames": int(frames),
        "stride": int(stride),
        "max_clips": max_clips if max_clips is None else int(max_clips),
        "version": 1,
    }
    if extras:
        payload["extras"] = _normalise_metadata_value(extras)
    return payload
