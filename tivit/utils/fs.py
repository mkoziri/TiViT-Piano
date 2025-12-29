"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write(path: str | Path, write_fn: Callable[[Path], None]) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    write_fn(tmp)
    tmp.replace(target)
    return target


__all__ = ["ensure_dir", "atomic_write"]

