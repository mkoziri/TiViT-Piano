"""Filesystem helpers.

Purpose:
    - Provide small, reusable helpers for safe filesystem writes.
    - Centralize directory creation and atomic write patterns.

Key Functions/Classes:
    - ensure_dir(): Create a directory path (parents=True, exist_ok=True).
    - atomic_write(): Write via a temp file and rename into place.
    - atomic_write_text(): Convenience wrapper for text payloads.
    - atomic_write_json(): Convenience wrapper for JSON payloads.

CLI:
    None. Import these helpers where safe writes are needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


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


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text to ``path`` using an atomic temp file swap."""

    def _write(tmp: Path) -> None:
        tmp.write_text(text, encoding=encoding)

    return atomic_write(path, _write)


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> Path:
    """Write JSON to ``path`` using an atomic temp file swap."""

    def _write(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys, ensure_ascii=True)
            handle.write("\n")

    return atomic_write(path, _write)


__all__ = ["ensure_dir", "atomic_write", "atomic_write_text", "atomic_write_json"]
