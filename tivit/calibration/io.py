"""Read/write calibration settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def read_calibration(path: str | Path) -> Mapping[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_calibration(payload: Mapping[str, Any], path: str | Path) -> Path:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return p


__all__ = ["read_calibration", "write_calibration"]

