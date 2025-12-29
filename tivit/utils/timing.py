"""Timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[float]:
    start = time.perf_counter()
    yield start


def elapsed_ms(start: float, end: float | None = None) -> float:
    if end is None:
        end = time.perf_counter()
    return (end - start) * 1000.0


__all__ = ["timer", "elapsed_ms"]
