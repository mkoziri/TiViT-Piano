"""Timing helpers.

Purpose:
    - Provide lightweight timing utilities for profiling and logging.
    - Offer context managers for elapsed time measurements.

Key Functions/Classes:
    - timer(): Context manager yielding a start timestamp.
    - timed(): Context manager yielding a Timer instance.
    - elapsed_ms(): Convert a start/end pair to milliseconds.
    - Timer: Simple timing dataclass with elapsed helpers.

CLI:
    None. Import these helpers where runtime measurement is needed.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@contextmanager
def timer() -> Iterator[float]:
    start = time.perf_counter()
    yield start


def elapsed_ms(start: float, end: float | None = None) -> float:
    if end is None:
        end = time.perf_counter()
    return (end - start) * 1000.0


@dataclass
class Timer:
    start: float
    end: float | None = None

    def stop(self) -> float:
        self.end = time.perf_counter()
        return self.elapsed_ms()

    def elapsed_ms(self, end: float | None = None) -> float:
        resolved = end if end is not None else (self.end if self.end is not None else time.perf_counter())
        return (resolved - self.start) * 1000.0


@contextmanager
def timed() -> Iterator[Timer]:
    timer = Timer(start=time.perf_counter())
    try:
        yield timer
    finally:
        timer.end = time.perf_counter()


__all__ = ["timer", "elapsed_ms", "Timer", "timed"]
