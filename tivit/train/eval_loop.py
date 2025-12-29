"""Evaluation loop placeholder."""

from __future__ import annotations

from typing import Callable, Mapping, Any


def run_evaluation(evaluate_fn: Callable[..., Any], *args: object, **kwargs: object) -> Any:
    """Call ``evaluate_fn`` to keep the new namespace stable."""

    return evaluate_fn(*args, **kwargs)


__all__ = ["run_evaluation"]

