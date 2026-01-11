"""TiViT-Piano evaluation loop utilities."""

# Purpose:
# - Provide a minimal, no-grad evaluation helper for the new training stack.
# - Aggregate loss/part metrics over a dataloader without extra memory overhead.
#
# Key Functions/Classes:
# - run_evaluation: iterate a loader with a user-provided step_fn to return metrics.
#
# CLI Arguments:
# - (none; import-only utilities).
#
# Usage:
# - from tivit.train.eval_loop import run_evaluation

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

import torch


def run_evaluation(
    step_fn: Callable[[Mapping[str, Any]], tuple[torch.Tensor, Mapping[str, Any]]],
    loader: Iterable[Mapping[str, Any]],
    *,
    max_batches: int | None = None,
) -> Mapping[str, Any]:
    """
    Run a no-grad evaluation loop using ``step_fn`` to compute loss/parts.

    ``step_fn`` should accept one batch mapping and return ``(loss_tensor, parts_dict)``.
    """

    total_loss = 0.0
    total_batches = 0
    parts_acc: dict[str, float] = {}

    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            loss_tensor, parts = step_fn(batch)
            loss_val = float(loss_tensor.detach().cpu().item())
            total_loss += loss_val
            total_batches += 1
            for key, value in parts.items():
                try:
                    parts_acc[key] = parts_acc.get(key, 0.0) + float(value)
                except (TypeError, ValueError):
                    continue

    if total_batches == 0:
        return {}

    metrics = {"loss": total_loss / total_batches}
    for key, value in parts_acc.items():
        metrics[key] = value / total_batches
    return metrics


__all__ = ["run_evaluation"]
