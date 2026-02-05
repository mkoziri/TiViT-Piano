from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Callable

__all__ = [
    "train_single",
    "evaluate",
    "eval_single",
    "autopilot",
    "export_model",
    "calibrate",
    "clean_tivit",
]

if TYPE_CHECKING:
    from .autopilot import autopilot as autopilot
    from .clean_tivit import clean_tivit as clean_tivit
    from .evaluate import evaluate as evaluate
    from .eval_single import eval_single as eval_single
    from .export import export_model as export_model
    from .calibrate import calibrate as calibrate
    from .train_single import train_single as train_single


def __getattr__(name: str) -> Callable:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = {
        "train_single": "tivit.pipelines.train_single",
        "evaluate": "tivit.pipelines.evaluate",
        "eval_single": "tivit.pipelines.eval_single",
        "autopilot": "tivit.pipelines.autopilot",
        "export_model": "tivit.pipelines.export",
        "calibrate": "tivit.pipelines.calibrate",
        "clean_tivit": "tivit.pipelines.clean_tivit",
    }[name]
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
