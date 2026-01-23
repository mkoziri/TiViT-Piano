"""Preprocessing utilities for dataset-only pipelines.

Purpose:
    - Provide standalone preprocessing helpers for dataset analysis and caches.
    - Keep preprocessing entrypoints independent from training/runtime codepaths.

Key Functions/Classes:
    - threshold_priors: Dataset-only stats and threshold bound generation.
    - refine_registration_cache: Registration refinement cache builder.

CLI Arguments:
    - (see individual modules)

Usage:
    python -m tivit.preproc.threshold_priors --config tivit/configs/dataset/pianoyt.yaml
"""

__all__ = ["threshold_priors", "refine_registration_cache"]
