"""Purpose:
    Load YAML configuration files with path expansion for TiViT-Piano scripts.

Key Functions/Classes:
    - load_config(): Reads the default ``configs/config.yaml`` (or a provided
      path) and returns a dictionary using ``yaml.safe_load``.

CLI:
    Not a CLI module; imported by training, evaluation, and diagnostic scripts.
"""

from pathlib import Path
from typing import Any, Mapping, Union

from tivit.core.config import load_experiment_config

ConfigPath = Union[str, Path]


def load_config(path: ConfigPath = "configs/config.yaml") -> Mapping[str, Any]:
    """Load a single config file or merge composed fragments."""

    return load_experiment_config([Path(path).expanduser()])
