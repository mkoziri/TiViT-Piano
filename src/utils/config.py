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
import yaml

ConfigPath = Union[str, Path]


def load_config(path: ConfigPath = "configs/config.yaml") -> Mapping[str, Any]:
    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

