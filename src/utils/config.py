"""Purpose:
    Load YAML configuration files with path expansion for TiViT-Piano scripts.

Key Functions/Classes:
    - load_config(): Reads the default ``configs/config.yaml`` (or a provided
      path) and returns a dictionary using ``yaml.safe_load``.

CLI:
    Not a CLI module; imported by training, evaluation, and diagnostic scripts.
"""

from pathlib import Path
import yaml

def load_config(path="configs/config.yaml"):
    p = Path(path).expanduser()
    with p.open("r") as f:
        return yaml.safe_load(f)

