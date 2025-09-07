from pathlib import Path
import yaml

def load_config(path="configs/config.yaml"):
    p = Path(path).expanduser()
    with p.open("r") as f:
        return yaml.safe_load(f)

