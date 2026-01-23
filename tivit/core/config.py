"""Config composition and provenance helpers.

Purpose:
- Load and deep-merge YAML fragments with optional implicit bases.
- Validate allowed namespaces and materialize resolved configs to disk.
- Record git/command provenance for reproducibility.

Key Functions/Classes:
- resolve_config_chain: Merge YAML fragments with base handling.
- load_experiment_config: Entry point for composed experiment configs.
- write_run_artifacts: Persist resolved config, git commit, and command.

CLI Arguments:
- (none; import-only utilities).

Usage:
- Import in pipelines: ``from tivit.core.config import load_experiment_config``.
"""

from __future__ import annotations

import copy
import shlex
import subprocess
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, MutableMapping, Sequence

import yaml

from tivit.utils.fs import atomic_write, atomic_write_text

ConfigLike = Mapping[str, Any]
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default.yaml"
ALLOWED_TOP_LEVEL_KEYS: Collection[str] = {
    "experiment",
    "dataset",
    "model",
    "tiling",
    "training",
    "optim",
    "train",
    "decoder",
    "logging",
    "autopilot",
    "inference",
    "priors",
    "task",
    "calibration",
    "eval",
    "metrics",
}


def _deep_merge(base: MutableMapping[str, Any], overlay: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge mapping ``overlay`` into ``base`` (in-place)."""
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml_file(path: str | Path) -> Mapping[str, Any]:
    """Load YAML mapping from disk, enforcing a top-level mapping."""
    resolved = Path(path).expanduser()
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config at {resolved} must contain a mapping, not {type(data)}")
    return data


def _validate_namespaces(data: Mapping[str, Any], allowed: Collection[str] | None, source: Path) -> None:
    """Validate that ``data`` only contains allowed top-level keys."""
    if allowed is None:
        return
    unknown = [key for key in data.keys() if key not in allowed]
    if unknown:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(f"Config {source} contains unknown top-level keys {unknown} (allowed: {allowed_str})")


def resolve_config_chain(
    paths: Sequence[str | Path],
    *,
    default_base: str | Path | None = None,
    allowed_namespaces: Collection[str] | None = ALLOWED_TOP_LEVEL_KEYS,
) -> Mapping[str, Any]:
    """
    Load and deep-merge a list of YAML configs.

    Each file may optionally declare ``base`` (str or list[str]) to include
    additional YAML fragments before applying its own keys.
    """

    merged: MutableMapping[str, Any] = {}
    seen: set[Path] = set()

    def _apply(path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        raw = dict(load_yaml_file(resolved))
        bases = raw.pop("base", None)
        if bases is None:
            bases = raw.pop("bases", None)
        if bases is None and default_base and resolved != Path(default_base).expanduser().resolve():
            # allow implicit base when provided
            bases = [default_base]
        if isinstance(bases, (str, Path)):
            bases = [bases]
        if isinstance(bases, Iterable):
            for parent in bases:
                _apply(Path(resolved).parent.joinpath(parent))
        _validate_namespaces(raw, allowed_namespaces, resolved)
        _deep_merge(merged, raw)

    for path in paths:
        _apply(path)
    return merged


def load_experiment_config(
    configs: Sequence[str | Path] | None = None,
    *,
    default_base: str | Path | None = DEFAULT_CONFIG_PATH,
) -> Mapping[str, Any]:
    """
    Load and merge experiment configs with an optional implicit base.

    ``default_base`` allows callers to request that configs without an explicit
    ``base``/``bases`` entry are applied on top of the shared default stack
    (for example ``tivit/configs/default.yaml``). Pass ``None`` to disable the
    implicit base when loading a fully self-contained legacy config.
    """

    if not configs:
        configs = [DEFAULT_CONFIG_PATH]
    return resolve_config_chain(list(configs), default_base=default_base, allowed_namespaces=ALLOWED_TOP_LEVEL_KEYS)


def save_resolved_config(cfg: ConfigLike, path: str | Path) -> Path:
    """Persist a merged config to ``path`` and return the resolved path."""
    def _write(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)

    return atomic_write(path, _write)


def _detect_git_commit(repo_root: Path | None = None) -> str:
    """Return the git SHA for provenance; fallback to ``unknown`` on failure."""
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _format_command_line(command: Sequence[str] | str | None) -> str:
    """Render the command line used for the run as a string."""
    if command is None:
        try:
            import sys

            command = sys.argv
        except Exception:
            return ""
    if isinstance(command, str):
        return command.strip()
    return " ".join(shlex.quote(str(part)) for part in command)


def write_run_artifacts(
    cfg: ConfigLike,
    *,
    log_dir: str | Path | None = None,
    command: Sequence[str] | str | None = None,
    configs: Sequence[str | Path] | None = None,
) -> Mapping[str, Path]:
    """
    Persist the merged config and basic provenance for reproducibility.

    Writes ``resolved_config.yaml``, ``git_commit.txt``, and ``command.txt``
    into ``log_dir`` (defaults to cfg['logging']['log_dir'] or ``logs``).
    """

    output_dir = Path(log_dir or (cfg.get("logging", {}) if isinstance(cfg, Mapping) else {}).get("log_dir", "logs")).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["resolved_config"] = save_resolved_config(cfg, output_dir / "resolved_config.yaml")

    commit = _detect_git_commit()
    commit_path = output_dir / "git_commit.txt"
    atomic_write_text(commit_path, f"{commit}\n", encoding="utf-8")
    paths["git_commit"] = commit_path

    cmd_text = _format_command_line(command)
    if configs:
        cfg_list = ", ".join(str(Path(p).expanduser()) for p in configs)
        cmd_text = f"{cmd_text}\nconfigs: {cfg_list}".strip()
    cmd_path = output_dir / "command.txt"
    atomic_write_text(cmd_path, f"{cmd_text}\n", encoding="utf-8")
    paths["command"] = cmd_path

    return paths


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "load_yaml_file",
    "load_experiment_config",
    "resolve_config_chain",
    "save_resolved_config",
    "write_run_artifacts",
]
