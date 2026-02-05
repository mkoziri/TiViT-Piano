"""TiViT workspace cleaner.

Purpose:
    - Remove derived artifacts (checkpoints, logs, caches, __pycache__) safely.
    - Protect dataset assets under common data directories.
    - Provide dry-run and targeted presets for cleanup.
Key Functions/Classes:
    - clean_tivit: build and execute cleanup plan.
CLI Arguments:
    - config: YAML fragments to resolve log/checkpoint dirs (optional).
    - project-root: repo root override.
    - preset: pre-run|logs|checkpoints|caches|pycache|custom.
    - dry-run/yes/keep-best: safety switches.
Usage:
    python -m tivit.pipelines.clean_tivit --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from tivit.core.config import load_experiment_config

PRESETS: dict[str, set[str]] = {
    "pre-run": {"logs", "checkpoints", "caches", "pycache"},
    "logs": {"logs"},
    "checkpoints": {"checkpoints"},
    "caches": {"caches"},
    "pycache": {"pycache"},
    "custom": set(),
}


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except Exception:
        return path.absolute()


def _resolve_project_root(project_root: str | Path | None) -> Path:
    if project_root is None:
        return _safe_resolve(Path(__file__).resolve().parents[2])
    return _safe_resolve(Path(project_root).expanduser())


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return _safe_resolve(path)
    return _safe_resolve(project_root / path)


def _resolve_config_paths(
    configs: Sequence[str | Path] | None,
    project_root: Path,
) -> Sequence[Path] | None:
    if not configs:
        return None
    return [_resolve_path(config, project_root) for config in configs]


def _resolve_logging_dirs(
    project_root: Path,
    configs: Sequence[str | Path] | None,
    *,
    ignore_config: bool,
) -> tuple[Path, Path]:
    if ignore_config:
        return project_root / "logs", project_root / "checkpoints"
    try:
        cfg = load_experiment_config(_resolve_config_paths(configs, project_root))
    except Exception as exc:
        print(f"[WARN] Failed to load configs; using defaults. ({exc})")
        return project_root / "logs", project_root / "checkpoints"
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(log_cfg, Mapping):
        log_cfg = {}
    log_dir = _resolve_path(log_cfg.get("log_dir", "logs"), project_root)
    ckpt_dir = _resolve_path(log_cfg.get("checkpoint_dir", "checkpoints"), project_root)
    return log_dir, ckpt_dir


def _dataset_protected_roots(project_root: Path) -> list[Path]:
    protected = []
    for name in ("data", "data_calib", "splits", "metadata"):
        path = project_root / name
        if path.exists():
            protected.append(_safe_resolve(path))
    return protected


def _find_pycache_dirs(project_root: Path) -> list[Path]:
    roots = [project_root / "tivit", project_root / "src", project_root / "scripts", project_root / "tests"]
    pycache_dirs: list[Path] = []
    seen: set[Path] = set()
    root_pycache = project_root / "__pycache__"
    if root_pycache.exists():
        resolved = _safe_resolve(root_pycache)
        pycache_dirs.append(root_pycache)
        seen.add(resolved)
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("__pycache__"):
            if not candidate.is_dir():
                continue
            resolved = _safe_resolve(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            pycache_dirs.append(candidate)
    pycache_dirs.sort(key=lambda path: str(path))
    return pycache_dirs


def _selective_checkpoint_targets(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return [checkpoint_dir]
    selective: list[Path] = []
    for entry in checkpoint_dir.iterdir():
        name = entry.name.lower()
        if entry.is_file() and "best" in name and entry.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            continue
        selective.append(entry)
    return selective or [checkpoint_dir]


def _build_targets(
    project_root: Path,
    configs: Sequence[str | Path] | None,
    *,
    ignore_config: bool,
    keep_best: bool,
) -> dict[str, list[Path]]:
    log_dir, ckpt_dir = _resolve_logging_dirs(project_root, configs, ignore_config=ignore_config)
    data_cache = project_root / "data" / "cache"
    data_processed = project_root / "data" / "processed"
    pycache_dirs = _find_pycache_dirs(project_root)
    targets = {
        "logs": [log_dir],
        "checkpoints": [ckpt_dir],
        "caches": [data_cache, data_processed],
        "pycache": pycache_dirs,
    }
    if keep_best:
        targets["checkpoints"] = _selective_checkpoint_targets(ckpt_dir)
    return targets


def _is_within(path: Path, root: Path) -> bool:
    try:
        return _safe_resolve(path).is_relative_to(_safe_resolve(root))
    except AttributeError:
        path_str = str(_safe_resolve(path))
        root_str = str(_safe_resolve(root))
        return path_str.startswith(root_str.rstrip("/") + "/") or path_str == root_str


def _is_allowed_target(
    path: Path,
    project_root: Path,
    protected_roots: Iterable[Path],
    allowlist_roots: Iterable[Path],
    *,
    allow_outside_root: bool,
) -> tuple[bool, str]:
    if not allow_outside_root and not _is_within(path, project_root):
        return False, "outside project root"
    for protected in protected_roots:
        if _is_within(path, protected):
            if any(_is_within(path, allowed) for allowed in allowlist_roots):
                return True, ""
            return False, "inside protected dataset root"
    return True, ""


def _remove_path(path: Path, *, dry_run: bool) -> bool:
    if not path.exists():
        return False
    if dry_run:
        print(f"[DRY] rm -rf {path}")
        return False
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except IsADirectoryError:
            shutil.rmtree(path, ignore_errors=True)
    return True


def _apply_plan(
    project_root: Path,
    targets: Mapping[str, Sequence[Path]],
    groups: Iterable[str],
    *,
    dry_run: bool,
    allow_outside_root: bool,
) -> list[Path]:
    removed: list[Path] = []
    protected_roots = _dataset_protected_roots(project_root)
    allowlist_roots = [project_root / "data" / "cache", project_root / "data" / "processed"]
    seen: set[Path] = set()
    for group in groups:
        for target in targets.get(group, []):
            resolved = _safe_resolve(target)
            if resolved in seen:
                continue
            seen.add(resolved)
            allowed, reason = _is_allowed_target(
                resolved,
                project_root,
                protected_roots,
                allowlist_roots,
                allow_outside_root=allow_outside_root,
            )
            if not allowed:
                print(f"[SKIP] {resolved} ({reason})")
                continue
            if _remove_path(resolved, dry_run=dry_run):
                removed.append(resolved)
    return removed


def clean_tivit(
    configs: Sequence[str | Path] | None = None,
    *,
    project_root: str | Path | None = None,
    preset: str = "pre-run",
    keep_best: bool = False,
    dry_run: bool = False,
    assume_yes: bool = False,
    ignore_config: bool = False,
    allow_outside_root: bool = False,
    delete_logs: bool = False,
    delete_checkpoints: bool = False,
    delete_caches: bool = False,
    delete_pycache: bool = False,
) -> list[Path]:
    resolved_root = _resolve_project_root(project_root)
    targets = _build_targets(
        resolved_root,
        configs,
        ignore_config=ignore_config,
        keep_best=keep_best,
    )
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Expected {sorted(PRESETS)}")
    groups = set(PRESETS[preset])
    if preset == "custom":
        groups = set()
        if delete_logs:
            groups.add("logs")
        if delete_checkpoints:
            groups.add("checkpoints")
        if delete_caches:
            groups.add("caches")
        if delete_pycache:
            groups.add("pycache")

    if not groups:
        print("[INFO] Nothing selected for deletion. Use --preset or custom toggles.")
        return []

    plan = {group: [str(path) for path in targets.get(group, [])] for group in sorted(groups)}
    print("[PLAN]", json.dumps(plan, indent=2))

    if not dry_run and not assume_yes:
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp not in {"y", "yes"}:
            print("Aborted.")
            return []

    removed = _apply_plan(
        resolved_root,
        targets,
        groups,
        dry_run=dry_run,
        allow_outside_root=allow_outside_root,
    )
    suffix = "(dry-run) " if dry_run else ""
    print(f"[DONE] {suffix}Removed {len(removed)} paths.")
    return removed


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean TiViT workspace artifacts (new stack)")
    ap.add_argument("--config", action="append", default=None, help="Config fragments to resolve log/checkpoint dirs")
    ap.add_argument("--project-root", type=Path, default=None, help="Path to TiViT repo root")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="pre-run")
    ap.add_argument("--dry-run", action="store_true", help="Preview actions only")
    ap.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    ap.add_argument("--keep-best", action="store_true", help="Keep checkpoints with 'best' in their name")
    ap.add_argument("--ignore-config", action="store_true", help="Skip config loading; assume logs/ and checkpoints/")
    ap.add_argument("--allow-outside-root", action="store_true", help="Allow deleting paths outside project root")
    ap.add_argument("--delete-logs", action="store_true", help="Custom: remove logs")
    ap.add_argument("--delete-checkpoints", action="store_true", help="Custom: remove checkpoints")
    ap.add_argument("--delete-caches", action="store_true", help="Custom: remove caches")
    ap.add_argument("--delete-pycache", action="store_true", help="Custom: remove __pycache__ dirs")
    return ap.parse_args()


def _main() -> None:
    args = _parse_args()
    clean_tivit(
        configs=args.config,
        project_root=args.project_root,
        preset=args.preset,
        keep_best=bool(args.keep_best),
        dry_run=bool(args.dry_run),
        assume_yes=bool(args.yes),
        ignore_config=bool(args.ignore_config),
        allow_outside_root=bool(args.allow_outside_root),
        delete_logs=bool(args.delete_logs),
        delete_checkpoints=bool(args.delete_checkpoints),
        delete_caches=bool(args.delete_caches),
        delete_pycache=bool(args.delete_pycache),
    )


if __name__ == "__main__":
    _main()


__all__ = ["clean_tivit"]
