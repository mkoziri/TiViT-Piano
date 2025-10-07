#!/usr/bin/env python3
"""Purpose:
    Remove derived artifacts such as checkpoints, logs, caches, and ``__pycache__``
    directories while protecting source code and dataset assets.

Key Functions/Classes:
    - list_targets(): Builds deletion plans based on presets or explicit toggles
      while respecting dataset exclusions and configuration overrides.
    - delete_groups(): Applies the removal plan with dry-run support and safety
      checks.
    - main(): Parses CLI arguments (e.g., ``--preset`` or ``--keep-best``) and
      coordinates cleanup operations.

CLI:
    Run ``python scripts/clean_tivit.py --project-root <path>`` with presets
    like ``pre-run`` or ``logs``.  Combine ``--dry-run`` and ``--yes`` to
    preview or confirm deletions.
"""
# scripts/clean_tivit.py
#
# TiViT cleaner that mirrors the exact repo layout described in your tivit_structure.txt.
# It removes ONLY known run artifacts:
#   - checkpoints/, logs/, tmp/
#   - data/cache/, data/processed/
#   - src/*/__pycache__/ (data, models, utils)
# …and NEVER touches: configs/, requirements*.txt, notebooks/, scripts/, src/ code, data/omaps tree.

import argparse, shutil, sys, json
from pathlib import Path

# --------- Helpers ---------
def rm_path(p: Path, dry: bool):
    if not p.exists():
        return False
    if dry:
        print(f"[DRY] rm -rf {p}")
        return False
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except IsADirectoryError:
            shutil.rmtree(p, ignore_errors=True)
    return True

def load_cfg_paths(root: Path):
    """Read configs/config.yaml to discover logging dirs if present."""
    cfgp = root / "configs" / "config.yaml"
    log_dir = "logs"
    ckpt_dir = "checkpoints"
    try:
        import yaml  # optional; safe if missing
        if cfgp.exists():
            cfg = yaml.safe_load(cfgp.read_text())
            log_dir = cfg.get("logging", {}).get("log_dir", log_dir)
            ckpt_dir = cfg.get("logging", {}).get("checkpoint_dir", ckpt_dir)
    except Exception:
        pass
    return (root / log_dir).expanduser(), (root / ckpt_dir).expanduser()

def assert_project_layout(root: Path):
    """Sanity check: must look like your TiViT repo root."""
    must_have = [
        root / "configs" / "config.yaml",
        root / "src" / "models" / "tivit_piano.py",
        root / "scripts" / "train.py",
    ]
    missing = [str(p) for p in must_have if not p.exists()]
    if missing:
        print("[WARN] This doesn't look like your TiViT root. Missing:", ", ".join(missing))
        # still proceed

def is_within(child: Path, parent: Path) -> bool:
    try:
        return child.resolve().is_relative_to(parent.resolve())
    except AttributeError:
        # py<3.9 fallback
        c = str(child.resolve())
        p = str(parent.resolve())
        return c.startswith(p.rstrip("/") + "/") or c == p

def list_targets(root: Path, preset: str, keep_best: bool, from_cfg=True):
    """Return the exact paths to delete based on the known structure."""
    # Fixed paths from your structure
    data_cache     = root / "data" / "cache"
    data_processed = root / "data" / "processed"
    tmp_dir        = root / "tmp"

    # pycache dirs exactly where they exist in your tree
    pyc_data   = root / "src" / "data"   / "__pycache__"
    pyc_models = root / "src" / "models" / "__pycache__"
    pyc_utils  = root / "src" / "utils"  / "__pycache__"
    pyc_scripts = root / "scripts" / "__pycache__"  # less common, but just in case

    logs_dir, ckpt_dir = load_cfg_paths(root) if from_cfg else (root / "logs", root / "checkpoints")

    targets = {
        "logs":   [logs_dir],
        "ckpts":  [ckpt_dir],
        "caches": [data_cache, data_processed],
        "tmp":    [tmp_dir],
        "pycache": [pyc_data, pyc_models, pyc_utils, pyc_scripts],
    }

    # When keeping the best checkpoint, replace full dir removal with selective file removals
    if keep_best and ckpt_dir.exists():
        selective_ckpts = []
        for p in ckpt_dir.iterdir():
            name = p.name.lower()
            # Keep anything that looks like a "best" model by common conventions
            if "best" in name and (name.endswith(".pt") or name.endswith(".pth") or name.endswith(".ckpt")):
                continue
            selective_ckpts.append(p)
        targets["ckpts"] = selective_ckpts or [ckpt_dir]  # fall back to dir if enumerate failed

    # Presets
    if preset == "logs":
        groups = {"logs"}
    elif preset == "checkpoints":
        groups = {"ckpts"}
    elif preset == "caches":
        groups = {"caches"}
    elif preset == "tmp":
        groups = {"tmp"}
    elif preset == "pre-run":
        # project as just before first run
        groups = {"logs", "ckpts", "caches", "tmp", "pycache"}
    else:
        groups = set()

    return targets, groups

def delete_groups(root: Path, targets: dict, groups: set, dry: bool):
    removed = []
    dataset_root = root / "data" / "omaps"  # protect any path inside data/omaps
    for g in groups:
        for p in targets.get(g, []):
            p = Path(p)
            # Never touch dataset subtree
            if is_within(p, dataset_root):
                print(f"[SKIP] {p} (inside dataset)")
                continue
            if rm_path(p, dry):
                removed.append(str(p))
    return removed

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser(description="TiViT workspace cleaner (mirrors your exact repo layout).")
    ap.add_argument("--project-root", type=Path, default=Path.cwd(), help="Path to TiViT repo root.")
    ap.add_argument("--dry-run", action="store_true", help="Preview actions only.")
    ap.add_argument("--yes", "-y", action="store_true", help="No prompt.")
    ap.add_argument("--preset",
        choices=["pre-run", "logs", "checkpoints", "caches", "tmp", "custom"],
        default="pre-run",
        help="'pre-run' removes logs, checkpoints, caches, tmp and __pycache__ (leave code & configs intact).")
    ap.add_argument("--keep-best", action="store_true",
        help="When removing checkpoints, keep files with 'best' in their name.")
    # Custom toggles (only used with --preset custom)
    ap.add_argument("--delete-logs", action="store_true")
    ap.add_argument("--delete-checkpoints", action="store_true")
    ap.add_argument("--delete-caches", action="store_true")
    ap.add_argument("--delete-tmp", action="store_true")
    ap.add_argument("--delete-pycache", action="store_true")
    ap.add_argument("--ignore-config-paths", action="store_true",
        help="Do not read configs/config.yaml; assume logs/ and checkpoints/.")
    args = ap.parse_args()

    root = args.project_root.resolve()
    assert_project_layout(root)

    targets, preset_groups = list_targets(
        root=root,
        preset=args.preset,
        keep_best=args.keep_best,
        from_cfg=not args.ignore_config_paths,  # <-- FIXED: underscore attribute
    )

    if args.preset == "custom":
        preset_groups = set()
        if args.delete_logs:        preset_groups.add("logs")
        if args.delete_checkpoints: preset_groups.add("ckpts")
        if args.delete_caches:      preset_groups.add("caches")
        if args.delete_tmp:         preset_groups.add("tmp")
        if args.delete_pycache:     preset_groups.add("pycache")

    if not preset_groups:
        print("[INFO] Nothing selected for deletion. Use --preset or custom toggles.")
        sys.exit(0)

    plan = {g: [str(p) for p in targets.get(g, [])] for g in sorted(preset_groups)}
    print("[PLAN]", json.dumps(plan, indent=2))

    if not args.dry_run and not args.yes:
        resp = input("Proceed? [y/N] ").strip().lower()
        if resp not in {"y", "yes"}:
            print("Aborted.")
            sys.exit(0)

    removed = delete_groups(root, targets, preset_groups, args.dry_run)
    print(f"[DONE] {'(dry-run) ' if args.dry_run else ''}Removed {len(removed)} paths.")

if __name__ == "__main__":
    main()

