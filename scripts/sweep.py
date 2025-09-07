#!/usr/bin/env python3
# scripts/sweep.py
import argparse
import hashlib
import itertools
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Please `pip install pyyaml` first.", file=sys.stderr)
    sys.exit(1)

VAL_LINE_RE = re.compile(r"^Val:\s*(.*)$")

# ---------- small helpers ----------
def load_yaml(p: Path):
    with p.open("r") as f:
        return yaml.safe_load(f)

def dump_yaml(cfg: dict, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def slug(s: str, max_len=80):
    """Filesystem-safe short slug."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(s))
    return s[:max_len]

def short_id(combo: str):
    """Short stable id from param combo."""
    return hashlib.md5(combo.encode("utf-8")).hexdigest()[:8]

def parse_list(s: str, cast=float):
    """Parse comma-separated list; empty -> [None]."""
    if not s:
        return [None]
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(cast(tok))
    return out or [None]

def base_from_config_name(name: str) -> str:
    """
    Strip any previous sweep suffix from a config experiment name.
    Examples:
      'TiViT-Piano_frame_v1' -> same
      'TiViT-Piano_frame_v1_sw_17f9ab0c_focal_sweep' -> 'TiViT-Piano_frame_v1'
      'proj_sw_aaaa1111_xxx_sw_bbbb2222' -> 'proj'
    """
    if not name:
        return "TiViT"
    # remove everything from the first occurrence of '_sw_' onward
    m = re.split(r"_sw_[0-9a-f]{8}.*", name, maxsplit=1)
    base = m[0]
    # also trim any trailing underscores
    return base.rstrip("_") or "TiViT"

def run_train(python_bin: str, train_script: Path, workdir: Path, env_extra: dict):
    """
    Run training and capture LAST 'Val:' line from stdout.
    Returns: (stdout_lines, last_val_line_str_or_None, return_code)
    """
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    proc = subprocess.Popen(
        [python_bin, str(train_script)],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    stdout_lines = []
    last_val = None
    for line in proc.stdout:
        sys.stdout.write(line)  # live mirror
        stdout_lines.append(line.rstrip("\n"))
        m = VAL_LINE_RE.match(line.strip())
        if m:
            last_val = m.group(1)
    proc.wait()
    return stdout_lines, last_val, proc.returncode

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Edit YAML, run training, collect last 'Val:' line.")
    ap.add_argument("--repo", default=".", help="Project root (where configs/, scripts/, src/ live).")
    ap.add_argument("--python", default=sys.executable, help="Python executable.")
    ap.add_argument("--base_config", default="configs/config.yaml", help="Config path relative to repo.")
    ap.add_argument("--results", default="sweep_results.txt", help="Output results file (relative to repo).")
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs for quick sweeps.")
    ap.add_argument("--base_exp_name", default="", help="Optional: override base experiment name (else derive from config).")

    # Sweep knobs (comma-separated lists). Leave blank to keep base value.
    ap.add_argument("--gamma", default="", help="focal_gamma list, e.g. '1.5,2.0,3.0'")
    ap.add_argument("--alpha", default="", help="focal_alpha list, e.g. '0.05,0.10'")
    ap.add_argument("--threshold", default="", help="prob_threshold list, e.g. '0.2,0.3,0.4'")
    ap.add_argument("--prior_mean", default="", help="onoff_prior_mean list, e.g. '0.01,0.02,0.05'")
    ap.add_argument("--prior_weight", default="", help="onoff_prior_weight list, e.g. '0.02,0.05,0.10'")
    ap.add_argument("--tolerance", default="", help="frame_targets.tolerance list, e.g. '0.05,0.10'")
    ap.add_argument("--dilate", default="", help="frame_targets.dilate_active_frames list, e.g. '0,1,2'")
    ap.add_argument("--max_clips", default="", help="dataset.max_clips list, e.g. '2,4'")
    ap.add_argument("--tag", default="", help="Optional tag to include in experiment name/logs.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    train_py = repo / "scripts" / "train.py"         # your layout: scripts/train.py
    conf_path = repo / args.base_config               # configs/config.yaml
    results_path = repo / args.results

    if not train_py.exists():
        print(f"ERROR: {train_py} not found.", file=sys.stderr)
        sys.exit(1)
    if not conf_path.exists():
        print(f"ERROR: {conf_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Load base config ONCE to capture (and sanitize) the original experiment base name
    base_cfg = load_yaml(conf_path)
    cfg_name = (base_cfg.get("experiment", {}) or {}).get("name", "TiViT")
    if args.base_exp_name.strip():
        BASE_EXP_NAME = args.base_exp_name.strip()
    else:
        BASE_EXP_NAME = base_from_config_name(cfg_name)

    # Parse lists
    gamma_list  = parse_list(args.gamma, float)
    alpha_list  = parse_list(args.alpha, float)
    thr_list    = parse_list(args.threshold, float)
    pmean_list  = parse_list(args.prior_mean, float)
    pwt_list    = parse_list(args.prior_weight, float)
    tol_list    = parse_list(args.tolerance, float)
    dil_list    = parse_list(args.dilate, int)
    mclips_list = parse_list(args.max_clips, int)

    # Results header
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        with results_path.open("w") as f:
            f.write("# TiViT sweep results\n")
            f.write("# iso8601\tgamma\talpha\tthr\tprior_mean\tprior_wt\ttol\tdilate\tmax_clips\texp\tval_line\tretcode\n")

    # Ensure PYTHONPATH includes src/ so imports resolve
    env_extra = {"PYTHONPATH": str(repo / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}

    for (gamma, alpha, thr, pmean, pwt, tol, dil, mclips) in itertools.product(
        gamma_list, alpha_list, thr_list, pmean_list, pwt_list, tol_list, dil_list, mclips_list
    ):
        # Fresh load each iteration so we don't accumulate previous edits
        cfg = load_yaml(conf_path)

        # Edit YAML keys only when a value is provided
        tr = cfg.setdefault("training", {})
        lw = tr.setdefault("loss_weights", {})
        mt = tr.setdefault("metrics", {})
        ds = cfg.setdefault("dataset", {})
        ft = ds.setdefault("frame_targets", {})

        if args.epochs is not None:
            tr["epochs"] = int(args.epochs)
        if gamma is not None:
            lw["focal_gamma"] = float(gamma)
        if alpha is not None:
            lw["focal_alpha"] = float(alpha)
        if thr is not None:
            mt["prob_threshold"] = float(thr)
        if pmean is not None:
            lw["onoff_prior_mean"] = float(pmean)
        if pwt is not None:
            lw["onoff_prior_weight"] = float(pwt)
        if tol is not None:
            ft["tolerance"] = float(tol)
        if dil is not None:
            ft["dilate_active_frames"] = int(dil)
        if mclips is not None:
            ds["max_clips"] = int(mclips)

        # Build compact, stable experiment name: sanitized base + short hash + optional tag
        combo = f"g{gamma}_a{alpha}_thr{thr}_pm{pmean}_pw{pwt}_tol{tol}_dil{dil}_mc{mclips}"
        run_id = short_id(combo)
        tag = (args.tag.strip() or "")
        exp_name = f"{BASE_EXP_NAME}_sw_{run_id}{('_'+slug(tag,20)) if tag else ''}"

        # Override experiment.name with the SHORT name (prevents path growth)
        cfg.setdefault("experiment", {})["name"] = exp_name

        # Write config for train.py to consume
        dump_yaml(cfg, conf_path)

        # Launch training
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S")
        stdout, val_line, ret = run_train(args.python, train_py, repo, env_extra=env_extra)

        # Append concise row to results (keep full combo via columns + store the short exp name)
        with results_path.open("a") as f:
            f.write(
                f"{t0}\t{gamma}\t{alpha}\t{thr}\t{pmean}\t{pwt}\t{tol}\t{dil}\t{mclips}\t{exp_name}\t{(val_line or '').strip()}\t{ret}\n"
            )

        # Save full stdout per-run with a short filename
        logs_dir = repo / "sweep_logs"
        logs_dir.mkdir(exist_ok=True)
        logfile = logs_dir / f"{slug(exp_name, 60)}.log"
        with logfile.open("w") as lf:
            lf.write("\n".join(stdout))

    print(f"\nDone. Results saved to: {results_path}")
    print("Per-run logs in: sweep_logs/")

if __name__ == "__main__":
    main()

