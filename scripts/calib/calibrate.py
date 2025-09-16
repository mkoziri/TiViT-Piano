#!/usr/bin/env python3
"""Purpose:
    Coordinate multi-phase calibration sweeps by editing ``configs/config.yaml``
    and invoking :mod:`scripts.sweep` with curated parameter grids.

Key Functions/Classes:
    - set_yaml(): Helper that updates nested YAML keys in the configuration
      file.
    - run_sweep(): Launches :mod:`scripts.sweep` with PYTHONPATH configured for
      the repository.
    - main(): Parses CLI arguments and orchestrates the staged calibration
      phases.

CLI:
    Run ``python scripts/calibrate.py`` with options such as ``--epochs`` for
    per-run training length, ``--max_clips`` for dataset truncation, and
    ``--skip_bias_toggle`` to omit the final comparison sweep.
"""

import argparse, subprocess, sys, os, time
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Please `pip install pyyaml` first.", file=sys.stderr); sys.exit(1)

REPO = Path(__file__).resolve().parents[1]  # repo root (…/tivit/)
SWEEP = REPO / "scripts" / "sweep.py"
CONFIG = REPO / "configs" / "config.yaml"

def load_cfg():
    with CONFIG.open("r") as f:
        return yaml.safe_load(f)

def save_cfg(cfg):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def set_yaml(path_keys, value):
    """Set a nested YAML key in configs/config.yaml (e.g., ('training','reset_head_bias'), False)."""
    cfg = load_cfg()
    d = cfg
    for k in path_keys[:-1]:
        d = d.setdefault(k, {})
    d[path_keys[-1]] = value
    save_cfg(cfg)

def run_sweep(args_list):
    """Run scripts/sweep.py with a list of CLI args."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(SWEEP)] + args_list
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO), env=env)

def main():
    ap = argparse.ArgumentParser(description="Calibration orchestrator for TiViT-Piano.")
    ap.add_argument("--epochs", type=int, default=2, help="Epochs per run.")
    ap.add_argument("--max_clips", type=int, default=2, help="Dataset clips per split for fast sweeps.")
    ap.add_argument("--results", default="calibration_results.txt", help="Where to append results.")
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M"), help="Tag added to experiment names.")
    ap.add_argument("--skip_bias_toggle", action="store_true", help="Skip the reset_head_bias OFF/ON comparison.")
    args = ap.parse_args()

    # Ensure the base config reflects fast-run defaults (kept modest & consistent across phases)
    # (We do NOT overwrite your other settings; just enforce the basics for the sweep.)
    set_yaml(("dataset","max_clips"), args.max_clips)
    set_yaml(("training","epochs"), args.epochs)
    set_yaml(("training","eval_freq"), 1)
    set_yaml(("training","debug_dummy_labels"), False)
    # default eval threshold used in focal grid phase
    set_yaml(("training","metrics","prob_threshold"), 0.30)
    # default prior and label density starting points
    set_yaml(("training","loss_weights","onoff_prior_mean"), 0.02)
    set_yaml(("training","loss_weights","onoff_prior_weight"), 0.05)
    set_yaml(("dataset","frame_targets","tolerance"), 0.10)
    set_yaml(("dataset","frame_targets","dilate_active_frames"), 1)

    # -------------------------------
    # Phase 1: Focal-loss grid (γ x α)
    # -------------------------------
    run_sweep([
        "--epochs", str(args.epochs),
        "--gamma", "1.5,2.0,3.0",
        "--alpha", "0.05,0.10",
        "--threshold", "0.30",
        "--prior_mean", "0.02",
        "--prior_weight", "0.05",
        "--tolerance", "0.10",
        "--dilate", "1",
        "--max_clips", str(args.max_clips),
        "--results", args.results,
        "--tag", f"phase1_focal_{args.tag}",
    ])

    # ---------------------------------------
    # Phase 2: Threshold sweep (eval only)
    # (Re-run quick evals across thresholds.)
    # ---------------------------------------
    run_sweep([
        "--epochs", str(args.epochs),
        "--gamma", "",            # keep last config γ
        "--alpha", "",            # keep last config α
        "--threshold", "0.20,0.30,0.40,0.50",
        "--prior_mean", "",       # unchanged
        "--prior_weight", "",     # unchanged
        "--tolerance", "",        # unchanged
        "--dilate", "",           # unchanged
        "--max_clips", str(args.max_clips),
        "--results", args.results,
        "--tag", f"phase2_thr_{args.tag}",
    ])

    # ------------------------------------
    # Phase 3: On/Off prior strength sweep
    # ------------------------------------
    run_sweep([
        "--epochs", str(args.epochs),
        "--gamma", "",  "--alpha", "",
        "--threshold", "0.30",
        "--prior_mean", "0.01,0.02",
        "--prior_weight", "0.02,0.05,0.10",
        "--tolerance", "", "--dilate", "",
        "--max_clips", str(args.max_clips),
        "--results", args.results,
        "--tag", f"phase3_prior_{args.tag}",
    ])

    # -----------------------------------------
    # Phase 4: Label density (tolerance/dilate)
    # -----------------------------------------
    run_sweep([
        "--epochs", str(args.epochs),
        "--gamma", "",  "--alpha", "",
        "--threshold", "0.30",
        "--prior_mean", "", "--prior_weight", "",
        "--tolerance", "0.05,0.10",
        "--dilate", "0,1,2",
        "--max_clips", str(args.max_clips),
        "--results", args.results,
        "--tag", f"phase4_labels_{args.tag}",
    ])

    # -------------------------------------------------------
    # Phase 5 (optional): Compare reset_head_bias OFF vs ON
    # We toggle the YAML key and re-run the focal grid quickly
    # -------------------------------------------------------
    if not args.skip_bias_toggle:
        # Turn OFF
        set_yaml(("training","reset_head_bias"), False)
        run_sweep([
            "--epochs", str(args.epochs),
            "--gamma", "1.5,2.0",
            "--alpha", "0.05,0.10",
            "--threshold", "0.30",
            "--prior_mean", "0.02",
            "--prior_weight", "0.05",
            "--tolerance", "0.10",
            "--dilate", "1",
            "--max_clips", str(args.max_clips),
            "--results", args.results,
            "--tag", f"phase5_biasOFF_{args.tag}",
        ])
        # Turn ON again
        set_yaml(("training","reset_head_bias"), True)
        run_sweep([
            "--epochs", str(args.epochs),
            "--gamma", "1.5,2.0",
            "--alpha", "0.05,0.10",
            "--threshold", "0.30",
            "--prior_mean", "0.02",
            "--prior_weight", "0.05",
            "--tolerance", "0.10",
            "--dilate", "1",
            "--max_clips", str(args.max_clips),
            "--results", args.results,
            "--tag", f"phase5_biasON_{args.tag}",
        ])

    print("\nAll calibration sweeps finished.")
    print(f"Results appended to: {args.results}")
    print("Per-run full logs are in sweep_logs/ (created by sweep.py).")

if __name__ == "__main__":
    main()

