#!/usr/bin/env python3
"""Automated training/calibration driver for TiViT-Piano.
""Purpose:
    This script coordinates short training bursts with calibration/evaluation
rounds.  It keeps a ledger of results, updates ``configs/config.yaml`` with the
latest thresholds, and mirrors stdout from the invoked helpers.

Key Functions/Classes:
    To be added

CLI:
    To be added
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - defensive guard for missing dep
    print("Please `pip install pyyaml` before running train_autopilot.py", file=sys.stderr)
    raise

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "config.yaml"
TRAIN = REPO / "scripts" / "train.py"
CALIBRATE_THRESH = REPO / "scripts" / "calib" / "calibrate_thresholds.py"
EVAL_THRESH = REPO / "scripts" / "calib" / "eval_thresholds.py"
PARSE_SWEEP = REPO / "scripts" / "calib" / "parse_sweep.py"

VAL_LINE_RE = re.compile(r"^Val:\s*(.*)$")
TABLE_HEADER_RE = re.compile(r"^onset_thr\t")

DEFAULT_RESULTS = REPO / "runs" / "auto" / "results.txt"
DEFAULT_STDOUT_DIR = REPO / "runs" / "auto"
CALIB_JSON = REPO / "calibration.json"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_cfg() -> dict:
    with CONFIG.open("r") as f:
        return yaml.safe_load(f)


def save_cfg(cfg: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def ensure_default(cfg: dict, keys: Iterable[str], value) -> bool:
    cur = cfg
    keys = list(keys)
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    if keys[-1] not in cur:
        cur[keys[-1]] = value
        return True
    return False


# ---------------------------------------------------------------------------
# Experiment naming helpers (mirrors scripts/calib/sweep.py)
# ---------------------------------------------------------------------------

def base_from_config_name(name: str) -> str:
    if not name:
        return "TiViT"
    parts = re.split(r"_sw_[0-9a-f]{8}.*", name, maxsplit=1)
    base = parts[0]
    return base.rstrip("_") or "TiViT"


def short_id(seed: str) -> str:
    import hashlib

    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Filesystem/log helpers
# ---------------------------------------------------------------------------

def ensure_results_header(path: Path, header_cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w") as f:
        f.write("# TiViT autopilot results\n")
        f.write("# columns:\t" + "\t".join(header_cols) + "\n")


def log_banner(results_path: Path, message: str) -> None:
    bold = f"\n\033[1m{message}\033[0m"
    print(bold)
    with results_path.open("a") as f:
        f.write(f"# {message}\n")


def run_command(cmd: List[str], log_file: Path, capture_last_val: bool = False) -> Tuple[int, Optional[str], List[str]]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(">>>", " ".join(cmd))
    last_val = None
    lines: List[str] = []
    with log_file.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
            lines.append(line.rstrip("\n"))
            if capture_last_val:
                m = VAL_LINE_RE.match(line.strip())
                if m:
                    last_val = m.group(1)
        proc.wait()
        ret = proc.returncode
    return ret, last_val, lines


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_ckpt(ckpt_dir: Path) -> Optional[Path]:
    best = ckpt_dir / "tivit_best.pt"
    if best.exists():
        return best
    last = ckpt_dir / "tivit_last.pt"
    if last.exists():
        return last
    epochs = sorted(ckpt_dir.glob("tivit_epoch_*.pt"), reverse=True)
    if epochs:
        return epochs[0]
    return None


def sync_last_to_best(ckpt_dir: Path) -> None:
    last = ckpt_dir / "tivit_last.pt"
    best = ckpt_dir / "tivit_best.pt"
    if last.exists():
        shutil.copy2(last, best)


# ---------------------------------------------------------------------------
# Calibration / evaluation helpers
# ---------------------------------------------------------------------------

def run_calibration(kind: str, ckpt: Path, log_dir: Path, split: str, max_clips: Optional[int], frames: Optional[int]) -> int:
    log_name = f"calibration_{kind}.txt"
    log_path = log_dir / log_name
    cmd = [sys.executable, str(CALIBRATE_THRESH), "--ckpt", str(ckpt)]
    if split:
        cmd.extend(["--split", split])
    if max_clips is not None:
        cmd.extend(["--max-clips", str(max_clips)])
    if frames is not None:
        cmd.extend(["--frames", str(frames)])
    ret, _, _ = run_command(cmd, log_path, capture_last_val=False)
    return ret


def run_fast_eval(
    ckpt: Path,
    log_dir: Path,
    split: str,
    calibration_json: Path,
    extra_probs: Optional[Tuple[float, float, float]] = None,
    temperature: Optional[float] = None,
    bias: Optional[float] = None,
) -> Tuple[int, List[str]]:
    log_path = log_dir / "eval_fast.txt"
    cmd = [
        sys.executable,
        str(EVAL_THRESH),
        "--ckpt",
        str(ckpt),
        "--calibration",
        str(calibration_json),
    ]
    if split:
        cmd.extend(["--split", split])
    if extra_probs is not None:
        probs = [max(0.0, min(1.0, p)) for p in extra_probs]
        cmd.append("--prob_thresholds")
        cmd.extend([f"{p:.3f}" for p in probs])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if bias is not None:
        cmd.extend(["--bias", str(bias)])
    ret, _, lines = run_command(cmd, log_path, capture_last_val=False)
    return ret, lines


def parse_eval_table(lines: List[str]) -> Optional[Dict[str, float]]:
    header_idx = None
    for i, line in enumerate(lines):
        if TABLE_HEADER_RE.match(line.strip()):
            header_idx = i
            break
    if header_idx is None:
        return None
    best = None
    for line in lines[header_idx + 1 :]:
        if not line.strip() or line.startswith("["):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 8:
            continue
        try:
            onset_thr = float(parts[0])
            offset_thr = float(parts[1])
            onset_f1 = float(parts[2])
            offset_f1 = float(parts[3])
            onset_pr = float(parts[4])
            onset_pos = float(parts[5])
            onset_ev = float(parts[6])
            offset_ev = float(parts[7])
        except ValueError:
            continue
        ev_mean = 0.5 * (onset_ev + offset_ev)
        row = {
            "onset_thr": onset_thr,
            "offset_thr": offset_thr,
            "onset_f1": onset_f1,
            "offset_f1": offset_f1,
            "onset_pred_rate": onset_pr,
            "onset_pos_rate": onset_pos,
            "onset_event_f1": onset_ev,
            "offset_event_f1": offset_ev,
            "ev_f1_mean": ev_mean,
        }
        if best is None or row["ev_f1_mean"] > best["ev_f1_mean"] + 1e-9:
            best = row
    return best


def load_calibration(calibration_json: Path) -> Optional[dict]:
    if not calibration_json.exists():
        return None
    with calibration_json.open("r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------
RESULT_HEADER = [
    "iso8601",
    "round",
    "burst_epochs",
    "ckpt",
    "onset_thr",
    "offset_thr",
    "onset_f1",
    "offset_f1",
    "onset_ev_f1",
    "offset_ev_f1",
    "ev_f1_mean",
    "patience",
    "exp",
    "val_line",
    "retcode",
]


def format_val_line(metrics: Dict[str, float], train_val: Optional[str]) -> str:
    bits = []
    if train_val:
        bits.append(train_val.strip())
    if metrics:
        bits.append(f"onset_f1={metrics['onset_f1']:.3f}")
        bits.append(f"offset_f1={metrics['offset_f1']:.3f}")
        bits.append(f"onset_event_f1={metrics['onset_event_f1']:.3f}")
        bits.append(f"offset_event_f1={metrics['offset_event_f1']:.3f}")
        bits.append(f"onset_pred_rate={metrics['onset_pred_rate']:.3f}")
        bits.append(f"onset_pos_rate={metrics['onset_pos_rate']:.3f}")
        bits.append("total=-1")
    return " ".join(bits)


def append_results(
    results_path: Path,
    round_idx: int,
    burst_epochs: int,
    ckpt_used: Path,
    metrics: Dict[str, float],
    patience_left: int,
    exp_name: str,
    val_line: str,
    retcode: int,
) -> None:
    iso = _dt.datetime.utcnow().isoformat(timespec="seconds")
    row = [
        iso,
        str(round_idx),
        str(burst_epochs),
        ckpt_used.name,
        f"{metrics['onset_thr']:.4f}",
        f"{metrics['offset_thr']:.4f}",
        f"{metrics['onset_f1']:.4f}",
        f"{metrics['offset_f1']:.4f}",
        f"{metrics['onset_event_f1']:.4f}",
        f"{metrics['offset_event_f1']:.4f}",
        f"{metrics['ev_f1_mean']:.4f}",
        str(patience_left),
        exp_name,
        val_line,
        str(retcode),
    ]
    with results_path.open("a") as f:
        f.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Automate training/calibration bursts for TiViT-Piano")
    ap.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
    ap.add_argument("--burst_epochs", type=int, default=2)
    ap.add_argument("--first_calib", choices=["thorough", "fast"], default="thorough")
    ap.add_argument("--target_ev_f1", type=float, default=0.15)
    ap.add_argument("--max_rounds", type=int, default=12)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--split_eval", default="val")
    ap.add_argument("--calib_max_clips", type=int)
    ap.add_argument("--calib_frames", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--bias", type=float)
    ap.add_argument("--stdout_dir", type=Path, default=DEFAULT_STDOUT_DIR)
    ap.add_argument("--dataset_max_clips", type=int)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg()
    exp_cfg = cfg.setdefault("experiment", {})
    base_name = base_from_config_name(exp_cfg.get("name", "TiViT"))

    if args.mode == "fresh":
        new_tag = short_id(_dt.datetime.utcnow().isoformat())
        new_name = f"{base_name}_sw_{new_tag}_auto"
        exp_cfg["name"] = new_name
        print(f"[autopilot] fresh mode → experiment name set to {new_name}")
    else:
        print(f"[autopilot] resume mode → keeping experiment name {exp_cfg.get('name', base_name)}")

    train_cfg = cfg.setdefault("training", {})
    metrics_cfg = train_cfg.setdefault("metrics", {})
    loss_cfg = train_cfg.setdefault("loss_weights", {})
    dataset_cfg = cfg.setdefault("dataset", {})
    frame_cfg = dataset_cfg.setdefault("frame_targets", {})

    changed = False
    if ensure_default(train_cfg, ("epochs",), args.burst_epochs):
        changed = True
    if ensure_default(train_cfg, ("eval_freq",), 1):
        changed = True
    if ensure_default(loss_cfg, ("onoff_prior_mean",), 0.02):
        changed = True
    if ensure_default(loss_cfg, ("onoff_prior_weight",), 0.05):
        changed = True
    if ensure_default(frame_cfg, ("tolerance",), 0.10):
        changed = True
    if ensure_default(frame_cfg, ("dilate_active_frames",), 1):
        changed = True
    if args.dataset_max_clips is not None:
        dataset_cfg["max_clips"] = int(args.dataset_max_clips)
        changed = True
    if changed:
        print("[autopilot] applied default knobs for training/calibration")

    save_cfg(cfg)

    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.results.expanduser()
    ensure_results_header(results_path, RESULT_HEADER)

    stdout_dir = args.stdout_dir.expanduser()
    stdout_dir.mkdir(parents=True, exist_ok=True)

    patience_left = 3
    best_ev_mean = -1.0
    first_round_kind = args.first_calib

    for round_idx in range(1, args.max_rounds + 1):
        cfg = load_cfg()
        train_cfg = cfg.setdefault("training", {})
        metrics_cfg = train_cfg.setdefault("metrics", {})
        train_cfg["epochs"] = int(args.burst_epochs)
        train_cfg["eval_freq"] = 1
        train_cfg["resume"] = round_idx > 1 or args.mode == "resume"
        if round_idx > 1 or args.mode == "resume":
            sync_last_to_best(ckpt_dir)
        save_cfg(cfg)

        banner = "NOTICE: Starting training burst (this may take a while)"
        log_banner(results_path, banner)
        if args.dry_run:
            print("[autopilot] dry-run: skipping training execution")
            train_ret = 0
            last_val = None
        else:
            log_path = stdout_dir / f"stdout_round{round_idx:02d}_train.txt"
            train_ret, last_val, _ = run_command(
                [sys.executable, str(TRAIN), "--config", str(CONFIG)],
                log_path,
                capture_last_val=True,
            )
        if train_ret != 0:
            print(f"Training failed with return code {train_ret}", file=sys.stderr)
            append_results(
                results_path,
                round_idx,
                args.burst_epochs,
                Path("n/a"),
                {
                    "onset_thr": 0.0,
                    "offset_thr": 0.0,
                    "onset_f1": 0.0,
                    "offset_f1": 0.0,
                    "onset_event_f1": 0.0,
                    "offset_event_f1": 0.0,
                    "ev_f1_mean": 0.0,
                    "onset_pred_rate": 0.0,
                    "onset_pos_rate": 0.0,
                },
                patience_left,
                cfg.get("experiment", {}).get("name", ""),
                last_val or "",
                train_ret,
            )
            return train_ret

        if args.dry_run:
            ckpt = ckpt_dir / "dry_run.pt"
        else:
            ckpt = find_ckpt(ckpt_dir)
            if ckpt is None:
                print(f"No checkpoint found in {ckpt_dir}", file=sys.stderr)
                return 1

        calib_kind = first_round_kind if round_idx == 1 else "fast"
        if args.dry_run:
            if calib_kind == "thorough":
                banner = "NOTICE: Running THOROUGH calibration (first round; will compute reliability & ECE/Brier)"
            else:
                banner = "NOTICE: Running FAST calibration (subsequent rounds)"
            log_banner(results_path, banner)
            metrics = {
                "onset_thr": 0.3,
                "offset_thr": 0.3,
                "onset_f1": 0.0,
                "offset_f1": 0.0,
                "onset_event_f1": 0.0,
                "offset_event_f1": 0.0,
                "ev_f1_mean": 0.0,
                "onset_pred_rate": 0.0,
                "onset_pos_rate": 0.0,
            }
            eval_ret = 0
        else:
            if calib_kind == "thorough":
                banner = "NOTICE: Running THOROUGH calibration (first round; will compute reliability & ECE/Brier)"
                log_banner(results_path, banner)
                ret = run_calibration(calib_kind, ckpt, stdout_dir, args.split_eval, args.calib_max_clips, args.calib_frames)
                if ret != 0:
                    print("Calibration failed", file=sys.stderr)
                    return ret
            else:
                banner = "NOTICE: Running FAST calibration (subsequent rounds)"
                log_banner(results_path, banner)
                if not CALIB_JSON.exists():
                    print("[autopilot] calibration.json missing; falling back to thorough calibration")
                    ret = run_calibration("thorough", ckpt, stdout_dir, args.split_eval, args.calib_max_clips, args.calib_frames)
                    if ret != 0:
                        print("Calibration failed", file=sys.stderr)
                        return ret

            calib = load_calibration(CALIB_JSON)
            if calib is None:
                print("calibration.json not found after calibration", file=sys.stderr)
                return 1

            onset_prob = float(calib.get("onset", {}).get("best_prob", 0.3))
            offset_prob = float(calib.get("offset", {}).get("best_prob", 0.3))
            prob_delta = 0.05
            extra_probs = (
                max(0.0, onset_prob - prob_delta),
                onset_prob,
                min(1.0, onset_prob + prob_delta),
            )
            eval_ret, lines = run_fast_eval(
                ckpt,
                stdout_dir,
                args.split_eval,
                CALIB_JSON,
                extra_probs=extra_probs,
                temperature=args.temperature,
                bias=args.bias,
            )
            if eval_ret != 0:
                print("Evaluation failed", file=sys.stderr)
                return eval_ret
            metrics = parse_eval_table(lines)
            if metrics is None:
                print("Failed to parse evaluation output", file=sys.stderr)
                return 1
            metrics["onset_thr"] = metrics.get("onset_thr", onset_prob)
            metrics["offset_thr"] = metrics.get("offset_thr", offset_prob)
            calib.setdefault("onset", {})["best_prob"] = metrics["onset_thr"]
            calib.setdefault("offset", {})["best_prob"] = metrics["offset_thr"]
            with CALIB_JSON.open("w") as f:
                json.dump(calib, f, indent=2)

        cfg = load_cfg()
        metrics_cfg = cfg.setdefault("training", {}).setdefault("metrics", {})
        metrics_cfg["prob_threshold"] = metrics["onset_thr"]
        save_cfg(cfg)

        val_line = format_val_line(metrics, last_val)
        ev_mean = metrics["ev_f1_mean"]
        ckpt_last = ckpt_dir / "tivit_last.pt"
        ckpt_best = ckpt_dir / "tivit_best.pt"
        improved = ckpt_last.exists() and ev_mean > best_ev_mean + 1e-9
        patience_record = 3 if improved else max(patience_left - 1, 0)
        append_results(
            results_path,
            round_idx,
            args.burst_epochs,
            ckpt,
            metrics,
            patience_record,
            cfg.get("experiment", {}).get("name", ""),
            val_line,
            eval_ret,
        )

        summary_csv = stdout_dir / "summary.csv"
        summary_md = stdout_dir / "summary.md"
        run_command(
            [
                sys.executable,
                str(PARSE_SWEEP),
                "--results",
                str(results_path),
                "--out_csv",
                str(summary_csv),
                "--out_md",
                str(summary_md),
            ],
            stdout_dir / f"stdout_round{round_idx:02d}_summary.txt",
            capture_last_val=False,
        )

        if improved:
            shutil.copy2(ckpt_last, ckpt_best)
            best_ev_mean = ev_mean
            patience_left = 3
            print(f"[autopilot] New best ev_f1_mean={ev_mean:.4f} (round {round_idx}); updated tivit_best.pt")
        else:
            patience_left = patience_record
            print(f"[autopilot] ev_f1_mean={ev_mean:.4f} (best={best_ev_mean:.4f}) patience_left={patience_left}")

        if ev_mean >= args.target_ev_f1:
            print("SUCCESS: target reached")
            return 0
        if patience_left <= 0:
            print("EARLY STOP: no improvement for 3 rounds")
            return 0

    print("Reached max rounds without meeting target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
