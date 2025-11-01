#!/usr/bin/env python3
"""Automated training and calibration driver for TiViT-Piano.

Purpose:
    Orchestrate repeated "train → calibrate → evaluate" rounds until the target
    event-level F1 is achieved or patience expires. The script keeps
    ``configs/config.yaml`` synchronized with calibration results, mirrors
    helper stdout/stderr to logs, and records each round in a TSV ledger. It
    can resume from an existing experiment, jump straight to calibration, or
    fall back to coarse sweeps when thorough passes fail.

Key Functions/Classes:
    - run_command: Execute helper scripts while teeing stdout/stderr to disk.
    - run_calibration / calibrate_and_score: Invoke calibration/evaluation helpers
      and parse their metrics.
    - append_results: Persist round metadata to ``runs/auto/results.txt``.
    - main: CLI entry point that drives the automation loop.

CLI Arguments:
    --mode {fresh,resume} (default: fresh)
        Reset experiment naming and ledgers or resume existing state.
    --burst_epochs INT (default: 4)
        Training epochs per burst before running calibration.
    --first_calib {thorough,fast} (default: thorough)
        Calibration mode used for the first round.
    --first_step {train,calib} (default: train)
        Initial action when starting the automation loop.
    --skip_train_round1 (default: False)
        Skip the first training burst and start with calibration.
    --fast_first_calib (default: False)
        Force the first calibration to use the fast sweep parameters.
    --target_ev_f1 FLOAT (default: 0.65)
        Target event-level F1 score that terminates the loop when reached.
    --target_metric {ev_f1_mean,onset_ev_f1,offset_ev_f1} (default: ev_f1_mean)
        Metric field evaluated against ``--target_ev_f1``.
    --max_rounds INT (default: 12)
        Maximum number of train/calibrate rounds to execute.
    --patience INT (default: 3)
        Allowed number of non-improving rounds before aborting.
    --results PATH (default: logs/auto/results.txt)
        Ledger file that stores per-round results.
    --ckpt_dir PATH (default: checkpoints)
        Directory containing training checkpoints for calibration/eval.
    --split_eval STR (default: val)
        Dataset split evaluated during fast metrics checks.
    --calib_max_clips INT (default: None)
        Override the maximum clips seen during calibration (falls back to config).
    --calib_frames INT (default: 96)
        Override the number of frames per clip when calibrating/evaluating.
    --temperature FLOAT (default: None)
        Manually set calibration temperature instead of discovering it.
    --bias FLOAT (default: None)
        Manually set calibration bias instead of discovering it.
    --stdout_dir PATH (default: logs/auto)
        Directory where captured stdout/stderr transcripts are written.
    --dataset_max_clips INT (default: None)
        Override dataset max clips for both training and calibration runs.
    --dry_run (default: False)
        Print the planned actions but skip execution.
    --seed INT (default: config or 1337)
        Forwarded seed for child training/calibration/eval scripts.
    --deterministic / --no-deterministic
        Toggle deterministic PyTorch backends for child processes.
    --verbose {quiet,info,debug} (default: env or quiet)
        Logging verbosity for the autopilot and child processes.
    --eval_extras STR (default: "")
        Additional CLI tokens appended to ``eval_thresholds.py`` during fast evaluation.

Usage:
    python scripts/train_autopilot.py --mode fresh --first_step train --burst_epochs 3
"""

from __future__ import annotations

import argparse
import copy
import io
import datetime as _dt
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
EPOCH_CKPT_RE = re.compile(r"tivit_epoch_(\d+)\.pt$")

DEFAULT_RESULTS = REPO / "logs" / "auto" / "results.txt"
DEFAULT_STDOUT_DIR = REPO / "logs" / "auto"
CALIB_JSON = REPO / "calibration.json"

TARGET_METRIC_FIELDS = {
    "ev_f1_mean": "ev_f1_mean",
    "onset_ev_f1": "onset_event_f1",
    "offset_ev_f1": "offset_event_f1",
}

sys.path.insert(0, str(REPO / "src"))
from utils.logging_utils import QUIET_INFO_FLAG, configure_verbosity
from utils.determinism import resolve_deterministic_flag, resolve_seed
from utils.selection import (
    SweepSpec,
    SelectionRequest,
    SelectionResult,
    SelectionContext,
    calibrate_and_score,
    record_best,
    SelectionError,
    decoder_snapshot_from_config,
    tolerance_snapshot_from_config,
)

LOGGER = logging.getLogger("autopilot")
QUIET_EXTRA = {QUIET_INFO_FLAG: True}


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_cfg() -> dict:
    with CONFIG.open("r") as f:
        return yaml.safe_load(f)


def _inject_decoder_comments(text: str) -> str:
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    target_prefix = ["training", "metrics", "decoder"]
    comment_keys = {
        "onset": {"open", "hold", "min_on", "merge_gap"},
        "offset": {"open", "hold", "min_off", "merge_gap"},
    }
    path: list[tuple[str, int]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        while path and indent <= path[-1][1]:
            path.pop()
        if stripped.endswith(":"):
            key = stripped[:-1]
            path.append((key, indent))
            continue
        keys_path = [item[0] for item in path]
        if len(keys_path) < 4 or keys_path[:3] != target_prefix:
            continue
        head = keys_path[3]
        allowed = comment_keys.get(head)
        if not allowed:
            continue
        key_name = stripped.split(":", 1)[0]
        if key_name not in allowed:
            continue
        if "# default" in line:
            continue
        lines[idx] = line.rstrip("\n").rstrip() + "  # default\n"
    return "".join(lines)


def save_cfg(cfg: dict) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    yaml.safe_dump(cfg, buffer, sort_keys=False)
    rendered = _inject_decoder_comments(buffer.getvalue())
    with CONFIG.open("w") as f:
        f.write(rendered)


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


def _with_verbose(cmd: Iterable[str], verbose: Optional[str]) -> List[str]:
    result = list(cmd)
    if not verbose:
        return result
    if "--verbose" in result:
        return result
    result.extend(["--verbose", verbose])
    return result


def _append_determinism_flags(
    cmd: List[str],
    *,
    seed: Optional[int],
    deterministic: Optional[bool],
) -> List[str]:
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    if deterministic is True:
        cmd.append("--deterministic")
    elif deterministic is False:
        cmd.append("--no-deterministic")
    return cmd


def run_command(
    cmd: List[str],
    log_file: Path,
    capture_last_val: bool = False,
    *,
    extra_env: Optional[Dict[str, str]] = None,
    unset_env: Optional[Iterable[str]] = None,
    echo_cmd: bool = True,
    verbose: Optional[str] = None,
) -> Tuple[int, Optional[str], List[str]]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if verbose:
        env["TIVIT_VERBOSE"] = verbose
    if unset_env:
        for key in unset_env:
            env.pop(key, None)
    if extra_env:
        env.update(extra_env)
    if verbose:
        cmd_display = " ".join(cmd)
        print(f"[autopilot] verbose={verbose} → passing to child {cmd_display}")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if echo_cmd:
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

def _build_dataset_cli(split: Optional[str], frames: Optional[int], max_clips: Optional[int]) -> List[str]:
    args: List[str] = []
    if split:
        args.extend(["--split", split])
    if frames is not None:
        args.extend(["--frames", str(frames)])
    if max_clips is not None:
        args.extend(["--max-clips", str(max_clips)])
    return args


def run_calibration(
    kind: str,
    ckpt: Path,
    log_dir: Path,
    split: str,
    max_clips: Optional[int],
    frames: Optional[int],
    *,
    verbose: Optional[str] = None,
    seed: Optional[int] = None,
    deterministic: Optional[bool] = None,
) -> int:
    log_name = f"calibration_{kind}.txt"
    log_path = log_dir / log_name
    dataset_cli = _build_dataset_cli(split, frames, max_clips)
    cmd = [sys.executable, "-u", str(CALIBRATE_THRESH), *dataset_cli, "--ckpt", str(ckpt)]
    cmd = _append_determinism_flags(cmd, seed=seed, deterministic=deterministic)
    cmd = _with_verbose(cmd, verbose)
    if kind == "thorough":
        env_to_unset = ["AVSYNC_DISABLE", "DEBUG"]
        cmd_display = [Path(cmd[0]).name] + cmd[1:]
        print(f"[autopilot:thorough] cmd: {' '.join(cmd_display)}")
        unset_desc = ", ".join(f"{name}=<unset>" for name in env_to_unset)
        print(f"[autopilot:thorough] env: {unset_desc}")
        ret, _, lines = run_command(
            cmd,
            log_path,
            capture_last_val=False,
            unset_env=env_to_unset,
            echo_cmd=False,
            verbose=verbose,
        )
        bad_markers = ("[debug] AV-lag disabled", "lag_source=no_avlag")
        if any(marker in line for line in lines for marker in bad_markers):
            print("[autopilot] ERROR: thorough calibration disabled A/V lag; aborting.", flush=True)
            return 1
        return ret
    ret, _, _ = run_command(cmd, log_path, capture_last_val=False, verbose=verbose)
    return ret


def _format_eval_value(value: Optional[object]) -> str:
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            pass
    return str(value)


def _log_eval_settings() -> None:
    cfg = load_cfg()
    train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    metrics_cfg = train_cfg.get("metrics", {}) if isinstance(train_cfg, dict) else {}
    agg_cfg = metrics_cfg.get("aggregation", {}) if isinstance(metrics_cfg, dict) else {}
    k_cfg = agg_cfg.get("k", {}) if isinstance(agg_cfg, dict) else {}

    k_onset_val = k_cfg.get("onset") if isinstance(k_cfg, dict) else None
    top_k_val = agg_cfg.get("top_k") if isinstance(agg_cfg, dict) else None
    onset_thr = metrics_cfg.get("prob_threshold_onset") if isinstance(metrics_cfg, dict) else None
    offset_thr = metrics_cfg.get("prob_threshold_offset") if isinstance(metrics_cfg, dict) else None

    message = (
        "[autopilot] eval config: "
        f"k_onset={_format_eval_value(k_onset_val)} "
        f"top_k={_format_eval_value(top_k_val)} "
        f"prob_threshold_onset={_format_eval_value(onset_thr)} "
        f"prob_threshold_offset={_format_eval_value(offset_thr)}"
    )
    print(message)




FAST_GRID_THRESHOLDS = [0.40, 0.44, 0.48, 0.52, 0.56, 0.60]

FAST_SWEEP_CLIP_MIN = 0.02
FAST_SWEEP_CLIP_MAX = 0.98
FAST_RESULT_MIN = 0.02
FAST_RESULT_MAX = 0.98


def _result_to_metrics(result: SelectionResult) -> Dict[str, float]:
    metrics = {
        "onset_thr": result.onset_threshold,
        "offset_thr": result.offset_threshold,
        "onset_f1": result.onset_f1 or 0.0,
        "offset_f1": result.offset_f1 or 0.0,
        "onset_event_f1": result.onset_event_f1,
        "offset_event_f1": result.offset_event_f1,
        "onset_pred_rate": result.onset_pred_rate or 0.0,
        "onset_pos_rate": result.onset_pos_rate or 0.0,
        "ev_f1_mean": result.mean_event_f1,
        "k_onset": result.k_onset,
    }
    if result.decoder_kind:
        metrics["decoder_kind"] = result.decoder_kind
    if result.decoder_settings:
        metrics.update(result.decoder_settings)
    return metrics


def _write_calibration_json(result: SelectionResult) -> None:
    data = load_calibration(CALIB_JSON) or {}
    onset = data.get("onset", {})
    offset = data.get("offset", {})
    onset["best_prob"] = _clamp_fast_result(float(result.onset_threshold))
    offset["best_prob"] = _clamp_fast_result(float(result.offset_threshold))
    data["onset"] = onset
    data["offset"] = offset
    with CALIB_JSON.open("w") as f:
        json.dump(data, f, indent=2)


def _coerce_optional_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            return float(value)
        return None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return float(parsed) if math.isfinite(parsed) else None
    return None


def _coerce_positive_int(value) -> Optional[int]:
    float_val = _coerce_optional_float(value)
    if float_val is None:
        return None
    if float_val < 1.0:
        return None
    rounded = int(round(float_val))
    if abs(float_val - rounded) > 1e-6:
        return None
    return rounded


def _logit_to_probability(logit: float) -> float:
    if logit >= 0.0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _clamp_probability(prob: float) -> float:
    return max(0.0, min(1.0, float(prob)))


def _clamp_range(prob: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(prob)))


def _clamp_fast_result(prob: float) -> float:
    return _clamp_range(prob, FAST_RESULT_MIN, FAST_RESULT_MAX)


def _extract_best_probability(
    calibration: Optional[dict], head: str, fallback: Optional[float]
) -> Optional[float]:
    entry = calibration.get(head) if isinstance(calibration, dict) else None
    prob_val: Optional[float] = None
    if isinstance(entry, dict):
        if "best_prob" in entry:
            best_prob = _coerce_optional_float(entry.get("best_prob"))
            if best_prob is not None:
                prob_val = _clamp_probability(best_prob)
        if prob_val is None:
            best_logit = _coerce_optional_float(entry.get("best_logit"))
            if best_logit is not None:
                prob_val = _clamp_probability(_logit_to_probability(best_logit))
    if prob_val is None:
        fallback_val = _coerce_optional_float(fallback)
        if fallback_val is None:
            return None
        if 0.0 <= fallback_val <= 1.0:
            prob_val = fallback_val
        else:
            prob_val = _logit_to_probability(fallback_val)
    return _clamp_probability(prob_val)


def _blend_and_clip_platt(
    platt: Optional[Dict[str, float]],
    *,
    neutral_temp: float,
    neutral_bias: float,
    temp_bounds: Tuple[float, float],
    bias_bounds: Tuple[float, float],
) -> Tuple[Optional[Tuple[float, float]], bool]:
    """Blend Platt params toward neutral defaults and enforce safe ranges.

    Returns a tuple ``(values, out_of_range)`` where ``values`` is either a
    ``(temperature, bias)`` pair ready for YAML persistence or ``None`` if the
    Platt entry is unusable for this round. ``out_of_range`` is ``True`` when
    the raw parameters existed but violated the configured bounds, signalling
    that the caller should log a warning about skipping the write.
    """

    if not isinstance(platt, dict):
        return None, False

    temp_raw = _coerce_optional_float(platt.get("temp"))
    bias_raw = _coerce_optional_float(platt.get("bias"))
    if temp_raw is None or bias_raw is None:
        return None, False

    temp = 0.5 * temp_raw + 0.5 * neutral_temp
    bias = 0.5 * bias_raw + 0.5 * neutral_bias

    temp_lo, temp_hi = temp_bounds
    bias_lo, bias_hi = bias_bounds

    if not (temp_lo <= temp <= temp_hi and bias_lo <= bias <= bias_hi):
        return None, True

    temp = min(max(temp, temp_lo), temp_hi)
    bias = min(max(bias, bias_lo), bias_hi)
    return (temp, bias), False


def apply_metrics_to_config(metrics: Dict[str, float]) -> None:
    cfg = load_cfg()
    train_cfg = cfg.setdefault("training", {})
    metrics_cfg = train_cfg.setdefault("metrics", {})
    decoder_snapshot = copy.deepcopy(metrics_cfg.get("decoder"))
    agg_cfg = metrics_cfg.setdefault("aggregation", {})
    k_cfg = agg_cfg.setdefault("k", {})
    calibration = load_calibration(CALIB_JSON)

    existing_onset = _coerce_optional_float(metrics_cfg.get("prob_threshold_onset"))
    existing_offset = _coerce_optional_float(metrics_cfg.get("prob_threshold_offset"))
    onset_fallback = _coerce_optional_float(metrics.get("onset_thr")) or existing_onset or 0.5
    offset_fallback = _coerce_optional_float(metrics.get("offset_thr")) or existing_offset or 0.5

    onset_prob = _extract_best_probability(calibration, "onset", onset_fallback)
    offset_prob = _extract_best_probability(calibration, "offset", offset_fallback)

    if onset_prob is not None:
        if 0.30 <= onset_prob <= 0.80:
            metrics_cfg["prob_threshold_onset"] = float(onset_prob)
            metrics_cfg["prob_threshold"] = float(onset_prob)
        else:
            print(
                f"[autopilot] WARNING: onset prob_threshold={onset_prob:.4f} outside [0.30, 0.80]; skipping write"
            )
    if offset_prob is not None:
        if 0.30 <= offset_prob <= 0.80:
            metrics_cfg["prob_threshold_offset"] = float(offset_prob)
        else:
            print(
                f"[autopilot] WARNING: offset prob_threshold={offset_prob:.4f} outside [0.30, 0.80]; skipping write"
            )
    agg_cfg["mode"] = "k_of_p"
    k_onset_metric = _coerce_positive_int(metrics.get("k_onset"))
    if k_onset_metric is not None:
        k_cfg["onset"] = k_onset_metric
    elif _coerce_positive_int(k_cfg.get("onset")) is None:
        k_cfg.setdefault("onset", 1)

    if "offset" not in k_cfg or not int(k_cfg.get("offset", 0)):
        k_cfg["offset"] = 1
    platt = read_platt_params(CALIB_JSON)
    if platt:
        onset_platt = platt.get("onset")
        if onset_platt:
            onset_vals, onset_oob = _blend_and_clip_platt(
                onset_platt,
                neutral_temp=1.0,
                neutral_bias=0.0,
                temp_bounds=(0.85, 1.50),
                bias_bounds=(-1.0, 1.0),
            )
            if onset_vals is not None:
                onset_temp, onset_bias = onset_vals
                metrics_cfg["prob_temperature_onset"] = onset_temp
                metrics_cfg["prob_logit_bias_onset"] = onset_bias
            elif onset_oob:
                print("[autopilot] WARNING: onset Platt params outside safe range; skipping write")
        offset_platt = platt.get("offset")
        if offset_platt:
            offset_vals, offset_oob = _blend_and_clip_platt(
                offset_platt,
                neutral_temp=1.0,
                neutral_bias=0.0,
                temp_bounds=(0.85, 1.80),
                bias_bounds=(-1.5, 1.0),
            )
            if offset_vals is not None:
                offset_temp, offset_bias = offset_vals
                metrics_cfg["prob_temperature_offset"] = offset_temp
                metrics_cfg["prob_logit_bias_offset"] = offset_bias
            elif offset_oob:
                print("[autopilot] WARNING: offset Platt params outside safe range; skipping write")
    onset_final = _coerce_optional_float(metrics_cfg.get("prob_threshold_onset"))
    offset_final = _coerce_optional_float(metrics_cfg.get("prob_threshold_offset"))
    k_onset_final = _coerce_positive_int(k_cfg.get("onset"))
    print(
        "[autopilot] final thresholds: onset={onset}, offset={offset}, k_onset={k_onset}".format(
            onset=f"{onset_final:.4f}" if onset_final is not None else "None",
            offset=f"{offset_final:.4f}" if offset_final is not None else "None",
            k_onset=k_onset_final if k_onset_final is not None else "None",
        )
    )
    if decoder_snapshot is None:
        metrics_cfg.pop("decoder", None)
    else:
        metrics_cfg["decoder"] = decoder_snapshot
    save_cfg(cfg)
    reloaded_cfg = load_cfg()
    reloaded_metrics = reloaded_cfg.get("training", {}).get("metrics", {}) if isinstance(reloaded_cfg, Mapping) else {}
    decoder_after = reloaded_metrics.get("decoder") if isinstance(reloaded_metrics, Mapping) else {}
    if decoder_after != decoder_snapshot:
        print("[autopilot] ERROR: decoder subtree changed during write-back; aborting to protect config", file=sys.stderr)
        raise SystemExit(1)


def infer_current_epoch(ckpt_dir: Path) -> int:
    best_epoch = 0
    for ckpt in ckpt_dir.glob("tivit_epoch_*.pt"):
        m = EPOCH_CKPT_RE.match(ckpt.name)
        if not m:
            continue
        try:
            epoch_val = int(m.group(1))
        except ValueError:
            continue
        best_epoch = max(best_epoch, epoch_val)
    if best_epoch > 0:
        return best_epoch
    try:
        import torch  # type: ignore
    except Exception:
        return best_epoch
    for name in ("tivit_last.pt", "tivit_best.pt"):
        path = ckpt_dir / name
        if not path.exists():
            continue
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if isinstance(ckpt, dict):
            for key in ("epoch", "current_epoch", "global_epoch"):
                value = ckpt.get(key)
                if isinstance(value, int) and value > 0:
                    return int(value)
    return best_epoch


def load_calibration(calibration_json: Path) -> Optional[dict]:
    if not calibration_json.exists():
        return None
    with calibration_json.open("r") as f:
        return json.load(f)


def _extract_platt_from_mapping(mapping: dict) -> Optional[Dict[str, float]]:
    candidates = []
    platt = mapping.get("platt")
    if isinstance(platt, dict):
        candidates.append(platt)
    candidates.append(mapping)

    temp_val: Optional[float] = None
    bias_val: Optional[float] = None
    temp_keys = {"temp", "temperature", "a"}
    bias_keys = {"bias", "b", "logit_bias"}

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key, value in candidate.items():
            if isinstance(value, (int, float)):
                val = float(value)
            elif isinstance(value, str):
                try:
                    val = float(value)
                except ValueError:
                    continue
            else:
                continue

            norm_key = key.lower()
            for prefix in ("platt_", "prob_", "logit_"):
                if norm_key.startswith(prefix):
                    norm_key = norm_key[len(prefix) :]
            if norm_key in temp_keys:
                temp_val = val
            elif norm_key in bias_keys:
                bias_val = val

    if temp_val is None or bias_val is None:
        return None
    if not (math.isfinite(temp_val) and math.isfinite(bias_val)):
        return None
    return {"temp": float(temp_val), "bias": float(bias_val)}


def read_platt_params(calibration_json: Path) -> Dict[str, Dict[str, float]]:
    data = load_calibration(calibration_json)
    if not isinstance(data, dict):
        return {}

    platt: Dict[str, Dict[str, float]] = {}
    for head in ("onset", "offset"):
        head_data = data.get(head)
        if not isinstance(head_data, dict):
            continue
        extracted = _extract_platt_from_mapping(head_data)
        if extracted is None:
            continue
        platt[head] = extracted
    return platt


def perform_calibration(
    *,
    ckpt: Path,
    args,
    results_path: Path,
    stdout_dir: Path,
    split: str,
    calibration_count: int,
    seed: int,
    deterministic: bool,
) -> Tuple[Optional[SelectionResult], Optional[SelectionContext], List[str], int]:
    first_calibration = calibration_count == 0
    prefer_fast_grid = first_calibration and args.fast_first_calib
    desired_kind = args.first_calib if first_calibration else "fast"

    cfg_snapshot = load_cfg()
    prev_metrics = (cfg_snapshot.get("training", {}) or {}).get("metrics", {}) or {}
    prev_onset_best = _coerce_optional_float(prev_metrics.get("prob_threshold_onset"))
    prev_offset_best = _coerce_optional_float(prev_metrics.get("prob_threshold_offset"))

    def _resolve_anchor(
        calib_entry: Optional[dict],
        *,
        previous: Optional[float],
        fallback: float,
    ) -> Tuple[float, Optional[float], str]:
        prob_val: Optional[float] = None
        logit_val: Optional[float] = None
        if isinstance(calib_entry, dict):
            cal_prob = _coerce_optional_float(calib_entry.get("best_prob"))
            if cal_prob is not None:
                prob_val = _clamp_probability(cal_prob)
            else:
                cal_logit = _coerce_optional_float(calib_entry.get("best_logit"))
                if cal_logit is not None:
                    logit_val = cal_logit
                    prob_val = _clamp_probability(_logit_to_probability(cal_logit))
        if prob_val is not None:
            return float(prob_val), logit_val, "calibration"
        if previous is not None:
            return float(_clamp_probability(previous)), None, "best"
        return float(_clamp_probability(fallback)), None, "fallback"

    decoder_snapshot = decoder_snapshot_from_config(cfg_snapshot)
    tolerance_snapshot = tolerance_snapshot_from_config(cfg_snapshot)
    exp_name = cfg_snapshot.get("experiment", {}).get("name", "")
    prob_delta = 0.05

    def _build_request(
        onset_center: float,
        offset_center: float,
        *,
        log_name: str,
        sweep_override: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        extras: Optional[Sequence[str]] = None,
        low_guard: float = 0.10,
    ) -> SelectionRequest:
        onset_explicit = sweep_override[0] if sweep_override else None
        offset_explicit = sweep_override[1] if sweep_override else None
        sweep = SweepSpec(
            onset_center=onset_center,
            offset_center=offset_center,
            delta=prob_delta,
            clamp_min=FAST_SWEEP_CLIP_MIN,
            clamp_max=FAST_SWEEP_CLIP_MAX,
            low_guard=low_guard,
            onset_explicit=onset_explicit,
            offset_explicit=offset_explicit,
        )
        eval_extras = list(args.eval_extras_tokens or [])
        if extras:
            eval_extras.extend(extras)
        frames = args.calib_frames if args.calib_frames is not None else 96
        max_clips = args.calib_max_clips if args.calib_max_clips is not None else 80
        return SelectionRequest(
            ckpt=ckpt,
            split=split,
            frames=frames,
            max_clips=max_clips,
            sweep=sweep,
            decoder=decoder_snapshot,
            tolerances=tolerance_snapshot,
            temperature=args.temperature,
            bias=args.bias,
            seed=seed,
            deterministic=deterministic,
            eval_extras=eval_extras,
            verbose=args.verbose,
            log_path=stdout_dir / log_name,
            run_id=exp_name,
        )

    def _run_selection_with_calibration(calib: dict, *, log_name: str) -> Tuple[SelectionResult, SelectionContext, List[str]]:
        banner = "NOTICE: Running FAST calibration — this may take a while"
        log_banner(results_path, banner)
        onset_entry = calib.get("onset", {}) if isinstance(calib, dict) else {}
        offset_entry = calib.get("offset", {}) if isinstance(calib, dict) else {}
        onset_prob, onset_logit, onset_src = _resolve_anchor(
            onset_entry,
            previous=prev_onset_best,
            fallback=0.3,
        )
        offset_prob, offset_logit, offset_src = _resolve_anchor(
            offset_entry,
            previous=prev_offset_best,
            fallback=0.3,
        )
        onset_prob = _clamp_fast_result(onset_prob)
        offset_prob = _clamp_fast_result(offset_prob)
        if onset_src == offset_src:
            anchor_source = onset_src
        else:
            anchor_source = f"onset:{onset_src},offset:{offset_src}"
        LOGGER.info(
            "[autopilot:grid] anchors onset=%.4f offset=%.4f (source=%s)",
            onset_prob,
            offset_prob,
            anchor_source,
            extra=QUIET_EXTRA,
        )
        request = _build_request(onset_prob, offset_prob, log_name=log_name)
        result, context, lines = calibrate_and_score(request)
        _write_calibration_json(result)
        return result, context, lines

    def _run_fast_grid(reason: Optional[str] = None) -> Tuple[Optional[SelectionResult], Optional[SelectionContext], List[str], int]:
        if reason:
            print(f"[autopilot] WARNING: {reason} → falling back to fast grid calibration")
        banner = "NOTICE: Running FAST calibration — this may take a while"
        log_banner(results_path, banner)
        try:
            request = _build_request(
                0.5,
                0.5,
                log_name="calibration_fast_grid.txt",
                sweep_override=(FAST_GRID_THRESHOLDS, FAST_GRID_THRESHOLDS),
                extras=["--grid_prob_thresholds"],
                low_guard=0.02,
            )
            result, context, lines = calibrate_and_score(request)
        except SelectionError as exc:
            LOGGER.error("fast grid calibration failed: %s", exc)
            return None, None, [], 1
        _write_calibration_json(result)
        return result, context, lines, 0

    if prefer_fast_grid:
        return _run_fast_grid()

    if desired_kind == "thorough":
        banner = "NOTICE: Running THOROUGH calibration — this may take a while"
        log_banner(results_path, banner)
        ret = run_calibration(
            "thorough",
            ckpt,
            stdout_dir,
            split,
            args.calib_max_clips,
            args.calib_frames,
            verbose=args.verbose,
            seed=seed,
            deterministic=deterministic,
        )
        if ret != 0:
            return _run_fast_grid("thorough calibration failed")
        calib = load_calibration(CALIB_JSON)
        if calib is None:
            return _run_fast_grid("calibration.json missing after thorough calibration")
        try:
            result, context, lines = _run_selection_with_calibration(calib, log_name="eval_fast.txt")
        except SelectionError as exc:
            LOGGER.error("fast evaluation failed after thorough calibration: %s", exc)
            return _run_fast_grid("fast evaluation failed after thorough calibration")
        return result, context, lines, 0

    calib = load_calibration(CALIB_JSON)
    if calib is None:
        return _run_fast_grid("calibration.json missing for fast calibration")
    try:
        result, context, lines = _run_selection_with_calibration(calib, log_name="eval_fast.txt")
    except SelectionError as exc:
        LOGGER.error("fast evaluation failed: %s", exc)
        return _run_fast_grid("fast evaluation failed")
    return result, context, lines, 0


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------
@dataclass
class ResumeState:
    next_round: int
    best_metric: float
    patience_left: int
    calibration_count: int


RESULT_HEADER = [
    "iso8601",
    "round",
    "burst_epochs",
    "ckpt",
    "onset_thr",
    "offset_thr",
    "k_onset",
    "onset_f1",
    "offset_f1",
    "onset_ev_f1",
    "offset_ev_f1",
    "ev_f1_mean",
    "decoder_kind",
    "decoder_onset_open",
    "decoder_onset_hold",
    "decoder_offset_open",
    "decoder_offset_hold",
    "decoder_onset_min_on",
    "decoder_offset_min_off",
    "decoder_onset_merge_gap",
    "decoder_offset_merge_gap",
    "patience",
    "exp",
    "val_line",
    "retcode",
]


def load_resume_state(results_path: Path, metric_key: str, default_patience: int) -> Optional[ResumeState]:
    if not results_path.exists():
        return None

    rows: List[Dict[str, str]] = []
    try:
        with results_path.open("r") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(RESULT_HEADER):
                    parts = parts + [""] * (len(RESULT_HEADER) - len(parts))
                elif len(parts) > len(RESULT_HEADER):
                    parts = parts[: len(RESULT_HEADER)]
                rows.append(dict(zip(RESULT_HEADER, parts)))
    except OSError:
        return None

    if not rows:
        return None

    metric_key = metric_key or "ev_f1_mean"
    best_metric = -1.0
    for row in rows:
        try:
            val = float(row.get(metric_key, "nan"))
        except (TypeError, ValueError):
            continue
        if not (val != val):  # NaN guard
            best_metric = max(best_metric, val)

    last_row = rows[-1]
    try:
        next_round = int(last_row.get("round", "0")) + 1
    except (TypeError, ValueError):
        next_round = len(rows) + 1
    try:
        patience_left = int(last_row.get("patience", str(default_patience)))
    except (TypeError, ValueError):
        patience_left = default_patience
    patience_left = max(patience_left, 0)

    return ResumeState(
        next_round=next_round,
        best_metric=best_metric,
        patience_left=patience_left,
        calibration_count=len(rows),
    )


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
        decoder_kind = metrics.get("decoder_kind")
        if decoder_kind:
            bits.append(f"decoder={decoder_kind}")
        decoder_fields = [
            ("decoder_onset_open", ("decoder_onset_open", "onset_open"), "{:.3f}", float),
            ("decoder_onset_hold", ("decoder_onset_hold", "onset_hold"), "{:.3f}", float),
            ("decoder_onset_min_on", ("decoder_onset_min_on", "onset_min_on"), "{:d}", int),
            ("decoder_onset_merge_gap", ("decoder_onset_merge_gap", "onset_merge_gap"), "{:d}", int),
            ("decoder_offset_open", ("decoder_offset_open", "offset_open"), "{:.3f}", float),
            ("decoder_offset_hold", ("decoder_offset_hold", "offset_hold"), "{:.3f}", float),
            ("decoder_offset_min_off", ("decoder_offset_min_off", "offset_min_off"), "{:d}", int),
            ("decoder_offset_merge_gap", ("decoder_offset_merge_gap", "offset_merge_gap"), "{:d}", int),
        ]
        for label, keys, fmt, caster in decoder_fields:
            value = None
            for key in keys:
                if key in metrics and metrics[key] is not None:
                    value = metrics[key]
                    break
            if value is None:
                continue
            if caster is int:
                try:
                    coerced = int(round(float(value)))
                except (TypeError, ValueError):
                    continue
                bits.append(f"{label}={fmt.format(coerced)}")
            else:
                try:
                    coerced = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(coerced) or math.isinf(coerced):
                    continue
                bits.append(f"{label}={fmt.format(coerced)}")
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
    def _fmt_float(val: Optional[float], precision: str = ".4f", default: str = "") -> str:
        if val is None:
            return default
        try:
            coerced = float(val)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(coerced):
            return default
        return format(coerced, precision)

    def _fmt_int(val: Optional[float], default: str = "") -> str:
        if val is None:
            return default
        try:
            coerced = float(val)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(coerced):
            return default
        try:
            return str(int(round(coerced)))
        except (TypeError, ValueError):
            return default

    iso = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    k_onset_val = metrics.get("k_onset")
    row = [
        iso,
        str(round_idx),
        str(burst_epochs),
        ckpt_used.name,
        _fmt_float(metrics.get("onset_thr"), ".4f", "0.0000"),
        _fmt_float(metrics.get("offset_thr"), ".4f", "0.0000"),
        _fmt_int(k_onset_val, "1"),
        _fmt_float(metrics.get("onset_f1"), ".4f", "0.0000"),
        _fmt_float(metrics.get("offset_f1"), ".4f", "0.0000"),
        _fmt_float(metrics.get("onset_event_f1"), ".4f", "0.0000"),
        _fmt_float(metrics.get("offset_event_f1"), ".4f", "0.0000"),
        _fmt_float(metrics.get("ev_f1_mean"), ".4f", "0.0000"),
        str(metrics.get("decoder_kind") or ""),
        _fmt_float(metrics.get("decoder_onset_open"), ".4f"),
        _fmt_float(metrics.get("decoder_onset_hold"), ".4f"),
        _fmt_float(metrics.get("decoder_offset_open"), ".4f"),
        _fmt_float(metrics.get("decoder_offset_hold"), ".4f"),
        _fmt_int(metrics.get("decoder_onset_min_on")),
        _fmt_int(metrics.get("decoder_offset_min_off")),
        _fmt_int(metrics.get("decoder_onset_merge_gap")),
        _fmt_int(metrics.get("decoder_offset_merge_gap")),
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
    ap.add_argument("--burst_epochs", type=int, default=4)
    ap.add_argument("--first_calib", choices=["thorough", "fast"], default="thorough")
    ap.add_argument("--first_step", choices=["train", "calib"], default="train")
    ap.add_argument("--skip_train_round1", action="store_true")
    ap.add_argument("--fast_first_calib", action="store_true")
    ap.add_argument("--target_ev_f1", type=float, default=0.65)
    ap.add_argument("--target_metric", choices=["ev_f1_mean", "onset_ev_f1", "offset_ev_f1"], default="ev_f1_mean")
    ap.add_argument("--max_rounds", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--split_eval", default="val")
    ap.add_argument("--calib_max_clips", type=int)
    ap.add_argument("--calib_frames", type=int, default=96,
                    help="Override frames per clip during calibration/eval (default: 96)")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--bias", type=float)
    ap.add_argument("--stdout_dir", type=Path, default=DEFAULT_STDOUT_DIR)
    ap.add_argument("--dataset_max_clips", type=int)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--seed", type=int, help="Seed forwarded to training/calibration/eval")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle deterministic torch backends for child runs",
    )
    ap.add_argument(
        "--verbose",
        choices=["quiet", "info", "debug"],
        help="Logging verbosity for autopilot and child runs (default: quiet or $TIVIT_VERBOSE)",
    )
    ap.add_argument(
        "--eval_extras",
        type=str,
        default="",
        help="Extra CLI arguments appended to eval_thresholds.py during fast evaluation",
    )
    args = ap.parse_args()
    args.verbose = configure_verbosity(args.verbose)
    try:
        args.eval_extras_tokens = shlex.split(args.eval_extras) if args.eval_extras else []
    except ValueError as exc:
        print(f"error: could not parse --eval_extras: {exc}", file=sys.stderr)
        return 1
    patience_budget = max(args.patience, 0)

    target_metric_field = TARGET_METRIC_FIELDS[args.target_metric]
    
    cfg = load_cfg()
    exp_cfg = cfg.setdefault("experiment", {})
    changed = False

    seed = resolve_seed(args.seed, cfg)
    deterministic = resolve_deterministic_flag(args.deterministic, cfg)
    if exp_cfg.get("seed") != seed:
        exp_cfg["seed"] = seed
        changed = True
    if exp_cfg.get("deterministic") != deterministic:
        exp_cfg["deterministic"] = deterministic
        changed = True
    args.seed = seed
    args.deterministic = deterministic
    print(
        f"[autopilot] determinism seed={seed} deterministic={'on' if deterministic else 'off'}"
    )

    base_name = base_from_config_name(exp_cfg.get("name", "TiViT"))

    if args.mode == "fresh":
        new_tag = short_id(_dt.datetime.now(_dt.UTC).isoformat())
        new_name = f"{base_name}_sw_{new_tag}_auto"
        exp_cfg["name"] = new_name
        print(f"[autopilot] fresh mode → experiment name set to {new_name}")
    else:
        print(f"[autopilot] resume mode → keeping experiment name {exp_cfg.get('name', base_name)}")

    train_cfg = cfg.setdefault("training", {})
    metrics_cfg = train_cfg.setdefault("metrics", {})
    decoder_snapshot = copy.deepcopy(metrics_cfg.get("decoder"))
    loss_cfg = train_cfg.setdefault("loss_weights", {})
    dataset_cfg = cfg.setdefault("dataset", {})
    frame_cfg = dataset_cfg.setdefault("frame_targets", {})

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
        print("[autopilot] updated config defaults/determinism for training/calibration")
    save_cfg(cfg)
    reloaded_cfg = load_cfg()
    reloaded_metrics = reloaded_cfg.get("training", {}).get("metrics", {}) if isinstance(reloaded_cfg, Mapping) else {}
    decoder_after = reloaded_metrics.get("decoder") if isinstance(reloaded_metrics, Mapping) else {}
    if decoder_after != decoder_snapshot:
        print("[autopilot] ERROR: decoder subtree changed during write-back; aborting to protect config", file=sys.stderr)
        raise SystemExit(1)

    ckpt_arg = args.ckpt_dir.expanduser()
    ckpt_hint: Optional[Path] = None
    if ckpt_arg.suffix == ".pt" or (ckpt_arg.exists() and ckpt_arg.is_file()):
        try:
            ckpt_hint = ckpt_arg.resolve()
        except OSError:
            ckpt_hint = ckpt_arg
        ckpt_arg = ckpt_arg.parent
    ckpt_dir = ckpt_arg.resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_hint is not None:
        print(f"[autopilot] ckpt_dir resolved to file {ckpt_hint.name}; using parent {ckpt_dir} for new checkpoints")

    results_path = args.results.expanduser()
    ensure_results_header(results_path, RESULT_HEADER)

    stdout_dir = args.stdout_dir.expanduser()
    stdout_dir.mkdir(parents=True, exist_ok=True)

    patience_left = patience_budget
    best_metric = -1.0
    calibration_count = 0

    resume_state: Optional[ResumeState] = None
    start_round = 1
    if args.mode == "resume":
        resume_state = load_resume_state(results_path, args.target_metric, patience_budget)
        if resume_state:
            start_round = max(resume_state.next_round, 1)
            best_metric = resume_state.best_metric
            patience_left = resume_state.patience_left
            calibration_count = resume_state.calibration_count
            print(
                "[autopilot] resume state → "
                f"starting from round {start_round} "
                f"(best {args.target_metric}={best_metric:.4f}, patience={patience_left})"
            )

    for round_idx in range(start_round, args.max_rounds + 1):
        cfg = load_cfg()
        autopilot_cfg = cfg.setdefault("autopilot", {})
        best_sel_cfg = autopilot_cfg.setdefault("best_selection", {})
        owner_value = str(best_sel_cfg.get("owner", "autopilot")).lower()
        if owner_value not in {"autopilot", "train"}:
            owner_value = "autopilot"
        autopilot_is_owner = owner_value == "autopilot"
        best_owner = owner_value

        train_cfg = cfg.setdefault("training", {})
        metrics_cfg = train_cfg.setdefault("metrics", {})
        agg_cfg = metrics_cfg.setdefault("aggregation", {})
        k_cfg = agg_cfg.setdefault("k", {})
        if _coerce_positive_int(k_cfg.get("onset")) is None:
            k_cfg.setdefault("onset", 1)
        if "offset" not in k_cfg or not int(k_cfg.get("offset", 0)):
            k_cfg["offset"] = 1
        train_cfg["eval_freq"] = 1
        ckpt = find_ckpt(ckpt_dir)
        if ckpt is None and ckpt_hint and ckpt_hint.exists():
            ckpt = ckpt_hint
        if ckpt is not None:
            ckpt_hint = ckpt
        resume_requested = round_idx > 1 or args.mode == "resume"
        resuming = bool(resume_requested and ckpt is not None)
        train_cfg["resume"] = resuming
        current_epoch = 0
        if resuming:
            train_cfg["reset_head_bias"] = False
            current_epoch = infer_current_epoch(ckpt_dir)
        elif resume_requested and ckpt is None:
            print("[autopilot] resume requested but no checkpoint found; starting fresh")
        epochs_target = args.burst_epochs
        if resuming and current_epoch > 0:
            epochs_target = current_epoch + args.burst_epochs
        train_cfg["epochs"] = int(epochs_target)
        save_cfg(cfg)
        if resuming:
            sync_last_to_best(ckpt_dir)

        train_ret = 0
        last_val: Optional[str] = None
        training_executed = False
        metrics: Optional[Dict[str, float]] = None
        selection_result: Optional[SelectionResult] = None
        selection_context: Optional[SelectionContext] = None
        selection_lines: List[str] = []
        eval_ret = 0
        ckpt = find_ckpt(ckpt_dir)
        if ckpt is None and ckpt_hint and ckpt_hint.exists():
            ckpt = ckpt_hint
        if ckpt is not None:
            ckpt_hint = ckpt

        pre_round_calib = (
            round_idx == start_round
            and args.first_step == "calib"
            and ckpt is not None
            and not args.dry_run
        )
        if pre_round_calib:
            assert ckpt is not None
            selection_result, selection_context, selection_lines, eval_ret = perform_calibration(
                ckpt=ckpt,
                args=args,
                results_path=results_path,
                stdout_dir=stdout_dir,
                split=args.split_eval,
                calibration_count=calibration_count,
                seed=seed,
                deterministic=deterministic,
            )
            if selection_result is None:
                print("Calibration failed", file=sys.stderr)
                return eval_ret or 1
            calibration_count += 1
            metrics = _result_to_metrics(selection_result)
            apply_metrics_to_config(metrics)

        skip_training = (
            round_idx == start_round
            and args.skip_train_round1
            and ckpt is not None
            and args.first_step != "train"
        )
        if args.dry_run:
            banner = "NOTICE: Training burst — this may take a while"
            log_banner(results_path, banner)
            print("[autopilot] dry-run: skipping training execution")
            train_ret = 0
            last_val = None
        elif skip_training:
            print("[autopilot] skip_train_round1 enabled → skipping training for round 1")
        else:
            banner = "NOTICE: Training burst — this may take a while"
            log_banner(results_path, banner)
            log_path = stdout_dir / f"stdout_round{round_idx:02d}_train.txt"
            train_cmd = [sys.executable, str(TRAIN), "--config", str(CONFIG)]
            train_cmd = _append_determinism_flags(train_cmd, seed=seed, deterministic=deterministic)
            train_cmd = _with_verbose(train_cmd, args.verbose)
            train_ret, last_val, _ = run_command(
                train_cmd,
                log_path,
                capture_last_val=True,
                verbose=args.verbose,
            )
            training_executed = True

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
                    "k_onset": 1,
                    "onset_f1": 0.0,
                    "offset_f1": 0.0,
                    "onset_event_f1": 0.0,
                    "offset_event_f1": 0.0,
                    "ev_f1_mean": 0.0,
                    "onset_pred_rate": 0.0,
                    "onset_pos_rate": 0.0,
                    "decoder_onset_open": float("nan"),
                    "decoder_onset_hold": float("nan"),
                    "decoder_offset_open": float("nan"),
                    "decoder_offset_hold": float("nan"),
                    "decoder_onset_min_on": float("nan"),
                    "decoder_offset_min_off": float("nan"),
                    "decoder_onset_merge_gap": float("nan"),
                    "decoder_offset_merge_gap": float("nan"),
                },
                patience_left,
                cfg.get("experiment", {}).get("name", ""),
                last_val or "",
                train_ret,
            )
            return train_ret

        if args.dry_run:
            training_executed = True
        
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
                "k_onset": 1,
                "decoder_onset_open": 0.36,
                "decoder_onset_hold": 0.28,
                "decoder_offset_open": 0.32,
                "decoder_offset_hold": 0.24,
                "decoder_onset_min_on": 2,
                "decoder_offset_min_off": 2,
                "decoder_onset_merge_gap": 1,
                "decoder_offset_merge_gap": 1,
            }
            eval_ret = 0
        else:
            if training_executed:
                ckpt = find_ckpt(ckpt_dir)
            if ckpt is None and ckpt_hint and ckpt_hint.exists():
                ckpt = ckpt_hint
            if ckpt is not None:
                ckpt_hint = ckpt
            if ckpt is None:
                print(f"No checkpoint found in {ckpt_dir}", file=sys.stderr)
                return 1
            ckpt_used = ckpt
            need_post_calib = selection_result is None or training_executed
            if need_post_calib:
                selection_result, selection_context, selection_lines, eval_ret = perform_calibration(
                    ckpt=ckpt,
                    args=args,
                    results_path=results_path,
                    stdout_dir=stdout_dir,
                    split=args.split_eval,
                    calibration_count=calibration_count,
                    seed=seed,
                    deterministic=deterministic,
                )
                if selection_result is None:
                    print("Calibration failed", file=sys.stderr)
                    return eval_ret or 1
                calibration_count += 1
                metrics = _result_to_metrics(selection_result)
                apply_metrics_to_config(metrics)

        if metrics is None:
            print("Calibration metrics missing", file=sys.stderr)
            return 1

        val_line = format_val_line(metrics, last_val)
        metric_raw = metrics.get(target_metric_field)
        if metric_raw is None:
            metric_raw = metrics.get("ev_f1_mean", 0.0)
        metric_value = float(metric_raw)
        ev_mean = float(metrics["ev_f1_mean"])
        patience_record = patience_budget if metric_value > best_metric + 1e-9 else max(patience_left - 1, 0)
        append_results(
            results_path,
            round_idx,
            args.burst_epochs,
            ckpt_used,
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

        improved = metric_value > best_metric + 1e-9
        if improved:
            target_ckpt = ckpt_dir / "tivit_last.pt"
            if not target_ckpt.exists() or not training_executed:
                target_ckpt = ckpt_used
            best_ckpt = ckpt_dir / "tivit_best.pt"
            best_metric = metric_value
            patience_left = patience_budget
            if autopilot_is_owner:
                if selection_result is not None and selection_context is not None:
                    record_best(
                        source=target_ckpt,
                        destination=best_ckpt,
                        result=selection_result,
                        context=selection_context,
                        repo_root=REPO,
                    )
                else:
                    tmp_ckpt = best_ckpt.with_name(best_ckpt.name + ".tmp")
                    try:
                        tmp_ckpt.unlink(missing_ok=True)
                    except OSError:
                        pass
                    shutil.copy2(target_ckpt, tmp_ckpt)
                    os.replace(tmp_ckpt, best_ckpt)
                    print(
                        "[autopilot] WARNING: selection result missing; copied checkpoint without metadata",
                        flush=True,
                    )
            print(
                f"[autopilot] New best {args.target_metric}={metric_value:.4f} "
                f"(round {round_idx}); ev_f1_mean={ev_mean:.4f} owner={best_owner}"
            )
        else:
            patience_left = patience_record
            print(
                f"[autopilot] {args.target_metric}={metric_value:.4f} "
                f"(best={best_metric:.4f}), ev_f1_mean={ev_mean:.4f} "
                f"patience_left={patience_left}"
            )

        if metric_value >= args.target_ev_f1:
            print("SUCCESS: target reached")
            return 0
        if patience_left <= 0:
            print(f"EARLY STOP: no improvement for {patience_budget} rounds")
            return 0


    print("Reached max rounds without meeting target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
