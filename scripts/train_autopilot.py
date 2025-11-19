#!/usr/bin/env python3
"""Automated training and calibration driver for TiViT-Piano.

Purpose:
    Orchestrate repeated "train → calibrate → evaluate" rounds until the target
    event-level F1 is achieved or patience expires. The script keeps
    ``configs/config.yaml`` synchronized with calibration results, mirrors
    helper stdout/stderr to logs, and records each round in a TSV ledger. It
    can resume from an existing experiment, jump straight to calibration, or run
    a two-stage onset decoder/threshold sweep when requested to stabilize
    event-F1 improvements. When thorough passes fail, it falls back to coarse
    sweeps.

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
    --fast_strategy {classic,two_stage} (default: two_stage)
        Select the fast calibration routine (legacy sweep or two-stage onset optimizer).
    --postproc-mode {never,eval_only,always} (default: eval_only)
        Forward decoder post-processing strategy to ``eval_thresholds.py`` so
        sweeps can skip DP/snap while final evals still apply them.
    --model-return-per-tile / --no-model-return-per-tile (default: disabled)
        When enabled, forward ``--model-return-per-tile`` to ``eval_thresholds.py`` so
        per-tile logits and boundary diagnostics are emitted during eval.
    --onset_open_grid FLOAT [FLOAT ...]
        Candidate onset decoder open gates explored during Stage-A (default: 0.22–0.28).
    --onset_min_on_grid INT [INT ...]
        Candidate onset decoder ``min_on`` values for Stage-A (default: 2, 3).
    --onset_hold_mode {fixed_delta,config}
        Hold-gate policy for Stage-A (default: fixed_delta → open − 0.06).
    --onset_thr_delta FLOAT (default: 0.05)
        Radius around the anchor used for the Stage-B micro-sweep.
    --onset_thr_steps INT (default: 5)
        Number of samples evaluated in the Stage-B micro-sweep.
    --decoder-impl {new,legacy} (default: new)
        Choose the decoder/evaluation stack; ``legacy`` toggles the rollback
        flags to run the pre-refactor calibrate/eval scripts.

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

# Flags from eval_thresholds.py that never take an explicit value.  When users
# pass extras without the leading "--" (for example "no_eval_cache"), we only
# auto-prefix them when they are not acting as a value for one of these.
EVAL_THRESHOLD_VALUELESS_FLAGS = {
    "--grid_prob_thresholds",
    "--no_eval_cache",
    "--sweep_k_onset",
    "--no-avlag",
    "--legacy-eval-thresholds",
    "--model-return-per-tile",
    "--progress",
    "--no-progress",
    "--deterministic",
    "--no-deterministic",
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
    parse_eval_rows,
)
from utils.tie_break import OnsetTieBreakContext, TIE_BREAK_EPS_F1, log_tie_break_eps, select_best_onset_row

LOGGER = logging.getLogger("autopilot")
QUIET_EXTRA = {QUIET_INFO_FLAG: True}


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_cfg() -> dict:
    with CONFIG.open("r") as f:
        return yaml.safe_load(f)


def _normalize_eval_extras(tokens: Sequence[str]) -> List[str]:
    """Ensure eval extras look like CLI flags, but keep flag values untouched."""
    normalized: List[str] = []
    expecting_value = False
    for token in tokens:
        stripped = token.strip()
        if not stripped:
            continue
        if expecting_value:
            normalized.append(stripped)
            expecting_value = False
            continue
        if stripped.startswith("-"):
            normalized.append(stripped)
            takes_value = (
                stripped.startswith("--")
                and "=" not in stripped
                and stripped not in EVAL_THRESHOLD_VALUELESS_FLAGS
            )
            expecting_value = takes_value
            continue
        normalized.append(f"--{stripped}")
        expecting_value = False
    return normalized


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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    tmp.replace(path)


def save_cfg(cfg: dict) -> None:
    buffer = io.StringIO()
    yaml.safe_dump(cfg, buffer, sort_keys=False)
    rendered = _inject_decoder_comments(buffer.getvalue())
    _atomic_write_text(CONFIG, rendered)


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
    use_legacy_decoder: bool = False,
) -> int:
    log_name = f"calibration_{kind}.txt"
    log_path = log_dir / log_name
    dataset_cli = _build_dataset_cli(split, frames, max_clips)
    cmd = [sys.executable, "-u", str(CALIBRATE_THRESH), *dataset_cli, "--ckpt", str(ckpt)]
    if use_legacy_decoder:
        cmd.append("--legacy-calibrate-thresholds")
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

DEFAULT_FAST_STRATEGY = "two_stage"
DEFAULT_ONSET_OPEN_GRID = [0.018, 0.020, 0.022, 0.024]
DEFAULT_ONSET_MIN_ON_GRID = [2, 3]
DEFAULT_ONSET_HOLD_MODE = "fixed_ratio"
DEFAULT_ONSET_HOLD_DELTA = 0.06
DEFAULT_ONSET_HOLD_MIN = 0.015
DEFAULT_ONSET_HOLD_RATIO = 0.75
DEFAULT_ONSET_MERGE_GAP = 1
DEFAULT_ONSET_THR_ANCHOR = 0.20
DEFAULT_ONSET_THR_DELTA = 0.05
DEFAULT_ONSET_THR_STEPS = 5
DEFAULT_OFFSET_THR_ANCHOR = 0.30
ONSET_OPEN_MIN = 0.01
ONSET_OPEN_MAX = 0.08
ONSET_THR_MIN = 0.10
ONSET_THR_MAX = 0.40
ONSET_PRED_RATE_MIN_FACTOR = 0.5
ONSET_PRED_RATE_MAX_FACTOR = 3.0
ONSET_TIE_TOL = TIE_BREAK_EPS_F1

DEFAULT_STAGEB_ADD_DELTA = 0.01
DEFAULT_STAGEB_ADD_STEPS = 5
DEFAULT_STAGEB_MUL_RATIO = 1.10
DEFAULT_STAGEB_MUL_ORDERS = 1
DEFAULT_STAGEB_MIN_PROB = 0.02
DEFAULT_STAGEB_MAX_PROB = 0.98
DEFAULT_STAGEB_LOW_GUARD = 0.05
DEFAULT_STAGEB_HIGH_GUARD = 0.95
DEFAULT_STAGEB_MIN_POINTS = 5
DEFAULT_STAGEB_MAX_POINTS = 11
STAGEB_PROB_FMT = "{:.4f}"
STAGEB_PROB_TOL = 1e-6

DEFAULT_ONSET_STAGEB_LOW_GUARD = ONSET_THR_MIN
DEFAULT_ONSET_STAGEB_HIGH_GUARD = 0.60
DEFAULT_OFFSET_STAGEB_LOW_GUARD = 0.02
DEFAULT_OFFSET_STAGEB_HIGH_GUARD = 0.95

THOROUGH_CACHE_FILENAME = "stageB_thorough_cache.json"
STAGEB_CANDIDATES_TEMPLATE = "stageB_round{round:02d}_calib{calib:02d}_candidates.json"
STAGEB_STATE_TEMPLATE = "stageB_round{round:02d}_calib{calib:02d}_state.json"
ROUND_STATE_TEMPLATE = "round{round:02d}_state.json"

ROUND_PHASE_STAGEA_DONE = "stageA_done"
ROUND_PHASE_STAGEB_STARTED = "stageB_started"
ROUND_PHASE_STAGEB_DONE = "stageB_done"


@dataclass
class StageBParams:
    add_delta: float
    add_steps: int
    mul_ratio: float
    mul_orders: int
    min_prob: float
    max_prob: float
    low_guard: Optional[float]
    high_guard: Optional[float]
    min_points: int
    max_points: Optional[int]
    explicit: Optional[List[float]]


@dataclass
class StageBAnchor:
    prob: float
    source: str
    details: Dict[str, Any]
    raw_prob: float

    def __post_init__(self) -> None:
        try:
            raw_val = float(self.raw_prob)
        except (TypeError, ValueError):
            raw_val = float(self.prob)
        if not math.isfinite(raw_val):
            raw_val = float(self.prob)
        self.raw_prob = raw_val


def _normalize_probability_list(values: Iterable[float], *, lo: float, hi: float) -> List[float]:
    seen: Dict[int, float] = {}
    for raw in values or []:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        val = max(lo, min(hi, val))
        key = int(round(val * 1000))
        if key not in seen:
            seen[key] = val
    return sorted(seen.values())


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


def _stageb_cache_path(stdout_dir: Path) -> Path:
    return stdout_dir / THOROUGH_CACHE_FILENAME


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _load_stageb_cache(stdout_dir: Path) -> Dict[str, Any]:
    path = _stageb_cache_path(stdout_dir)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _extract_thorough_anchor(entry: Mapping[str, Any] | None, *, provenance: str) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, Mapping):
        return None
    best_prob = _coerce_optional_float(entry.get("best_prob"))
    if best_prob is None:
        return None
    payload: Dict[str, Any] = {
        "prob": _clamp_probability(best_prob),
        "source": provenance,
    }
    temp_val = (
        _coerce_optional_float(entry.get("temperature"))
        or _coerce_optional_float(entry.get("platt_scale"))
    )
    if temp_val is not None:
        payload["temperature"] = temp_val
    bias_val = (
        _coerce_optional_float(entry.get("logit_bias"))
        or _coerce_optional_float(entry.get("platt_bias"))
    )
    if bias_val is not None:
        payload["bias"] = bias_val
    return payload


def _update_stageb_cache_from_calib(stdout_dir: Path, calib: Mapping[str, Any]) -> Dict[str, Any]:
    existing = _load_stageb_cache(stdout_dir)
    payload: Dict[str, Any] = dict(existing)
    anchors: Dict[str, Any] = dict(payload.get("anchors", {}))
    changed = False
    for head in ("onset", "offset"):
        entry = calib.get(head) if isinstance(calib, Mapping) else None
        anchor = _extract_thorough_anchor(entry, provenance="calibration.json")
        if anchor is None:
            continue
        anchors[head] = anchor
        changed = True
    if changed:
        payload["anchors"] = anchors
        payload["timestamp"] = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        _atomic_write_json(_stageb_cache_path(stdout_dir), payload)
        return payload
    return existing


def _extract_anchor_platt(anchor: StageBAnchor) -> Tuple[Optional[float], Optional[float]]:
    details = anchor.details if isinstance(anchor.details, Mapping) else {}
    temp_val: Optional[float] = None
    bias_val: Optional[float] = None
    for key in ("temperature", "platt_scale", "scale"):
        candidate = details.get(key)
        temp_val = _coerce_optional_float(candidate)
        if temp_val is not None:
            break
    for key in ("bias", "platt_bias", "logit_bias"):
        candidate = details.get(key)
        bias_val = _coerce_optional_float(candidate)
        if bias_val is not None:
            break
    return temp_val, bias_val


_THOROUGH_LINE_RE = re.compile(
    r"^(Onset|Offset):.*best_prob=(?P<prob>[0-9]*\.?[0-9]+).*temp=(?P<temp>[+-]?\d+(?:\.\d+)?).*bias=(?P<bias>[+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_thorough_log(stdout_dir: Path) -> Dict[str, Dict[str, Any]]:
    log_path = stdout_dir / "calibration_thorough.txt"
    if not log_path.exists():
        return {}
    anchors: Dict[str, Dict[str, Any]] = {}
    try:
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                match = _THOROUGH_LINE_RE.search(line)
                if not match:
                    continue
                head = match.group(1).lower()
                prob_val = _coerce_optional_float(match.group("prob"))
                if prob_val is None:
                    continue
                anchor: Dict[str, Any] = {
                    "prob": _clamp_probability(prob_val),
                    "source": "calibration_thorough.txt",
                }
                temp_val = _coerce_optional_float(match.group("temp"))
                if temp_val is not None:
                    anchor["temperature"] = temp_val
                bias_val = _coerce_optional_float(match.group("bias"))
                if bias_val is not None:
                    anchor["bias"] = bias_val
                anchors[head] = anchor
    except OSError:
        return {}
    return anchors


def _build_anchor_from_entry(entry: Mapping[str, Any] | None, *, params: StageBParams) -> Optional[StageBAnchor]:
    if not isinstance(entry, Mapping):
        return None
    prob_val = _coerce_optional_float(entry.get("prob"))
    if prob_val is None:
        return None
    raw_prob = float(prob_val)
    bounded_prob = max(params.min_prob, min(params.max_prob, raw_prob))
    details: Dict[str, Any] = {"raw_prob": raw_prob}
    for key in ("temperature", "bias", "platt_scale", "platt_bias", "logit_bias"):
        val = entry.get(key)
        coerced = _coerce_optional_float(val)
        if coerced is not None:
            details[key] = coerced
    source = str(entry.get("source") or "unknown")
    return StageBAnchor(prob=bounded_prob, source=source, details=details, raw_prob=raw_prob)


def _resolve_stageb_anchor(
    head: str,
    *,
    params: StageBParams,
    thorough_cache: Mapping[str, Any],
    log_fallback: Mapping[str, Any],
    previous_best: Optional[float],
    config_anchor: float,
) -> StageBAnchor:
    anchors_map = thorough_cache.get("anchors") if isinstance(thorough_cache, Mapping) else None
    if isinstance(anchors_map, Mapping):
        entry = anchors_map.get(head)
        anchor = _build_anchor_from_entry(entry, params=params)
        if anchor:
            anchor.details.setdefault("origin", "thorough_cache")
            return anchor
    entry = log_fallback.get(head)
    anchor = _build_anchor_from_entry(entry, params=params)
    if anchor:
        anchor.details.setdefault("origin", "thorough_log")
        return anchor
    if previous_best is not None:
        raw_prev = float(previous_best)
        prob = max(params.min_prob, min(params.max_prob, raw_prev))
        return StageBAnchor(
            prob=prob,
            source="previous",
            details={"previous_round": previous_best, "raw_prob": raw_prev},
            raw_prob=raw_prev,
        )
    raw_cfg = float(config_anchor)
    prob = max(params.min_prob, min(params.max_prob, raw_cfg))
    return StageBAnchor(
        prob=prob,
        source="config",
        details={"config_anchor": config_anchor, "raw_prob": raw_cfg},
        raw_prob=raw_cfg,
    )


def _dedupe_probabilities(values: Iterable[float]) -> List[float]:
    processed: List[float] = []
    for raw in values:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        processed.append(val)
    processed.sort()
    deduped: List[float] = []
    for val in processed:
        if not deduped or abs(val - deduped[-1]) > 1e-6:
            deduped.append(val)
    return deduped


def _clamp_and_filter_candidates(values: Iterable[float], params: StageBParams) -> Tuple[List[float], float, float]:
    processed = _dedupe_probabilities(values)
    bounded = [max(params.min_prob, min(params.max_prob, val)) for val in processed]
    guard_min = params.low_guard if params.low_guard is not None else params.min_prob
    guard_max = params.high_guard if params.high_guard is not None else params.max_prob
    guard_min = max(params.min_prob, min(params.max_prob, float(guard_min)))
    guard_max = max(params.min_prob, min(params.max_prob, float(guard_max)))
    if guard_min > guard_max:
        guard_min = guard_max
    filtered = [val for val in bounded if guard_min - 1e-6 <= val <= guard_max + 1e-6]
    return _dedupe_probabilities(filtered), guard_min, guard_max


def _pad_candidates(
    candidates: List[float],
    anchor: float,
    params: StageBParams,
    guard_min: float,
    guard_max: float,
) -> List[float]:
    target = max(1, params.min_points)
    if len(candidates) >= target:
        return _dedupe_probabilities(candidates)
    delta = params.add_delta if params.add_delta > 0 else (guard_max - guard_min) / max(target, 1)
    delta = max(delta, 1e-3)
    working = list(candidates)
    seen = {int(round(val * 1e6)) for val in working}
    step = 1
    max_iters = 512
    while len(working) < target and step < max_iters:
        added = False
        lower = anchor - step * delta
        if lower >= guard_min - 1e-6:
            lower_clamped = max(guard_min, min(guard_max, lower))
            key = int(round(lower_clamped * 1e6))
            if key not in seen:
                working.append(lower_clamped)
                seen.add(key)
                added = True
        upper = anchor + step * delta
        if len(working) < target and upper <= guard_max + 1e-6:
            upper_clamped = max(guard_min, min(guard_max, upper))
            key = int(round(upper_clamped * 1e6))
            if key not in seen:
                working.append(upper_clamped)
                seen.add(key)
                added = True
        if not added:
            break
        step += 1
    return _dedupe_probabilities(working)


def _trim_candidates(candidates: List[float], anchor: float, max_points: Optional[int]) -> List[float]:
    limit = max_points if max_points is None else int(max_points)
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return _dedupe_probabilities(candidates)
    ordered = _dedupe_probabilities(candidates)
    if not ordered:
        return ordered
    anchor_idx = min(range(len(ordered)), key=lambda idx: (abs(ordered[idx] - anchor), -ordered[idx]))
    picked = {anchor_idx}
    left = anchor_idx - 1
    right = anchor_idx + 1
    prefer_high_extra = limit % 2 == 0
    while len(picked) < limit and (left >= 0 or right < len(ordered)):
        options = []
        if left >= 0:
            options.append(("left", abs(ordered[left] - anchor), ordered[left]))
        if right < len(ordered):
            options.append(("right", abs(ordered[right] - anchor), ordered[right]))
        if not options:
            break
        force_high_side = prefer_high_extra and len(picked) == limit - 1
        chosen = None
        if force_high_side:
            for option in options:
                if option[0] == "right":
                    chosen = option
                    break
        if chosen is None:
            options.sort(key=lambda item: (item[1], -item[2]))
            chosen = options[0]
        side = chosen[0]
        if side == "left":
            picked.add(left)
            left -= 1
        else:
            picked.add(right)
            right += 1
        if len(picked) >= limit:
            break
    return sorted(ordered[idx] for idx in picked)


def _generate_stageb_candidates(anchor: StageBAnchor, params: StageBParams) -> Tuple[List[float], float, float, float]:
    base_values: List[float] = []
    anchor_raw = _anchor_raw_probability(anchor)
    anchor_used = max(params.min_prob, min(params.max_prob, float(anchor.prob)))
    if params.explicit:
        base_values = [float(val) for val in params.explicit]
    else:
        steps = params.add_steps if params.add_steps > 0 else 1
        if steps % 2 == 0:
            steps += 1
        half = steps // 2
        for idx in range(-half, half + 1):
            base_values.append(anchor_used + idx * params.add_delta)
        ratio = params.mul_ratio if params.mul_ratio and params.mul_ratio > 0 else 1.0
        if params.mul_orders > 0 and abs(ratio - 1.0) > 1e-9:
            for order in range(-params.mul_orders, params.mul_orders + 1):
                base_values.append(anchor_used * (ratio ** order))
    base_values.append(anchor_used)
    candidates, guard_min, guard_max = _clamp_and_filter_candidates(base_values, params)
    if not any(abs(anchor_used - val) <= STAGEB_PROB_TOL for val in candidates):
        candidates.append(anchor_used)
    candidates = _dedupe_probabilities(candidates)
    candidates = _pad_candidates(candidates, anchor_used, params, guard_min, guard_max)
    candidates = _trim_candidates(candidates, anchor_used, params.max_points)
    return candidates, anchor_used, guard_min, guard_max


def _round_state_path(stdout_dir: Path, round_idx: int) -> Path:
    return stdout_dir / ROUND_STATE_TEMPLATE.format(round=round_idx)


def _load_round_state(stdout_dir: Path, round_idx: int) -> Dict[str, Any]:
    path = _round_state_path(stdout_dir, round_idx)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _save_round_state(stdout_dir: Path, round_idx: int, state: Mapping[str, Any]) -> None:
    _atomic_write_json(_round_state_path(stdout_dir, round_idx), state)


def _extract_offset_gates(decoder_snapshot: Mapping[str, Any] | None) -> Dict[str, Optional[float]]:
    if not isinstance(decoder_snapshot, Mapping):
        return {}
    offset_raw = decoder_snapshot.get("offset")
    if not isinstance(offset_raw, Mapping):
        return {}
    return {
        "open": _coerce_optional_float(offset_raw.get("open")),
        "hold": _coerce_optional_float(offset_raw.get("hold")),
        "min_off": _coerce_positive_int(offset_raw.get("min_off")),
        "merge_gap": _coerce_positive_int(offset_raw.get("merge_gap")),
    }


def _format_prob(value: float) -> str:
    return STAGEB_PROB_FMT.format(float(value))


def _format_candidates(values: Sequence[float]) -> str:
    return "[" + ",".join(_format_prob(val) for val in values) + "]"


def _contains_probability(values: Sequence[float], target: float, *, tol: float = STAGEB_PROB_TOL) -> bool:
    for val in values:
        if abs(float(val) - float(target)) <= tol:
            return True
    return False


def _anchor_raw_probability(anchor: StageBAnchor) -> float:
    raw_value = getattr(anchor, "raw_prob", None)
    try:
        return float(raw_value) if raw_value is not None else float(anchor.prob)
    except (TypeError, ValueError):
        return float(anchor.prob)


def _write_stageb_artifacts(
    stdout_dir: Path,
    *,
    round_idx: int,
    calib_index: int,
    anchors: Dict[str, StageBAnchor],
    candidates: Dict[str, Sequence[float]],
    tie_break: Optional[str],
    winner_row: Mapping[str, Any],
    gates: Mapping[str, Any],
    anchor_retained: Mapping[str, bool],
) -> None:
    timestamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    anchors_payload = {
        head: {
            "prob": anchor.prob,
            "raw_prob": _anchor_raw_probability(anchor),
            "source": anchor.source,
            "details": anchor.details,
        }
        for head, anchor in anchors.items()
    }
    candidates_payload = {
        head: [float(val) for val in seq]
        for head, seq in candidates.items()
    }
    base_payload = {
        "timestamp": timestamp,
        "round": round_idx,
        "calibration_index": calib_index,
        "anchors": anchors_payload,
        "candidates": candidates_payload,
        "tie_break": tie_break,
    }
    winner_payload = {
        "onset_thr": _coerce_optional_float(winner_row.get("onset_thr")),
        "offset_thr": _coerce_optional_float(winner_row.get("offset_thr")),
        "onset_event_f1": _coerce_optional_float(winner_row.get("onset_event_f1")),
        "offset_event_f1": _coerce_optional_float(winner_row.get("offset_event_f1")),
        "onset_pred_rate": _coerce_optional_float(winner_row.get("onset_pred_rate")),
        "onset_pos_rate": _coerce_optional_float(winner_row.get("onset_pos_rate")),
        "ev_f1_mean": _coerce_optional_float(winner_row.get("ev_f1_mean")),
    }
    base_payload["winner_row"] = winner_payload
    base_payload["gates"] = copy.deepcopy(gates)
    base_payload["anchor_retained"] = {head: bool(flag) for head, flag in anchor_retained.items()}

    candidates_path = stdout_dir / STAGEB_CANDIDATES_TEMPLATE.format(round=round_idx, calib=calib_index)
    state_path = stdout_dir / STAGEB_STATE_TEMPLATE.format(round=round_idx, calib=calib_index)
    _atomic_write_json(candidates_path, base_payload)
    _atomic_write_json(state_path, base_payload)


def _resolve_stageb_params_from_cfg(
    cfg: Mapping[str, Any] | None,
    *,
    explicit_key: str,
    low_guard_default: float,
    high_guard_default: float,
) -> StageBParams:
    cfg = cfg or {}
    add_delta = _coerce_optional_float(cfg.get("thr_add_delta"))
    if add_delta is None:
        add_delta = DEFAULT_STAGEB_ADD_DELTA
    add_delta = max(0.0, float(add_delta))

    add_steps_val = cfg.get("thr_add_steps", DEFAULT_STAGEB_ADD_STEPS)
    add_steps = _coerce_positive_int(add_steps_val) or int(add_steps_val or DEFAULT_STAGEB_ADD_STEPS)
    if add_steps < 1:
        add_steps = 1
    if add_steps % 2 == 0:
        add_steps += 1

    mul_ratio_val = _coerce_optional_float(cfg.get("thr_mul_ratio"))
    mul_ratio = float(mul_ratio_val) if mul_ratio_val is not None else DEFAULT_STAGEB_MUL_RATIO
    if not math.isfinite(mul_ratio) or mul_ratio <= 1.0:
        mul_ratio = 1.0

    mul_orders_val = cfg.get("thr_mul_orders")
    if mul_orders_val is None:
        mul_orders = DEFAULT_STAGEB_MUL_ORDERS
    else:
        try:
            mul_orders = int(mul_orders_val)
        except (TypeError, ValueError):
            mul_orders = DEFAULT_STAGEB_MUL_ORDERS
        mul_orders = max(0, mul_orders)

    min_prob_val = _coerce_optional_float(cfg.get("thr_min_prob"))
    if min_prob_val is None:
        min_prob_val = DEFAULT_STAGEB_MIN_PROB
    min_prob = max(0.0, min(1.0, float(min_prob_val)))

    max_prob_val = _coerce_optional_float(cfg.get("thr_max_prob"))
    if max_prob_val is None:
        max_prob_val = DEFAULT_STAGEB_MAX_PROB
    max_prob = max(min_prob, min(1.0, float(max_prob_val)))

    low_guard_val = _coerce_optional_float(cfg.get("thr_low_guard"))
    if low_guard_val is None:
        low_guard_val = low_guard_default
    low_guard = None if low_guard_val is None else max(min_prob, min(max_prob, float(low_guard_val)))

    high_guard_val = _coerce_optional_float(cfg.get("thr_high_guard"))
    if high_guard_val is None:
        high_guard_val = high_guard_default
    high_guard = None if high_guard_val is None else max(min_prob, min(max_prob, float(high_guard_val)))
    if high_guard is not None and low_guard is not None and high_guard < low_guard:
        low_guard = high_guard

    min_points_val = cfg.get("thr_min_points", DEFAULT_STAGEB_MIN_POINTS)
    try:
        min_points = int(min_points_val)
    except (TypeError, ValueError):
        min_points = DEFAULT_STAGEB_MIN_POINTS
    if min_points < 1:
        min_points = 1

    max_points_val = cfg.get("thr_max_points", DEFAULT_STAGEB_MAX_POINTS)
    max_points: Optional[int]
    if max_points_val is None:
        max_points = None
    else:
        try:
            max_points_candidate = int(max_points_val)
        except (TypeError, ValueError):
            max_points_candidate = DEFAULT_STAGEB_MAX_POINTS
        max_points = None if max_points_candidate <= 0 else max(max_points_candidate, min_points)

    explicit = cfg.get(explicit_key)
    if explicit is None:
        explicit = cfg.get("thr_explicit")
    explicit_list: Optional[List[float]] = None
    if isinstance(explicit, (list, tuple)):
        explicit_list = _dedupe_probabilities(explicit)

    return StageBParams(
        add_delta=add_delta,
        add_steps=add_steps,
        mul_ratio=mul_ratio,
        mul_orders=mul_orders,
        min_prob=min_prob,
        max_prob=max_prob,
        low_guard=low_guard,
        high_guard=high_guard,
        min_points=min_points,
        max_points=max_points,
        explicit=explicit_list,
    )


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


def _stringify_diff_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, str)):
        return str(value)
    return repr(value)


def _nested_lookup(mapping: Mapping[str, Any] | None, path: Sequence[str]) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _log_metrics_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    tracked_paths: List[Sequence[str]] = [
        ("prob_threshold_onset",),
        ("prob_threshold_offset",),
        ("decoder", "onset", "open"),
        ("decoder", "onset", "hold"),
        ("decoder", "onset", "min_on"),
        ("decoder", "onset", "merge_gap"),
    ]
    diffs: List[str] = []
    for path in tracked_paths:
        old_val = _nested_lookup(before, path)
        new_val = _nested_lookup(after, path)
        if old_val == new_val:
            continue
        dotted = ".".join(path)
        diffs.append(f"{dotted}: {_stringify_diff_value(old_val)} → {_stringify_diff_value(new_val)}")
    if diffs:
        print("[autopilot] config diff: " + "; ".join(diffs))


def apply_metrics_to_config(metrics: Dict[str, float]) -> None:
    cfg = load_cfg()
    train_cfg = cfg.setdefault("training", {})
    metrics_cfg = train_cfg.setdefault("metrics", {})
    metrics_before = copy.deepcopy(metrics_cfg)
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
    onset_open_new = _coerce_optional_float(metrics.get("decoder_onset_open"))
    onset_hold_new = _coerce_optional_float(metrics.get("decoder_onset_hold"))
    onset_min_on_new = _coerce_positive_int(metrics.get("decoder_onset_min_on"))
    onset_merge_gap_new = _coerce_positive_int(metrics.get("decoder_onset_merge_gap"))
    if onset_open_new is not None or onset_hold_new is not None or onset_min_on_new is not None or onset_merge_gap_new is not None:
        if isinstance(decoder_snapshot, Mapping):
            decoder_dict: Dict[str, Any] = copy.deepcopy(dict(decoder_snapshot))
        else:
            decoder_dict = {}
        onset_decoder_cfg = decoder_dict.setdefault("onset", {})
        if onset_open_new is not None:
            onset_decoder_cfg["open"] = max(0.0, min(1.0, float(onset_open_new)))
        if onset_hold_new is not None:
            hold_val = max(0.0, float(onset_hold_new))
            open_cap = onset_decoder_cfg.get("open")
            if isinstance(open_cap, (int, float)):
                hold_val = min(hold_val, float(open_cap))
            onset_decoder_cfg["hold"] = hold_val
        if onset_min_on_new is not None:
            onset_decoder_cfg["min_on"] = int(max(0, onset_min_on_new))
        if onset_merge_gap_new is not None:
            onset_decoder_cfg["merge_gap"] = int(max(0, onset_merge_gap_new))
        decoder_dict.setdefault("offset", decoder_dict.get("offset", {}))
        decoder_snapshot = decoder_dict

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
    _log_metrics_diff(metrics_before, metrics_cfg)


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


def _extract_partial_platt(mapping: Mapping[str, Any] | None) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(mapping, Mapping):
        return None, None
    temp_val = (
        _coerce_optional_float(mapping.get("temperature"))
        or _coerce_optional_float(mapping.get("platt_scale"))
    )
    bias_val = (
        _coerce_optional_float(mapping.get("logit_bias"))
        or _coerce_optional_float(mapping.get("platt_bias"))
    )
    return temp_val, bias_val


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
    target_metric_field: str,
    calibration_count: int,
    seed: int,
    deterministic: bool,
    round_index: int,
) -> Tuple[Optional[SelectionResult], Optional[SelectionContext], List[str], int]:
    first_calibration = calibration_count == 0
    prefer_fast_grid = first_calibration and args.fast_first_calib
    desired_kind = args.first_calib if first_calibration else "fast"

    cfg_snapshot = load_cfg()
    prev_metrics = (cfg_snapshot.get("training", {}) or {}).get("metrics", {}) or {}
    prev_onset_best = _coerce_optional_float(prev_metrics.get("prob_threshold_onset"))
    prev_offset_best = _coerce_optional_float(prev_metrics.get("prob_threshold_offset"))

    autop_cfg = cfg_snapshot.get("autopilot", {}) if isinstance(cfg_snapshot, Mapping) else {}
    onset_stageb_cfg = autop_cfg.get("onset_optimizer", {}) if isinstance(autop_cfg, Mapping) else {}
    offset_stageb_cfg = autop_cfg.get("offset_optimizer", {}) if isinstance(autop_cfg, Mapping) else {}

    onset_stageb_params = _resolve_stageb_params_from_cfg(
        onset_stageb_cfg,
        explicit_key="thr_explicit_onset",
        low_guard_default=DEFAULT_ONSET_STAGEB_LOW_GUARD,
        high_guard_default=DEFAULT_ONSET_STAGEB_HIGH_GUARD,
    )
    offset_stageb_params = _resolve_stageb_params_from_cfg(
        offset_stageb_cfg,
        explicit_key="thr_explicit_offset",
        low_guard_default=DEFAULT_OFFSET_STAGEB_LOW_GUARD,
        high_guard_default=DEFAULT_OFFSET_STAGEB_HIGH_GUARD,
    )

    offset_anchor_cfg = _coerce_optional_float((offset_stageb_cfg or {}).get("thr_anchor"))
    offset_thr_anchor = offset_anchor_cfg if offset_anchor_cfg is not None else DEFAULT_OFFSET_THR_ANCHOR

    thorough_cache = _load_stageb_cache(stdout_dir)

    decoder_snapshot = decoder_snapshot_from_config(cfg_snapshot)
    tolerance_snapshot = tolerance_snapshot_from_config(cfg_snapshot)
    exp_name = cfg_snapshot.get("experiment", {}).get("name", "")
    prob_delta = 0.05
    legacy_decoder = args.decoder_impl == "legacy"

    def _build_request(
        onset_center: float,
        offset_center: float,
        *,
        log_name: str,
        sweep_override: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        extras: Optional[Sequence[str]] = None,
        low_guard: float = 0.10,
        platt_overrides: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
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
        def _strip_eval_flag(flag: str) -> None:
            needle = f"{flag}="
            idx = 0
            while idx < len(eval_extras):
                token = eval_extras[idx]
                if token == flag:
                    del eval_extras[idx]
                    if idx < len(eval_extras) and not eval_extras[idx].startswith("--"):
                        del eval_extras[idx]
                    continue
                if token.startswith(needle):
                    del eval_extras[idx]
                    continue
                idx += 1
        _strip_eval_flag("--postproc-mode")
        eval_extras.extend(["--postproc-mode", args.postproc_mode])
        if args.model_return_per_tile:
            _strip_eval_flag("--model-return-per-tile")
            eval_extras.append("--model-return-per-tile")
        if legacy_decoder and "--legacy-eval-thresholds" not in eval_extras:
            eval_extras.append("--legacy-eval-thresholds")
        if extras:
            eval_extras.extend(extras)
        frames = args.calib_frames if args.calib_frames is not None else 96
        max_clips = args.calib_max_clips if args.calib_max_clips is not None else 80
        def _resolve_platt_tuple(head: str) -> Tuple[Optional[float], Optional[float]]:
            if not platt_overrides:
                return (None, None)
            values = platt_overrides.get(head)
            if isinstance(values, tuple):
                return values
            return (None, None)

        onset_platt_vals = _resolve_platt_tuple("onset")
        offset_platt_vals = _resolve_platt_tuple("offset")
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
            temperature_onset=onset_platt_vals[0],
            bias_onset=onset_platt_vals[1],
            temperature_offset=offset_platt_vals[0],
            bias_offset=offset_platt_vals[1],
            seed=seed,
            deterministic=deterministic,
            eval_extras=eval_extras,
            verbose=args.verbose,
            log_path=stdout_dir / log_name,
            run_id=exp_name,
            target_metric=target_metric_field,
            onset_anchor_used=onset_center,
            prev_onset_threshold=prev_onset_best,
        )

    def _run_two_stage(calib: dict, *, log_name: str) -> Tuple[SelectionResult, SelectionContext, List[str]]:
        log_fallback = _parse_thorough_log(stdout_dir)

        onset_anchor = _resolve_stageb_anchor(
            "onset",
            params=onset_stageb_params,
            thorough_cache=thorough_cache,
            log_fallback=log_fallback,
            previous_best=prev_onset_best,
            config_anchor=args.onset_thr_anchor,
        )
        offset_anchor = _resolve_stageb_anchor(
            "offset",
            params=offset_stageb_params,
            thorough_cache=thorough_cache,
            log_fallback=log_fallback,
            previous_best=prev_offset_best,
            config_anchor=offset_thr_anchor,
        )
        onset_prob = float(onset_anchor.prob)
        offset_prob = float(offset_anchor.prob)
        anchor_source = (
            onset_anchor.source
            if onset_anchor.source == offset_anchor.source
            else f"onset:{onset_anchor.source},offset:{offset_anchor.source}"
        )

        needs_temp = args.temperature is None
        needs_bias = args.bias is None
        platt_overrides: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
        onset_platt_temp: Optional[float] = None
        onset_platt_bias: Optional[float] = None
        offset_platt_temp: Optional[float] = None
        offset_platt_bias: Optional[float] = None
        if needs_temp or needs_bias:
            onset_platt_temp, onset_platt_bias = _extract_anchor_platt(onset_anchor)
            offset_platt_temp, offset_platt_bias = _extract_anchor_platt(offset_anchor)
            missing_bits = []
            if needs_temp and (onset_platt_temp is None or offset_platt_temp is None):
                missing_bits.append("temperature")
            if needs_bias and (onset_platt_bias is None or offset_platt_bias is None):
                missing_bits.append("bias")
            if missing_bits:
                raise SelectionError(
                    "Stage-B requires Platt {} from thorough calibration (anchor sources onset={} offset={}); rerun thorough to refresh stageB cache".format(
                        "/".join(missing_bits),
                        onset_anchor.source,
                        offset_anchor.source,
                    )
                )
            platt_overrides = {
                "onset": (
                    onset_platt_temp if needs_temp else None,
                    onset_platt_bias if needs_bias else None,
                ),
                "offset": (
                    offset_platt_temp if needs_temp else None,
                    offset_platt_bias if needs_bias else None,
                ),
            }

        LOGGER.info(
            "[autopilot:two-stage] anchors onset=%.4f offset=%.4f (source=%s)",
            onset_prob,
            offset_prob,
            anchor_source,
            extra=QUIET_EXTRA,
        )

        prev_onset_decoder = {}
        if isinstance(decoder_snapshot, Mapping):
            prev_onset_decoder = decoder_snapshot.get("onset", {}) or {}
        prev_open_cfg = _coerce_optional_float(prev_onset_decoder.get("open"))
        prev_hold_cfg = _coerce_optional_float(prev_onset_decoder.get("hold"))
        prev_min_on_cfg = _coerce_positive_int(prev_onset_decoder.get("min_on")) or 2

        merge_gap = int(args.onset_merge_gap)
        hold_mode = (args.onset_hold_mode or DEFAULT_ONSET_HOLD_MODE).lower()
        hold_delta = float(args.onset_hold_delta)
        hold_min = float(args.onset_hold_min)
        hold_ratio = float(args.onset_hold_ratio)

        def _compute_hold(open_val: float) -> float:
            if hold_mode == "config" and prev_hold_cfg is not None:
                hold = prev_hold_cfg
            elif hold_mode == "fixed_ratio":
                ratio_hold = open_val * hold_ratio
                hold = max(hold_min, min(open_val, ratio_hold))
            else:
                hold = open_val - hold_delta
                if hold_min > 0.0:
                    hold = max(hold, hold_min)
            return max(0.0, min(open_val, hold))

        def _is_prev_candidate(open_val: float, min_on_val: int) -> bool:
            if prev_open_cfg is None:
                return False
            if abs(open_val - prev_open_cfg) > 1e-4:
                return False
            return min_on_val == prev_min_on_cfg

        def _stage_log_prefix() -> str:
            base = Path(log_name).stem
            return base or "eval_fast"

        def _has_onset_gates(payload: Mapping[str, Any] | None) -> bool:
            if not isinstance(payload, Mapping):
                return False
            onset = payload.get("onset")
            if not isinstance(onset, Mapping):
                return False
            open_val = _coerce_optional_float(onset.get("open"))
            hold_val = _coerce_optional_float(onset.get("hold"))
            min_on_val = _coerce_positive_int(onset.get("min_on"))
            merge_gap_val = _coerce_positive_int(onset.get("merge_gap"))
            return (
                open_val is not None
                and hold_val is not None
                and min_on_val is not None
                and merge_gap_val is not None
            )

        stage_b_state_path = stdout_dir / STAGEB_STATE_TEMPLATE.format(round=round_index, calib=calibration_count)
        stage_b_artifacts_exist = stage_b_state_path.exists()
        round_state = _load_round_state(stdout_dir, round_index)
        if round_state.get("round") not in (None, round_index):
            round_state = {}
        stage_a_state_payload = round_state.get("stageA") if isinstance(round_state, Mapping) else None
        stage_a_state_valid = _has_onset_gates(stage_a_state_payload)
        stage_a_state_source = ""
        stage_a_tie_note = ""
        stage_a_reused = False

        stage_a_records: List[Dict[str, Any]] = []
        stage_a_winner: Optional[Dict[str, Any]] = None
        total_candidates = max(1, len(args.onset_open_grid) * len(args.onset_min_on_grid))
        frames = args.calib_frames if args.calib_frames is not None else 96
        max_clips = args.calib_max_clips if args.calib_max_clips is not None else 80
        if stage_a_state_valid:
            onset_state = stage_a_state_payload.get("onset") if isinstance(stage_a_state_payload, Mapping) else {}
            open_val = _coerce_optional_float(onset_state.get("open")) if isinstance(onset_state, Mapping) else None
            hold_val = _coerce_optional_float(onset_state.get("hold")) if isinstance(onset_state, Mapping) else None
            min_on_val = _coerce_positive_int(onset_state.get("min_on")) if isinstance(onset_state, Mapping) else None
            merge_gap_val = _coerce_positive_int(onset_state.get("merge_gap")) if isinstance(onset_state, Mapping) else None
            stage_a_state_source = stage_a_state_payload.get("source", "state") if isinstance(stage_a_state_payload, Mapping) else "state"
            stage_a_tie_note = stage_a_state_payload.get("tie_break", "") if isinstance(stage_a_state_payload, Mapping) else ""
            stage_a_reused = True
            print(
                "[Stage-A] reuse onset gates open={:.3f} hold={:.3f} min_on={} merge_gap={} (source={})".format(
                    float(open_val) if open_val is not None else prev_open_cfg or 0.0,
                    float(hold_val) if hold_val is not None else (prev_hold_cfg or 0.0),
                    int(min_on_val) if min_on_val is not None else prev_min_on_cfg,
                    int(merge_gap_val) if merge_gap_val is not None else merge_gap,
                    stage_a_state_source,
                )
            )
        else:
            if args.mode == "resume" and stage_b_artifacts_exist:
                fallback_open = _coerce_optional_float(prev_onset_decoder.get("open"))
                fallback_hold = _coerce_optional_float(prev_onset_decoder.get("hold"))
                fallback_min_on = _coerce_positive_int(prev_onset_decoder.get("min_on"))
                if fallback_open is not None and fallback_min_on is not None:
                    if fallback_hold is None:
                        fallback_hold = _compute_hold(float(fallback_open))
                    fallback_hold = max(0.0, min(float(fallback_open), float(fallback_hold)))
                    timestamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
                    stage_a_state_payload = {
                        "onset": {
                            "open": float(fallback_open),
                            "hold": float(fallback_hold),
                            "min_on": int(fallback_min_on),
                            "merge_gap": int(merge_gap),
                        },
                        "offset": _extract_offset_gates(decoder_snapshot),
                        "source": "previous_round",
                        "timestamp": timestamp,
                        "tie_break": "resume_previous",
                    }
                    round_state = dict(round_state) if isinstance(round_state, Mapping) else {}
                    round_state.update(
                        {
                            "round": round_index,
                            "phase": ROUND_PHASE_STAGEA_DONE,
                            "stageA": stage_a_state_payload,
                        }
                    )
                    _save_round_state(stdout_dir, round_index, round_state)
                    stage_a_state_valid = True
                    stage_a_reused = True
                    stage_a_state_source = "previous_round"
                    stage_a_tie_note = "resume_previous"
                    print(
                        "[Stage-A] resume fallback → using previous gates open={:.3f} hold={:.3f} min_on={} merge_gap={}".format(
                            fallback_open,
                            fallback_hold,
                            fallback_min_on,
                            merge_gap,
                        )
                    )
            if not stage_a_state_valid:
                candidate_idx = 0
                for open_val in args.onset_open_grid:
                    open_clamped = max(ONSET_OPEN_MIN, min(ONSET_OPEN_MAX, float(open_val)))
                    hold_val = _compute_hold(open_clamped)
                    hold_val = max(0.0, min(open_clamped, hold_val))
                    for min_on_val in args.onset_min_on_grid:
                        candidate_idx += 1
                        extras = [
                            "--decoder-onset-open",
                            f"{open_clamped:.4f}",
                            "--decoder-onset-hold",
                            f"{hold_val:.4f}",
                            "--decoder-onset-min-on",
                            str(int(min_on_val)),
                            "--decoder-onset-merge-gap",
                            str(merge_gap),
                        ]
                        log_file = f"{_stage_log_prefix()}_stageA_{candidate_idx:02d}.txt"
                        try:
                            request = _build_request(
                                onset_prob,
                                offset_prob,
                                log_name=log_file,
                                sweep_override=([onset_prob], [offset_prob]),
                                extras=extras,
                                low_guard=ONSET_THR_MIN,
                                platt_overrides=platt_overrides or None,
                            )
                            stage_result, stage_context, stage_lines = calibrate_and_score(request)
                            rows = parse_eval_rows(stage_lines)
                        except SelectionError as exc:
                            LOGGER.warning(
                                "Stage-A candidate failed (open=%.3f min_on=%d): %s",
                                open_clamped,
                                min_on_val,
                                exc,
                                extra=QUIET_EXTRA,
                            )
                            continue
                        if not rows:
                            LOGGER.warning(
                                "Stage-A candidate open=%.3f min_on=%d produced no rows",
                                open_clamped,
                                min_on_val,
                                extra=QUIET_EXTRA,
                            )
                            continue
                        row = rows[0]
                        onset_pred = float(row["onset_pred_rate"])
                        onset_pos = float(row["onset_pos_rate"])
                        guard_lo = ONSET_PRED_RATE_MIN_FACTOR * onset_pos
                        guard_hi = ONSET_PRED_RATE_MAX_FACTOR * onset_pos
                        valid = guard_lo <= onset_pred <= guard_hi
                        note = ""
                        if not valid:
                            note = (
                                f"pred_rate={onset_pred:.4f} outside "
                                f"[{guard_lo:.4f},{guard_hi:.4f}]"
                            )
                        status = "[OK]"
                        if not valid:
                            status = f"[REJECT:{note}]"
                        print(
                            "[Stage-A] {:2d}/{:2d} open={:.3f} hold={:.3f} min_on={} "
                            "onset_event_f1={:.4f} pred_rate={:.4f} pos_rate={:.4f} {}".format(
                                candidate_idx,
                                total_candidates,
                                open_clamped,
                                hold_val,
                                int(min_on_val),
                                float(row["onset_event_f1"]),
                                onset_pred,
                                onset_pos,
                                status,
                            )
                        )
                        stage_a_records.append(
                            {
                                "open": open_clamped,
                                "hold": hold_val,
                                "min_on": int(min_on_val),
                                "merge_gap": merge_gap,
                                "row": row,
                                "valid": valid,
                                "note": note,
                                "extras": extras,
                                "context": stage_context,
                                "tie_reason": "first",
                            }
                        )
                if not stage_a_records:
                    raise SelectionError("Stage-A search produced no candidates")

                valid_stage_a = [rec for rec in stage_a_records if rec["valid"]]
                stage_a_pool = valid_stage_a or stage_a_records
                stage_a_winner = stage_a_pool[0]
                stage_a_winner["tie_reason"] = stage_a_winner.get("tie_reason", "first")
                for rec in stage_a_pool[1:]:
                    cur_f1 = float(rec["row"]["onset_event_f1"])
                    best_f1 = float(stage_a_winner["row"]["onset_event_f1"])
                    if cur_f1 > best_f1 + ONSET_TIE_TOL:
                        rec["tie_reason"] = "higher_f1"
                        stage_a_winner = rec
                        continue
                    if abs(cur_f1 - best_f1) <= ONSET_TIE_TOL:
                        rec_prev = _is_prev_candidate(rec["open"], rec["min_on"])
                        best_prev = _is_prev_candidate(stage_a_winner["open"], stage_a_winner["min_on"])
                        if rec_prev and not best_prev:
                            rec["tie_reason"] = "prefer_previous"
                            stage_a_winner = rec
                            continue
                        if rec_prev == best_prev:
                            if rec["open"] > stage_a_winner["open"] + 1e-6:
                                rec["tie_reason"] = "higher_open"
                                stage_a_winner = rec
                                continue
                            if (
                                abs(rec["open"] - stage_a_winner["open"]) <= 1e-6
                                and rec["min_on"] < stage_a_winner["min_on"]
                            ):
                                rec["tie_reason"] = "lower_min_on"
                                stage_a_winner = rec
                                continue
                print(
                    "[Stage-A] winner open={:.3f} hold={:.3f} min_on={} "
                    "onset_event_f1={:.4f}".format(
                        stage_a_winner["open"],
                        stage_a_winner["hold"],
                        stage_a_winner["min_on"],
                        float(stage_a_winner["row"]["onset_event_f1"]),
                    )
                )
                stage_a_state_source = "grid_search"
                stage_a_tie_note = stage_a_winner.get("tie_reason", "")
                timestamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
                stage_a_state_payload = {
                    "onset": {
                        "open": float(stage_a_winner["open"]),
                        "hold": float(stage_a_winner["hold"]),
                        "min_on": int(stage_a_winner["min_on"]),
                        "merge_gap": int(stage_a_winner["merge_gap"]),
                    },
                    "offset": _extract_offset_gates(decoder_snapshot),
                    "source": stage_a_state_source,
                    "timestamp": timestamp,
                    "tie_break": stage_a_tie_note,
                }
                round_state = dict(round_state) if isinstance(round_state, Mapping) else {}
                round_state.update(
                    {
                        "round": round_index,
                        "phase": ROUND_PHASE_STAGEA_DONE,
                        "stageA": stage_a_state_payload,
                    }
                )
                _save_round_state(stdout_dir, round_index, round_state)
                stage_a_state_valid = True

        if not stage_a_state_valid or not isinstance(stage_a_state_payload, Mapping):
            raise SelectionError("Stage-A state unavailable for Stage-B evaluation")
        stage_a_state_payload = dict(stage_a_state_payload)
        stage_a_onset_state = stage_a_state_payload.get("onset")
        if not isinstance(stage_a_onset_state, Mapping):
            raise SelectionError("Stage-A onset gates missing from state")
        stage_a_onset_state = dict(stage_a_onset_state)
        stage_a_state_payload["onset"] = stage_a_onset_state
        onset_open_val = _coerce_optional_float(stage_a_onset_state.get("open"))
        onset_hold_val = _coerce_optional_float(stage_a_onset_state.get("hold"))
        onset_min_on_val = _coerce_positive_int(stage_a_onset_state.get("min_on"))
        onset_merge_gap_val = _coerce_positive_int(stage_a_onset_state.get("merge_gap"))
        if onset_open_val is None or onset_hold_val is None or onset_min_on_val is None or onset_merge_gap_val is None:
            raise SelectionError("Stage-A onset gates incomplete")

        current_onset_open = float(onset_open_val)
        current_onset_hold = float(onset_hold_val)
        current_onset_min_on = int(onset_min_on_val)
        current_onset_merge_gap = int(onset_merge_gap_val)
        base_merge_gap = current_onset_merge_gap
        stage_a_state_source = stage_a_state_source or stage_a_state_payload.get("source", "state")
        stage_a_tie_note = stage_a_tie_note or stage_a_state_payload.get("tie_break", "")
        fallback_prev_attempted = stage_a_state_source in {"previous_round", "fallback_previous"}
        fallback_minigrid_attempted = False

        def _stage_b_has_no_events(rows: Sequence[Mapping[str, Any]]) -> bool:
            if not rows:
                return True
            for entry in rows:
                pred_rate = _coerce_optional_float(entry.get("onset_pred_rate"))
                if pred_rate is not None and pred_rate > 1e-4:
                    return False
            return True

        def _mark_stage_b_started(source_label: str) -> None:
            nonlocal round_state, stage_a_state_source
            stage_a_state_source = source_label
            timestamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
            stage_a_state_payload["timestamp"] = timestamp
            stage_a_state_payload["source"] = source_label
            stage_a_state_payload["tie_break"] = stage_a_tie_note
            stage_a_onset_state["open"] = current_onset_open
            stage_a_onset_state["hold"] = current_onset_hold
            stage_a_onset_state["min_on"] = current_onset_min_on
            stage_a_onset_state["merge_gap"] = current_onset_merge_gap
            stage_b_state_block = {
                "status": "running",
                "timestamp": timestamp,
                "gates": {
                    "onset": {
                        "open": current_onset_open,
                        "hold": current_onset_hold,
                        "min_on": current_onset_min_on,
                        "merge_gap": current_onset_merge_gap,
                    },
                    "offset": copy.deepcopy(stage_a_state_payload.get("offset") or {}),
                },
                "resume": "reused" if stage_a_reused else "fresh",
                "attempt": source_label,
                "decoder_impl": args.decoder_impl,
            }
            new_state = dict(round_state) if isinstance(round_state, Mapping) else {}
            new_state.update(
                {
                    "round": round_index,
                    "phase": ROUND_PHASE_STAGEB_STARTED,
                    "stageA": stage_a_state_payload,
                    "stageB": stage_b_state_block,
                }
            )
            _save_round_state(stdout_dir, round_index, new_state)
            round_state = new_state

        def _apply_previous_fallback() -> bool:
            nonlocal current_onset_open, current_onset_hold, current_onset_min_on, current_onset_merge_gap, stage_a_tie_note, stage_a_state_source
            fallback_open_val = _coerce_optional_float(prev_onset_decoder.get("open"))
            fallback_min_on_val = _coerce_positive_int(prev_onset_decoder.get("min_on"))
            if fallback_open_val is None or fallback_min_on_val is None:
                return False
            fallback_hold_val = _coerce_optional_float(prev_onset_decoder.get("hold"))
            if fallback_hold_val is None:
                fallback_hold_val = _compute_hold(float(fallback_open_val))
            if (
                abs(current_onset_open - float(fallback_open_val)) <= 1e-6
                and current_onset_min_on == int(fallback_min_on_val)
            ):
                return False
            current_onset_open = float(max(ONSET_OPEN_MIN, min(ONSET_OPEN_MAX, float(fallback_open_val))))
            current_onset_hold = float(max(0.0, min(current_onset_open, float(fallback_hold_val))))
            current_onset_min_on = int(fallback_min_on_val)
            current_onset_merge_gap = base_merge_gap
            stage_a_tie_note = "fallback_previous"
            stage_a_state_source = "fallback_previous"
            print(
                "[Stage-B] fallback: using previous-round gates open={:.3f} hold={:.3f} min_on={}".format(
                    current_onset_open,
                    current_onset_hold,
                    current_onset_min_on,
                )
            )
            return True

        def _apply_minigrid_fallback() -> bool:
            nonlocal current_onset_open, current_onset_hold, current_onset_min_on, current_onset_merge_gap, stage_a_tie_note, stage_a_state_source
            candidate_open = max(ONSET_OPEN_MIN, current_onset_open - 0.02)
            if abs(candidate_open - current_onset_open) <= 1e-6:
                candidate_open = max(ONSET_OPEN_MIN, current_onset_open - 0.01)
            if abs(candidate_open - current_onset_open) <= 1e-6:
                return False
            current_onset_open = float(candidate_open)
            current_onset_hold = float(max(0.0, min(current_onset_open, _compute_hold(current_onset_open))))
            min_on_candidates = args.onset_min_on_grid if args.onset_min_on_grid else [current_onset_min_on]
            current_onset_min_on = int(min(current_onset_min_on, min(min_on_candidates)))
            current_onset_merge_gap = base_merge_gap
            stage_a_tie_note = "fallback_minigrid"
            stage_a_state_source = "fallback_minigrid"
            print(
                "[Stage-B] fallback: widening Stage-A grid open={:.3f} hold={:.3f} min_on={}".format(
                    current_onset_open,
                    current_onset_hold,
                    current_onset_min_on,
                )
            )
            return True

        onset_candidates, onset_anchor_used, onset_guard_min, onset_guard_max = _generate_stageb_candidates(
            onset_anchor, onset_stageb_params
        )
        offset_candidates, offset_anchor_used, offset_guard_min, offset_guard_max = _generate_stageb_candidates(
            offset_anchor, offset_stageb_params
        )
        onset_anchor_raw = _anchor_raw_probability(onset_anchor)
        offset_anchor_raw = _anchor_raw_probability(offset_anchor)
        print(
            "[Stage-B] onset anchor_raw={} anchor_used={} source={} guard=[{:.3f},{:.3f}] candidates={} (n={})".format(
                _format_prob(onset_anchor_raw),
                _format_prob(onset_anchor_used),
                onset_anchor.source,
                onset_guard_min,
                onset_guard_max,
                _format_candidates(onset_candidates),
                len(onset_candidates),
            )
        )
        print(
            "[Stage-B] offset anchor_raw={} anchor_used={} source={} guard=[{:.3f},{:.3f}] candidates={} (n={})".format(
                _format_prob(offset_anchor_raw),
                _format_prob(offset_anchor_used),
                offset_anchor.source,
                offset_guard_min,
                offset_guard_max,
                _format_candidates(offset_candidates),
                len(offset_candidates),
            )
        )
        kept_anchor_onset = _contains_probability(onset_candidates, onset_anchor_used, tol=STAGEB_PROB_TOL)
        kept_anchor_offset = _contains_probability(offset_candidates, offset_anchor_used, tol=STAGEB_PROB_TOL)
        print(
            "stageB: anchor_raw_onset={} anchor_used_onset={} kept_anchor_onset={} "
            "anchor_raw_offset={} anchor_used_offset={} kept_anchor_offset={} "
            "list_onset={} list_offset={}".format(
                _format_prob(onset_anchor_raw),
                _format_prob(onset_anchor_used),
                "true" if kept_anchor_onset else "false",
                _format_prob(offset_anchor_raw),
                _format_prob(offset_anchor_used),
                "true" if kept_anchor_offset else "false",
                _format_candidates(onset_candidates),
                _format_candidates(offset_candidates),
            )
        )
        anchor_retained_flags = {"onset": kept_anchor_onset, "offset": kept_anchor_offset}

        effective_onset_temp = float(args.temperature) if args.temperature is not None else onset_platt_temp
        effective_offset_temp = float(args.temperature) if args.temperature is not None else offset_platt_temp
        effective_onset_bias = float(args.bias) if args.bias is not None else onset_platt_bias
        effective_offset_bias = float(args.bias) if args.bias is not None else offset_platt_bias

        stage_b_log = f"{_stage_log_prefix()}_stageB.txt"
        stage_b_context: Optional[SelectionContext] = None
        stage_b_lines: List[str] = []
        rows_b: List[Dict[str, Any]] = []
        attempt_counter = 0
        while True:
            attempt_counter += 1
            stage_b_extras = [
                "--decoder-onset-open",
                f"{current_onset_open:.4f}",
                "--decoder-onset-hold",
                f"{current_onset_hold:.4f}",
                "--decoder-onset-min-on",
                str(int(current_onset_min_on)),
                "--decoder-onset-merge-gap",
                str(int(current_onset_merge_gap)),
            ]
            onset_thr_display = "[" + ",".join(_format_prob(val) for val in onset_candidates) + "]"
            offset_thr_display = "[" + ",".join(_format_prob(val) for val in offset_candidates) + "]"
            print(
                "stageB: gates on(open={:.3f},hold={:.3f},min_on={},gap={}) | "
                "T_on={:.4f}, b_on={:.4f}, T_off={:.4f}, b_off={:.4f} | "
                "thr_onset={} thr_offset={}".format(
                    current_onset_open,
                    current_onset_hold,
                    current_onset_min_on,
                    current_onset_merge_gap,
                    float(effective_onset_temp) if effective_onset_temp is not None else 1.0,
                    float(effective_onset_bias) if effective_onset_bias is not None else 0.0,
                    float(effective_offset_temp) if effective_offset_temp is not None else 1.0,
                    float(effective_offset_bias) if effective_offset_bias is not None else 0.0,
                    onset_thr_display,
                    offset_thr_display,
                )
            )
            print(
                "[Stage-B] using Stage-A gates open={:.3f} hold={:.3f} min_on={} merge_gap={} (source={})".format(
                    current_onset_open,
                    current_onset_hold,
                    current_onset_min_on,
                    current_onset_merge_gap,
                    stage_a_state_source,
                )
            )
            _mark_stage_b_started(stage_a_state_source)
            try:
                request = _build_request(
                    onset_anchor_used,
                    offset_anchor_used,
                    log_name=stage_b_log,
                    sweep_override=(onset_candidates, offset_candidates),
                    extras=stage_b_extras,
                    low_guard=onset_stageb_params.low_guard or onset_stageb_params.min_prob,
                    platt_overrides=platt_overrides or None,
                )
                _, stage_b_context, stage_b_lines = calibrate_and_score(request)
                rows_b = parse_eval_rows(stage_b_lines)
            except SelectionError as exc:
                raise SelectionError(f"Stage-B evaluation failed: {exc}") from exc
            if rows_b and not _stage_b_has_no_events(rows_b):
                break
            reason = "empty rows" if not rows_b else "no-onset-events"
            print(f"[Stage-B] WARNING: sweep returned {reason}; evaluating fallbacks")
            if not fallback_prev_attempted and _apply_previous_fallback():
                fallback_prev_attempted = True
                continue
            fallback_prev_attempted = True
            if not fallback_minigrid_attempted and _apply_minigrid_fallback():
                fallback_minigrid_attempted = True
                continue
            fallback_minigrid_attempted = True
            raise SelectionError("Stage-B evaluation returned no usable rows after fallbacks")

        onset_open = current_onset_open
        onset_hold = current_onset_hold
        onset_min_on = current_onset_min_on
        onset_merge_gap = current_onset_merge_gap

        stage_b_candidates: List[Dict[str, Any]] = []
        for row in rows_b:
            thr_val = float(row["onset_thr"])
            pred_rate = float(row["onset_pred_rate"])
            pos_rate = float(row["onset_pos_rate"])
            guard_lo = ONSET_PRED_RATE_MIN_FACTOR * pos_rate
            guard_hi = ONSET_PRED_RATE_MAX_FACTOR * pos_rate
            valid = guard_lo <= pred_rate <= guard_hi
            note = ""
            if not valid:
                note = (
                    f"pred_rate={pred_rate:.4f} outside "
                    f"[{guard_lo:.4f},{guard_hi:.4f}]"
                )
            status = "[OK]"
            if not valid:
                status = f"[REJECT:{note}]"
            print(
                "[Stage-B] thr={:.4f} onset_event_f1={:.4f} pred_rate={:.4f} pos_rate={:.4f} {}".format(
                    thr_val,
                    float(row["onset_event_f1"]),
                    pred_rate,
                    pos_rate,
                    status,
                )
            )
            stage_b_candidates.append(
                {
                    "row": row,
                    "valid": valid,
                    "note": note,
                }
            )

        valid_stage_b = [cand for cand in stage_b_candidates if cand["valid"]]
        stage_b_pool = valid_stage_b or stage_b_candidates
        if not stage_b_pool:
            raise SelectionError("Stage-B evaluation produced no candidates")
        stage_b_winner = stage_b_pool[0]
        tie_break_note = ""
        tie_break_reason = ""
        target_is_onset_metric = target_metric_field == "onset_event_f1"
        if target_is_onset_metric:
            log_tie_break_eps()
            ctx = OnsetTieBreakContext(anchor_used=onset_anchor_used, prev_threshold=prev_onset_best)
            ranked_rows = [cand["row"] for cand in stage_b_pool]
            best_row_obj, tie_break_reason = select_best_onset_row(ranked_rows, ctx)
            for cand in stage_b_pool:
                if cand["row"] is best_row_obj:
                    stage_b_winner = cand
                    break
            if not tie_break_reason:
                tie_break_reason = "metric"
        else:
            for cand in stage_b_pool[1:]:
                cur_f1 = float(cand["row"]["onset_event_f1"])
                best_f1 = float(stage_b_winner["row"]["onset_event_f1"])
                if cur_f1 > best_f1 + ONSET_TIE_TOL:
                    stage_b_winner = cand
                    tie_break_note = ""
                    continue
                if abs(cur_f1 - best_f1) <= ONSET_TIE_TOL:
                    prev_threshold = prev_onset_best
                    cand_thr = float(cand["row"]["onset_thr"])
                    best_thr = float(stage_b_winner["row"]["onset_thr"])
                    cand_prev = prev_threshold is not None and abs(cand_thr - prev_threshold) <= 1e-5
                    best_prev = prev_threshold is not None and abs(best_thr - prev_threshold) <= 1e-5
                    if cand_prev and not best_prev:
                        stage_b_winner = cand
                        tie_break_note = "tie→prev_threshold"
                        continue
                    if cand_prev == best_prev:
                        if cand_thr > best_thr + 1e-6:
                            stage_b_winner = cand
                            tie_break_note = "tie→higher_threshold"
                            continue
                        if abs(cand_thr - best_thr) <= 1e-6:
                            cand_diff = abs(float(cand["row"]["onset_pred_rate"]) - float(cand["row"]["onset_pos_rate"]))
                            best_diff = abs(float(stage_b_winner["row"]["onset_pred_rate"]) - float(stage_b_winner["row"]["onset_pos_rate"]))
                            if cand_diff < best_diff - 1e-6:
                                stage_b_winner = cand
                                tie_break_note = "tie→pred_rate_balance"
                                continue
        if not tie_break_note and not stage_b_winner["valid"]:
            tie_break_note = "guard-fallback"
        if target_is_onset_metric:
            tie_break_reason = tie_break_reason or "metric"
        else:
            tie_break_reason = tie_break_note or ""
        best_row = stage_b_winner["row"]
        if target_is_onset_metric:
            anchor_label = "--" if onset_anchor_used is None else f"{float(onset_anchor_used):.4f}"
            prev_label = "--" if prev_onset_best is None else f"{float(prev_onset_best):.4f}"
            print(
                "target={target} | best_onset_f1={f1:.4f} | tie_break={reason} | "
                "pred={pred:.4f} pos={pos:.4f} | thr={thr:.4f} | anchor_used={anchor} | prev_thr={prev}".format(
                    target=target_metric_field,
                    f1=float(best_row["onset_event_f1"]),
                    reason=tie_break_reason,
                    pred=float(best_row["onset_pred_rate"]),
                    pos=float(best_row["onset_pos_rate"]),
                    thr=float(best_row["onset_thr"]),
                    anchor=anchor_label,
                    prev=prev_label,
                )
            )
        print(
            "[Stage-B] winner thr={:.4f} onset_event_f1={:.4f} pred_rate={:.4f} pos_rate={:.4f} open={:.4f} hold={:.4f} min_on={} {}".format(
                float(best_row["onset_thr"]),
                float(best_row["onset_event_f1"]),
                float(best_row["onset_pred_rate"]),
                float(best_row["onset_pos_rate"]),
                float(onset_open),
                float(onset_hold),
                int(onset_min_on),
                tie_break_note or tie_break_reason,
            )
        )
        summary_tie = tie_break_note or tie_break_reason or "none"
        print(
            "stageB: gates(open={:.3f}, hold={:.3f}, min_on={}, gap={}) -> best onset_thr={:.4f} onset_event_f1={:.4f} (tie: {})".format(
                float(onset_open),
                float(onset_hold),
                int(onset_min_on),
                int(onset_merge_gap),
                float(best_row["onset_thr"]),
                float(best_row["onset_event_f1"]),
                summary_tie,
            )
        )

        onset_anchor.details["sweep_center"] = onset_anchor_used
        onset_anchor.details["guard"] = [onset_guard_min, onset_guard_max]
        onset_anchor.details["candidate_count"] = len(onset_candidates)
        offset_anchor.details["sweep_center"] = offset_anchor_used
        offset_anchor.details["guard"] = [offset_guard_min, offset_guard_max]
        offset_anchor.details["candidate_count"] = len(offset_candidates)
        stage_b_gates_payload = {
            "onset": {
                "open": float(onset_open),
                "hold": float(onset_hold),
                "min_on": int(onset_min_on),
                "merge_gap": int(onset_merge_gap),
            },
            "offset": copy.deepcopy(stage_a_state_payload.get("offset") or {}),
        }
        _write_stageb_artifacts(
            stdout_dir,
            round_idx=round_index,
            calib_index=calibration_count,
            anchors={"onset": onset_anchor, "offset": offset_anchor},
            candidates={"onset": onset_candidates, "offset": offset_candidates},
            tie_break=tie_break_note,
            winner_row=best_row,
            gates=stage_b_gates_payload,
            anchor_retained=anchor_retained_flags,
        )

        stage_b_summary = {
            "status": "completed",
            "timestamp": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "decoder_impl": args.decoder_impl,
            "platt": {
                "onset": {
                    "temperature": _coerce_optional_float(effective_onset_temp),
                    "bias": _coerce_optional_float(effective_onset_bias),
                },
                "offset": {
                    "temperature": _coerce_optional_float(effective_offset_temp),
                    "bias": _coerce_optional_float(effective_offset_bias),
                },
            },
            "winner": {
                "onset_thr": _coerce_optional_float(best_row.get("onset_thr")),
                "offset_thr": _coerce_optional_float(best_row.get("offset_thr")),
                "onset_event_f1": _coerce_optional_float(best_row.get("onset_event_f1")),
                "offset_event_f1": _coerce_optional_float(best_row.get("offset_event_f1")),
                "ev_f1_mean": _coerce_optional_float(best_row.get("ev_f1_mean")),
                "tie_break": tie_break_note or tie_break_reason,
                "tie_break_reason": tie_break_reason or tie_break_note,
                "anchor_used": _coerce_optional_float(onset_anchor_used),
                "prev_onset_thr": _coerce_optional_float(prev_onset_best),
                "onset_pred_rate": _coerce_optional_float(best_row.get("onset_pred_rate")),
                "onset_pos_rate": _coerce_optional_float(best_row.get("onset_pos_rate")),
            },
            "metrics": {
                "onset_pred_rate": _coerce_optional_float(best_row.get("onset_pred_rate")),
                "onset_pos_rate": _coerce_optional_float(best_row.get("onset_pos_rate")),
            },
            "gates": {
                "onset": copy.deepcopy(stage_a_onset_state),
                "offset": copy.deepcopy(stage_a_state_payload.get("offset") or {}),
            },
        }
        round_state = dict(round_state) if isinstance(round_state, Mapping) else {}
        round_state.update(
            {
                "round": round_index,
                "phase": ROUND_PHASE_STAGEB_DONE,
                "stageA": stage_a_state_payload,
                "stageB": stage_b_summary,
            }
        )
        _save_round_state(stdout_dir, round_index, round_state)
        summary_payload = {
            "round": round_index,
            "calibration_index": calibration_count,
            "decoder_impl": args.decoder_impl,
            "stageA": copy.deepcopy(stage_a_state_payload),
            "stageB": copy.deepcopy(stage_b_summary),
        }
        summary_path = stdout_dir / f"round{round_index:02d}_summary.json"
        _atomic_write_json(summary_path, summary_payload)

        decoder_payload = {
            key: best_row[key]
            for key in best_row.keys()
            if key.startswith("decoder_")
        }
        decoder_payload.update(
            {
                "decoder_onset_open": float(onset_open),
                "decoder_onset_hold": float(onset_hold),
                "decoder_onset_min_on": int(onset_min_on),
                "decoder_onset_merge_gap": int(onset_merge_gap),
            }
        )
        if tie_break_reason == "metric":
            winner_reason_for_payload = ""
        else:
            winner_reason_for_payload = tie_break_reason
        winner_tie_label = tie_break_note or winner_reason_for_payload
        if winner_tie_label:
            decoder_payload["tie_break_note"] = winner_tie_label

        selection = SelectionResult(
            onset_threshold=float(best_row["onset_thr"]),
            offset_threshold=float(best_row["offset_thr"]),
            k_onset=int(best_row.get("k_onset", 1)),
            onset_event_f1=float(best_row["onset_event_f1"]),
            offset_event_f1=float(best_row["offset_event_f1"]),
            mean_event_f1=float(best_row["ev_f1_mean"]),
            onset_f1=float(best_row["onset_f1"]),
            offset_f1=float(best_row["offset_f1"]),
            onset_pred_rate=float(best_row["onset_pred_rate"]),
            onset_pos_rate=float(best_row["onset_pos_rate"]),
            decoder_kind=best_row.get("decoder_kind"),
            decoder_settings=decoder_payload,
        )
        _write_calibration_json(selection)
        return selection, stage_b_context, stage_b_lines

    def _run_selection_with_calibration(calib: dict, *, log_name: str) -> Tuple[SelectionResult, SelectionContext, List[str]]:
        if args.fast_strategy == "two_stage":
            banner = "NOTICE: Running TWO-STAGE fast calibration — this may take a while"
            log_banner(results_path, banner)
            return _run_two_stage(calib, log_name=log_name)
        banner = "NOTICE: Running FAST calibration — this may take a while"
        log_banner(results_path, banner)
        log_fallback = _parse_thorough_log(stdout_dir)
        needs_temp = args.temperature is None
        needs_bias = args.bias is None
        platt_overrides_calib: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
        if needs_temp or needs_bias:
            platt_overrides_calib = {}
            missing_heads: List[str] = []
            for head in ("onset", "offset"):
                head_entry = calib.get(head) if isinstance(calib, Mapping) else None
                temp_val, bias_val = _extract_partial_platt(head_entry)
                if (needs_temp and temp_val is None) or (needs_bias and bias_val is None):
                    missing_heads.append(head)
                platt_overrides_calib[head] = (
                    temp_val if needs_temp else None,
                    bias_val if needs_bias else None,
                )
            if missing_heads:
                raise SelectionError(
                    "Calibration missing Platt parameters for heads: {}".format(", ".join(missing_heads))
                )
        onset_anchor = _resolve_stageb_anchor(
            "onset",
            params=onset_stageb_params,
            thorough_cache=thorough_cache,
            log_fallback=log_fallback,
            previous_best=prev_onset_best,
            config_anchor=args.onset_thr_anchor,
        )
        offset_anchor = _resolve_stageb_anchor(
            "offset",
            params=offset_stageb_params,
            thorough_cache=thorough_cache,
            log_fallback=log_fallback,
            previous_best=prev_offset_best,
            config_anchor=offset_thr_anchor,
        )
        onset_prob = _clamp_fast_result(onset_anchor.prob)
        offset_prob = _clamp_fast_result(offset_anchor.prob)
        if onset_anchor.source == offset_anchor.source:
            anchor_source = onset_anchor.source
        else:
            anchor_source = f"onset:{onset_anchor.source},offset:{offset_anchor.source}"
        LOGGER.info(
            "[autopilot:grid] anchors onset=%.4f offset=%.4f (source=%s)",
            onset_prob,
            offset_prob,
            anchor_source,
            extra=QUIET_EXTRA,
        )
        request = _build_request(
            onset_prob,
            offset_prob,
            log_name=log_name,
            platt_overrides=platt_overrides_calib or None,
        )
        result, context, lines = calibrate_and_score(request)
        _write_calibration_json(result)
        return result, context, lines

    def _run_fast_grid(reason: Optional[str] = None) -> Tuple[Optional[SelectionResult], Optional[SelectionContext], List[str], int]:
        if reason:
            print(f"[autopilot] WARNING: {reason} → falling back to fast grid calibration")
        banner = "NOTICE: Running FAST calibration — this may take a while"
        log_banner(results_path, banner)
        needs_temp = args.temperature is None
        needs_bias = args.bias is None
        platt_overrides_grid: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
        if needs_temp or needs_bias:
            calib_stub = load_calibration(CALIB_JSON)
            if calib_stub:
                platt_overrides_grid = {}
                missing_heads: List[str] = []
                for head in ("onset", "offset"):
                    head_entry = calib_stub.get(head) if isinstance(calib_stub, Mapping) else None
                    temp_val, bias_val = _extract_partial_platt(head_entry)
                    if (needs_temp and temp_val is None) or (needs_bias and bias_val is None):
                        missing_heads.append(head)
                    platt_overrides_grid[head] = (
                        temp_val if needs_temp else None,
                        bias_val if needs_bias else None,
                    )
                if missing_heads:
                    raise SelectionError(
                        "Fast grid requires Platt parameters for heads: {}".format(", ".join(missing_heads))
                    )
        try:
            request = _build_request(
                0.5,
                0.5,
                log_name="calibration_fast_grid.txt",
                sweep_override=(FAST_GRID_THRESHOLDS, FAST_GRID_THRESHOLDS),
                extras=["--grid_prob_thresholds"],
                low_guard=0.02,
                platt_overrides=platt_overrides_grid or None,
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
            use_legacy_decoder=legacy_decoder,
        )
        if ret != 0:
            return _run_fast_grid("thorough calibration failed")
        calib = load_calibration(CALIB_JSON)
        if calib is None:
            return _run_fast_grid("calibration.json missing after thorough calibration")
        thorough_cache = _update_stageb_cache_from_calib(stdout_dir, calib)
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
    "pitch_pos_f1",
    "pitch_frame_exact_acc",
    "onset_ev_f1",
    "offset_ev_f1",
    "ev_f1_mean",
    "decoder_kind",
    "decoder_impl",
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
        pitch_pos_val = metrics.get("pitch_pos_f1")
        if isinstance(pitch_pos_val, (int, float)):
            bits.append(f"pitch_pos_f1={pitch_pos_val:.3f}")
        pitch_exact_val = metrics.get("pitch_frame_exact_acc")
        if isinstance(pitch_exact_val, (int, float)):
            bits.append(f"pitch_frame_exact_acc={pitch_exact_val:.3f}")
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
        tie_note = metrics.get("tie_break_note")
        if tie_note:
            bits.append(f"tie_break={tie_note}")
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
        _fmt_float(metrics.get("pitch_pos_f1"), ".4f", "0.0000"),
        _fmt_float(metrics.get("pitch_frame_exact_acc"), ".4f", "0.0000"),
        _fmt_float(metrics.get("onset_event_f1"), ".4f", "0.0000"),
        _fmt_float(metrics.get("offset_event_f1"), ".4f", "0.0000"),
        _fmt_float(metrics.get("ev_f1_mean"), ".4f", "0.0000"),
        str(metrics.get("decoder_kind") or ""),
        str(metrics.get("decoder_impl") or ""),
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
        help=(
            "Extra CLI arguments appended to eval_thresholds.py during fast evaluation "
            '(tokens without a leading "-" are auto-prefixed, e.g. "no_eval_cache")'
        ),
    )
    ap.add_argument(
        "--postproc-mode",
        choices=["never", "eval_only", "always"],
        default="eval_only",
        help="Forward to eval_thresholds.py --postproc-mode (default: eval_only)",
    )
    ap.add_argument(
        "--model-return-per-tile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forward to eval_thresholds.py --model-return-per-tile for per-tile logits",
    )
    ap.add_argument(
        "--decoder-impl",
        choices=["new", "legacy"],
        default="new",
        help="Select decoder/calibration implementation; 'legacy' toggles rollback flags",
    )
    ap.add_argument(
        "--key-prior-mode",
        choices=["config", "enabled", "disabled"],
        default="config",
        help="Override decoder.key_prior.enabled before running child scripts (default: respect config.yaml)",
    )
    ap.add_argument(
        "--fast_strategy",
        choices=["classic", "two_stage"],
        default=DEFAULT_FAST_STRATEGY,
        help="Fast calibration strategy after training bursts",
    )
    ap.add_argument(
        "--onset_open_grid",
        type=float,
        nargs="+",
        help="Candidate onset decoder open gates for Stage-A search",
    )
    ap.add_argument(
        "--onset_min_on_grid",
        type=int,
        nargs="+",
        help="Candidate onset decoder min_on values for Stage-A search",
    )
    ap.add_argument(
        "--onset_hold_mode",
        choices=["fixed_delta", "config", "fixed_ratio"],
        help="Hold calculation strategy for Stage-A search (default: fixed_ratio → open * hold_ratio)",
    )
    ap.add_argument(
        "--onset_hold_delta",
        type=float,
        help="Hold subtraction applied when hold_mode=fixed_delta (default: 0.06)",
    )
    ap.add_argument(
        "--onset_hold_min",
        type=float,
        help="Minimum hold gate enforced when hold_mode=fixed_delta/fixed_ratio (default: 0.015)",
    )
    ap.add_argument(
        "--onset_hold_ratio",
        type=float,
        help="Hold ratio applied when hold_mode=fixed_ratio (default: 0.75)",
    )
    ap.add_argument(
        "--onset_merge_gap",
        type=int,
        help="Merge gap applied during Stage-A decoder search (default: 1)",
    )
    ap.add_argument(
        "--onset_thr_anchor",
        type=float,
        help="Anchor onset probability threshold for Stage-B micro-sweep (default: previous best or 0.20)",
    )
    ap.add_argument(
        "--onset_thr_delta",
        type=float,
        help="Radius around the anchor for Stage-B threshold micro-sweep (default: 0.05)",
    )
    ap.add_argument(
        "--onset_thr_steps",
        type=int,
        help="Number of points in the Stage-B micro-sweep (default: 5, must include anchor)",
    )
    args = ap.parse_args()
    args.verbose = configure_verbosity(args.verbose)
    try:
        raw_eval_extras = shlex.split(args.eval_extras) if args.eval_extras else []
    except ValueError as exc:
        print(f"error: could not parse --eval_extras: {exc}", file=sys.stderr)
        return 1
    args.eval_extras_tokens = _normalize_eval_extras(raw_eval_extras)
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

    decoder_root = cfg.setdefault("decoder", {})
    key_prior_cfg: Dict[str, Any] = decoder_root.setdefault("key_prior", {})
    if "enabled" not in key_prior_cfg:
        key_prior_cfg["enabled"] = False
        changed = True
    if args.key_prior_mode != "config":
        desired = args.key_prior_mode == "enabled"
        if bool(key_prior_cfg.get("enabled")) != desired:
            key_prior_cfg["enabled"] = desired
            changed = True
            state = "enabled" if desired else "disabled"
            print(f"[autopilot] key prior override → {state}")

    autop_cfg_root = cfg.setdefault("autopilot", {})
    onset_opt_cfg = autop_cfg_root.setdefault("onset_optimizer", {})

    def _normalize_int_list(values: Iterable[int], *, allowed: Optional[Iterable[int]] = None, min_value: int = 0) -> List[int]:
        allowed_set = set(int(v) for v in allowed) if allowed is not None else None
        seen: Dict[int, int] = {}
        for raw in values:
            try:
                val = int(raw)
            except (TypeError, ValueError):
                continue
            if allowed_set is not None and val not in allowed_set:
                continue
            if val < min_value:
                continue
            seen[val] = val
        ordered = sorted(seen.values())
        return ordered

    def _assign_setting(
        arg_value,
        *,
        cfg_key: str,
        default,
        transform,
        attr_name: str,
    ) -> None:
        nonlocal changed
        if arg_value is not None:
            coerced = transform(arg_value)
            onset_opt_cfg[cfg_key] = coerced
            setattr(args, attr_name, coerced)
            changed = True
            return
        if cfg_key in onset_opt_cfg:
            coerced = transform(onset_opt_cfg[cfg_key])
        else:
            coerced = transform(default)
            onset_opt_cfg[cfg_key] = coerced
            changed = True
        setattr(args, attr_name, coerced)

    _assign_setting(
        args.onset_open_grid,
        cfg_key="open_grid",
        default=DEFAULT_ONSET_OPEN_GRID,
        transform=lambda vals: _normalize_probability_list(vals, lo=ONSET_OPEN_MIN, hi=ONSET_OPEN_MAX),
        attr_name="onset_open_grid",
    )
    if not args.onset_open_grid:
        args.onset_open_grid = _normalize_probability_list(DEFAULT_ONSET_OPEN_GRID, lo=ONSET_OPEN_MIN, hi=ONSET_OPEN_MAX)
    _assign_setting(
        args.onset_min_on_grid,
        cfg_key="min_on_grid",
        default=DEFAULT_ONSET_MIN_ON_GRID,
        transform=lambda vals: _normalize_int_list(vals, allowed=DEFAULT_ONSET_MIN_ON_GRID, min_value=1),
        attr_name="onset_min_on_grid",
    )
    if not args.onset_min_on_grid:
        args.onset_min_on_grid = list(DEFAULT_ONSET_MIN_ON_GRID)
    _assign_setting(
        args.onset_hold_mode,
        cfg_key="hold_mode",
        default=DEFAULT_ONSET_HOLD_MODE,
        transform=lambda val: str(val),
        attr_name="onset_hold_mode",
    )
    _assign_setting(
        args.onset_hold_delta,
        cfg_key="hold_delta",
        default=DEFAULT_ONSET_HOLD_DELTA,
        transform=lambda val: max(0.0, float(val)),
        attr_name="onset_hold_delta",
    )
    _assign_setting(
        args.onset_hold_min,
        cfg_key="hold_min",
        default=DEFAULT_ONSET_HOLD_MIN,
        transform=lambda val: max(0.0, float(val)),
        attr_name="onset_hold_min",
    )
    _assign_setting(
        args.onset_hold_ratio,
        cfg_key="hold_ratio",
        default=DEFAULT_ONSET_HOLD_RATIO,
        transform=lambda val: max(0.0, min(1.0, float(val))),
        attr_name="onset_hold_ratio",
    )
    _assign_setting(
        args.onset_merge_gap,
        cfg_key="merge_gap",
        default=DEFAULT_ONSET_MERGE_GAP,
        transform=lambda val: int(max(0, int(val))),
        attr_name="onset_merge_gap",
    )
    _assign_setting(
        args.onset_thr_anchor,
        cfg_key="thr_anchor",
        default=DEFAULT_ONSET_THR_ANCHOR,
        transform=lambda val: float(max(ONSET_THR_MIN, min(ONSET_THR_MAX, float(val)))),
        attr_name="onset_thr_anchor",
    )
    _assign_setting(
        args.onset_thr_delta,
        cfg_key="thr_add_delta",
        default=DEFAULT_ONSET_THR_DELTA,
        transform=lambda val: max(0.0, float(val)),
        attr_name="onset_thr_delta",
    )
    _assign_setting(
        args.onset_thr_steps,
        cfg_key="thr_add_steps",
        default=DEFAULT_ONSET_THR_STEPS,
        transform=lambda val: max(3, int(val)),
        attr_name="onset_thr_steps",
    )
    offset_opt_cfg = autop_cfg_root.setdefault("offset_optimizer", {})

    def _ensure_stageb_defaults(cfg_dict: dict, defaults: Mapping[str, Any]) -> None:
        nonlocal changed
        for key, value in defaults.items():
            if key not in cfg_dict:
                cfg_dict[key] = value
                changed = True

    _ensure_stageb_defaults(
        onset_opt_cfg,
        {
            "thr_mul_ratio": DEFAULT_STAGEB_MUL_RATIO,
            "thr_mul_orders": DEFAULT_STAGEB_MUL_ORDERS,
            "thr_min_prob": DEFAULT_STAGEB_MIN_PROB,
            "thr_max_prob": DEFAULT_STAGEB_MAX_PROB,
            "thr_low_guard": DEFAULT_ONSET_STAGEB_LOW_GUARD,
            "thr_high_guard": DEFAULT_ONSET_STAGEB_HIGH_GUARD,
            "thr_min_points": DEFAULT_STAGEB_MIN_POINTS,
            "thr_max_points": DEFAULT_STAGEB_MAX_POINTS,
        },
    )
    _ensure_stageb_defaults(
        offset_opt_cfg,
        {
            "thr_anchor": DEFAULT_OFFSET_THR_ANCHOR,
            "thr_add_delta": DEFAULT_STAGEB_ADD_DELTA,
            "thr_add_steps": DEFAULT_STAGEB_ADD_STEPS,
            "thr_mul_ratio": DEFAULT_STAGEB_MUL_RATIO,
            "thr_mul_orders": DEFAULT_STAGEB_MUL_ORDERS,
            "thr_min_prob": DEFAULT_STAGEB_MIN_PROB,
            "thr_max_prob": DEFAULT_STAGEB_MAX_PROB,
            "thr_low_guard": DEFAULT_OFFSET_STAGEB_LOW_GUARD,
            "thr_high_guard": DEFAULT_OFFSET_STAGEB_HIGH_GUARD,
            "thr_min_points": DEFAULT_STAGEB_MIN_POINTS,
            "thr_max_points": DEFAULT_STAGEB_MAX_POINTS,
        },
    )
    resolved_strategy = args.fast_strategy or DEFAULT_FAST_STRATEGY
    if autop_cfg_root.get("fast_strategy") != resolved_strategy:
        autop_cfg_root["fast_strategy"] = resolved_strategy
        changed = True
    args.fast_strategy = resolved_strategy

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
                target_metric_field=target_metric_field,
                calibration_count=calibration_count,
                seed=seed,
                deterministic=deterministic,
                round_index=round_idx,
            )
            if selection_result is None:
                print("Calibration failed", file=sys.stderr)
                return eval_ret or 1
            calibration_count += 1
            metrics = _result_to_metrics(selection_result)
            metrics["decoder_impl"] = args.decoder_impl
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
                    "pitch_pos_f1": 0.0,
                    "pitch_frame_exact_acc": 0.0,
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
                "pitch_pos_f1": 0.0,
                "pitch_frame_exact_acc": 0.0,
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
                    target_metric_field=target_metric_field,
                    calibration_count=calibration_count,
                    seed=seed,
                    deterministic=deterministic,
                    round_index=round_idx,
                )
                if selection_result is None:
                    print("Calibration failed", file=sys.stderr)
                    return eval_ret or 1
                calibration_count += 1
                metrics = _result_to_metrics(selection_result)
                metrics["decoder_impl"] = args.decoder_impl
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
            tie_note = metrics.get("tie_break_note")
            tie_suffix = f" tie_break={tie_note}" if tie_note else ""
            print(
                f"[autopilot] New best {args.target_metric}={metric_value:.4f} "
                f"(round {round_idx}); ev_f1_mean={ev_mean:.4f} owner={best_owner}{tie_suffix}"
            )
        else:
            patience_left = patience_record
            tie_note = metrics.get("tie_break_note")
            tie_suffix = f" tie_break={tie_note}" if tie_note else ""
            print(
                f"[autopilot] {args.target_metric}={metric_value:.4f} "
                f"(best={best_metric:.4f}), ev_f1_mean={ev_mean:.4f} "
                f"patience_left={patience_left}{tie_suffix}"
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
