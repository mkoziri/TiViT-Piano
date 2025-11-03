from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


FAST_SWEEP_CLIP_MIN = 0.02
FAST_SWEEP_CLIP_MAX = 0.98
FAST_RESULT_MIN = 0.02
FAST_RESULT_MAX = 0.98

DEFAULT_DELTA = 0.05
DEFAULT_LOW_GUARD = 0.10

EVAL_SCRIPT = Path("scripts") / "calib" / "eval_thresholds.py"
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
FLOAT_PREFIX_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


class SelectionError(RuntimeError):
    """Raised when calibrated selection fails."""


def _clamp_range(prob: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(prob)))


def _clamp_fast_result(prob: float) -> float:
    return _clamp_range(prob, FAST_RESULT_MIN, FAST_RESULT_MAX)


def _bounded_candidates(
    center: float,
    delta: float,
    *,
    lower: float,
    upper: float,
    low_guard: float,
    extras: Sequence[float] | None = None,
) -> Tuple[float, ...]:
    raw = [center - delta, center, center + delta]
    if extras:
        raw.extend(float(x) for x in extras)
    seen = set()
    ordered: List[float] = []
    guard_lower = max(lower, min(low_guard, upper))
    for value in raw:
        clipped = _clamp_range(value, lower, upper)
        if clipped < guard_lower:
            clipped = guard_lower
        key = int(round(clipped * 1000))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(clipped)
    ordered.sort()
    return tuple(ordered)


@dataclass
class SweepSpec:
    onset_center: float
    offset_center: float
    delta: float = DEFAULT_DELTA
    clamp_min: float = FAST_SWEEP_CLIP_MIN
    clamp_max: float = FAST_SWEEP_CLIP_MAX
    low_guard: float = DEFAULT_LOW_GUARD
    onset_extras: Sequence[float] = field(default_factory=tuple)
    offset_extras: Sequence[float] = field(default_factory=tuple)
    onset_explicit: Optional[Sequence[float]] = None
    offset_explicit: Optional[Sequence[float]] = None
    k_onset_candidates: Sequence[int] = field(default_factory=lambda: (1,))

    def onset_candidates(self) -> Tuple[float, ...]:
        if self.onset_explicit is not None:
            return tuple(_clamp_fast_result(v) for v in self.onset_explicit)
        return _bounded_candidates(
            self.onset_center,
            self.delta,
            lower=self.clamp_min,
            upper=self.clamp_max,
            low_guard=self.low_guard,
            extras=self.onset_extras,
        )

    def offset_candidates(self) -> Tuple[float, ...]:
        if self.offset_explicit is not None:
            return tuple(_clamp_fast_result(v) for v in self.offset_explicit)
        return _bounded_candidates(
            self.offset_center,
            self.delta,
            lower=self.clamp_min,
            upper=self.clamp_max,
            low_guard=self.low_guard,
            extras=self.offset_extras,
        )


@dataclass
class SelectionContext:
    split: str
    frames: int
    max_clips: int
    seed: Optional[int]
    deterministic: Optional[bool]
    decoder: Mapping[str, Any]
    tolerances: Mapping[str, Any]
    sweep: Dict[str, Any]
    temperature: Optional[float]
    bias: Optional[float]
    run_id: Optional[str]
    start_time: float
    end_time: float

    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class SelectionResult:
    onset_threshold: float
    offset_threshold: float
    k_onset: int
    onset_event_f1: float
    offset_event_f1: float
    mean_event_f1: float
    onset_f1: Optional[float] = None
    offset_f1: Optional[float] = None
    onset_pred_rate: Optional[float] = None
    onset_pos_rate: Optional[float] = None
    decoder_kind: Optional[str] = None
    decoder_settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SelectionMetadata:
    checkpoint: Path
    sidecar: Path
    payload: Dict[str, Any]


def _build_dataset_cli(split: str, frames: int | None, max_clips: int | None) -> List[str]:
    args: List[str] = []
    if split:
        args.extend(["--split", split])
    if frames is not None:
        args.extend(["--frames", str(frames)])
    if max_clips is not None and max_clips > 0:
        args.extend(["--max-clips", str(max_clips)])
    return args


def _format_decoder_settings_line(line: str) -> Dict[str, Any]:
    if not line:
        return {}
    stripped = line.strip()
    if not stripped.startswith("[decoder-settings]"):
        return {}
    payload = {}
    for token in stripped.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "[decoder-settings]":
            continue
        if key == "decoder":
            payload["decoder_kind"] = value
            continue
        try:
            num = float(value)
        except ValueError:
            payload[key] = value
        else:
            if abs(num - round(num)) <= 1e-9:
                payload[key] = int(round(num))
            else:
                payload[key] = num
    return payload


def _alias_decoder(payload: Dict[str, Any]) -> None:
    alias_map = {
        "onset_open": "decoder_onset_open",
        "onset_hold": "decoder_onset_hold",
        "onset_min_on": "decoder_onset_min_on",
        "onset_merge_gap": "decoder_onset_merge_gap",
        "offset_open": "decoder_offset_open",
        "offset_hold": "decoder_offset_hold",
        "offset_min_off": "decoder_offset_min_off",
        "offset_merge_gap": "decoder_offset_merge_gap",
    }
    for src, dst in alias_map.items():
        if src in payload and dst not in payload:
            payload[dst] = payload[src]


def _parse_eval_table(lines: Sequence[str]) -> List[Dict[str, Any]]:
    cleaned_lines = [ANSI_ESCAPE_RE.sub("", line) for line in lines]
    decoder_settings: Dict[str, Any] = {}
    for line in cleaned_lines:
        decoded = _format_decoder_settings_line(line)
        if decoded:
            decoder_settings = decoded
            break

    header_idx = None
    for i, line in enumerate(cleaned_lines):
        if line.strip().startswith("onset_thr"):
            header_idx = i
            break
    if header_idx is None:
        return []
    header = cleaned_lines[header_idx].strip().split("\t")
    col_idx = {name: idx for idx, name in enumerate(header)}
    required = {
        "onset_thr",
        "offset_thr",
        "onset_f1",
        "offset_f1",
        "onset_pred_rate",
        "onset_pos_rate",
        "onset_event_f1",
        "offset_event_f1",
    }
    if not required.issubset(col_idx):
        return []
    rows: List[Dict[str, Any]] = []
    for raw_line in cleaned_lines[header_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("[") or stripped.startswith("#"):
            continue
        parts = [segment.strip() for segment in raw_line.split("\t")]
        if len(parts) < len(header):
            continue
        def _scrub(cell: str) -> str:
            cell = cell.strip()
            if not cell:
                return cell
            # Drop any inline annotations (e.g. "0.42 [DEBUG ...]")
            for marker in (" [", "\t[", " {", " (", " <"):
                if marker in cell:
                    cell = cell.split(marker, 1)[0]
            if "[" in cell:
                cell = cell.split("[", 1)[0]
            if "#" in cell:
                cell = cell.split("#", 1)[0]
            return cell.strip()

        def _parse_float(col: str) -> float:
            idx = col_idx.get(col)
            if idx is None or idx >= len(parts):
                raise ValueError
            cell = _scrub(parts[idx])
            if not cell:
                raise ValueError
            try:
                return float(cell)
            except ValueError:
                match = FLOAT_PREFIX_RE.match(cell)
                if match:
                    return float(match.group(0))
                raise

        try:
            onset_thr = _parse_float("onset_thr")
            offset_thr = _parse_float("offset_thr")
            onset_f1 = _parse_float("onset_f1")
            offset_f1 = _parse_float("offset_f1")
            onset_pr = _parse_float("onset_pred_rate")
            onset_pos = _parse_float("onset_pos_rate")
            onset_ev = _parse_float("onset_event_f1")
            offset_ev = _parse_float("offset_event_f1")
        except ValueError:
            continue
        k_onset = 1
        if "k_onset" in col_idx:
            try:
                k_val = _parse_float("k_onset")
                k_onset = int(round(k_val))
            except ValueError:
                k_onset = 1
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
            "k_onset": k_onset,
        }
        rows.append(row)
    if decoder_settings:
        for row in rows:
            for key, value in decoder_settings.items():
                row.setdefault(key, value)
            _alias_decoder(row)
    return rows


def parse_eval_rows(lines: Sequence[str]) -> List[Dict[str, Any]]:
    """Public helper to expose parsed evaluation rows for downstream selectors."""
    rows = _parse_eval_table(lines)
    return [dict(row) for row in rows]


@dataclass
class SelectionRequest:
    ckpt: Path
    split: str = "val"
    frames: int = 96
    max_clips: int = 80
    sweep: SweepSpec = field(default_factory=lambda: SweepSpec(0.4, 0.4))
    decoder: Mapping[str, Any] = field(default_factory=dict)
    tolerances: Mapping[str, Any] = field(default_factory=dict)
    temperature: Optional[float] = None
    bias: Optional[float] = None
    seed: Optional[int] = None
    deterministic: Optional[bool] = None
    eval_extras: Sequence[str] = field(default_factory=tuple)
    verbose: Optional[str] = None
    log_path: Optional[Path] = None
    run_id: Optional[str] = None


def calibrate_and_score(request: SelectionRequest) -> Tuple[SelectionResult, SelectionContext, List[str]]:
    t_start = time.time()
    if not Path(EVAL_SCRIPT).exists():
        raise SelectionError(f"Evaluation script missing at {EVAL_SCRIPT}")

    sweep = request.sweep
    onset_candidates = sweep.onset_candidates()
    offset_candidates = sweep.offset_candidates()

    sweep_payload = {
        "delta": sweep.delta,
        "clamp_min": sweep.clamp_min,
        "clamp_max": sweep.clamp_max,
        "low_guard": sweep.low_guard,
        "onset_candidates": list(onset_candidates),
        "offset_candidates": list(offset_candidates),
        "k_onset": list(sweep.k_onset_candidates),
    }

    banner_bits = [
        f"split={request.split}",
        f"frames={request.frames}",
        f"clips={request.max_clips}",
        f"delta={sweep.delta:.3f}",
        f"guard≥{sweep.low_guard:.2f}",
    ]
    if request.temperature is not None:
        banner_bits.append(f"temp={request.temperature:.3f}")
    if request.bias is not None:
        banner_bits.append(f"bias={request.bias:.3f}")
    print("[selection] " + " ".join(banner_bits), flush=True)

    cmd: List[str] = [
        sys.executable,
        "-u",
        str(EVAL_SCRIPT),
        "--ckpt",
        str(request.ckpt),
    ]
    cmd.extend(_build_dataset_cli(request.split, request.frames, request.max_clips))
    need_grid = False
    if onset_candidates and offset_candidates and len(onset_candidates) != len(offset_candidates):
        need_grid = True

    if onset_candidates:
        cmd.extend(
            [
                "--prob_thresholds",
                ",".join(f"{v:.4f}" for v in onset_candidates),
            ]
        )
    if offset_candidates:
        cmd.extend(
            [
                "--offset_prob_thresholds",
                ",".join(f"{v:.4f}" for v in offset_candidates),
            ]
        )
    if request.temperature is not None:
        cmd.extend(["--temperature", str(request.temperature)])
    if request.bias is not None:
        cmd.extend(["--bias", str(request.bias)])
    if len(tuple(sweep.k_onset_candidates)) > 1:
        cmd.append("--sweep_k_onset")
    extras_list = list(request.eval_extras or [])
    has_grid_flag = any(token == "--grid_prob_thresholds" for token in extras_list)
    if need_grid and not has_grid_flag:
        cmd.append("--grid_prob_thresholds")
    if extras_list:
        cmd.extend(extras_list)
    if request.seed is not None:
        cmd.extend(["--seed", str(request.seed)])
    if request.deterministic is not None:
        flag = "--deterministic" if request.deterministic else "--no-deterministic"
        cmd.append(flag)
    if request.verbose:
        cmd.extend(["--verbose", request.verbose])
    cmd.extend(["--decoder", "auto"])

    log_path = request.log_path
    stdout_target: Optional[Path] = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_target = log_path

    timestamp = datetime.utcnow().isoformat() + "Z"
    log_handle = None
    if stdout_target is not None:
        log_handle = stdout_target.open("w", encoding="utf-8")
        log_handle.write(f"# selection run at {timestamp}\n")
        log_handle.flush()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        if log_handle is not None:
            log_handle.close()
        raise SelectionError(f"Failed to launch evaluation: {exc}") from exc

    assert proc.stdout is not None
    lines: List[str] = []
    try:
        for raw_line in proc.stdout:
            if log_handle is not None:
                log_handle.write(raw_line)
                log_handle.flush()
            # Mirror child output to our stdout for real-time visibility.
            sys.stdout.write(raw_line)
            sys.stdout.flush()
            lines.append(raw_line.rstrip("\n"))
    finally:
        proc.stdout.close()
        retcode = proc.wait()
        if log_handle is not None:
            log_handle.close()

    if retcode != 0:
        tail = "\n".join(lines[-10:]) if lines else "<no stdout captured>"
        raise SelectionError(
            "Evaluation script failed with code {code}\nLast output lines:\n{tail}".format(
                code=retcode, tail=tail
            )
        )
    rows = _parse_eval_table(lines)
    if not rows:
        tail = "\n".join(lines[-10:]) if lines else "<no stdout captured>"
        raise SelectionError(
            "Could not parse evaluation table from eval_thresholds output. "
            f"Last lines:\n{tail}"
        )
    best = rows[0]
    for row in rows[1:]:
        if row["ev_f1_mean"] > best["ev_f1_mean"] + 1e-9:
            best = row
        elif abs(row["ev_f1_mean"] - best["ev_f1_mean"]) <= 1e-9 and row["onset_event_f1"] > best["onset_event_f1"] + 1e-9:
            best = row
    decoder_payload = {k: best[k] for k in best.keys() if k.startswith("decoder_")}

    result = SelectionResult(
        onset_threshold=float(best["onset_thr"]),
        offset_threshold=float(best["offset_thr"]),
        k_onset=int(best.get("k_onset", 1)),
        onset_event_f1=float(best["onset_event_f1"]),
        offset_event_f1=float(best["offset_event_f1"]),
        mean_event_f1=float(best["ev_f1_mean"]),
        onset_f1=float(best["onset_f1"]),
        offset_f1=float(best["offset_f1"]),
        onset_pred_rate=float(best["onset_pred_rate"]),
        onset_pos_rate=float(best["onset_pos_rate"]),
        decoder_kind=best.get("decoder_kind"),
        decoder_settings=decoder_payload,
    )

    ctx = SelectionContext(
        split=request.split,
        frames=request.frames,
        max_clips=request.max_clips,
        seed=request.seed,
        deterministic=request.deterministic,
        decoder=dict(request.decoder),
        tolerances=dict(request.tolerances),
        sweep=sweep_payload,
        temperature=request.temperature,
        bias=request.bias,
        run_id=request.run_id,
        start_time=t_start,
        end_time=time.time(),
    )

    return result, ctx, lines


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.strip() or None


def _git_dirty(repo_root: Path) -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(out.strip())


def record_best(
    *,
    source: Path,
    destination: Path,
    result: SelectionResult,
    context: SelectionContext,
    repo_root: Optional[Path] = None,
    metadata_extra: Optional[Mapping[str, Any]] = None,
    copy_checkpoint: bool = True,
) -> SelectionMetadata:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_checkpoint:
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        try:
            tmp.unlink(missing_ok=True)
        except AttributeError:
            if tmp.exists():
                tmp.unlink()
        shutil.copy2(source, tmp)
        tmp.replace(destination)

    timestamp = datetime.utcnow().isoformat() + "Z"
    repo_root = repo_root or Path.cwd()
    metadata = {
        "timestamp": timestamp,
        "source": str(source),
        "destination": str(destination),
        "metrics": {
            "onset_event_f1": result.onset_event_f1,
            "offset_event_f1": result.offset_event_f1,
            "mean_event_f1": result.mean_event_f1,
            "onset_f1": result.onset_f1,
            "offset_f1": result.offset_f1,
            "onset_pred_rate": result.onset_pred_rate,
            "onset_pos_rate": result.onset_pos_rate,
        },
        "thresholds": {
            "onset": result.onset_threshold,
            "offset": result.offset_threshold,
            "k_onset": result.k_onset,
        },
        "decoder": dict(result.decoder_settings),
        "decoder_kind": result.decoder_kind,
        "context": {
            "split": context.split,
            "frames": context.frames,
            "max_clips": context.max_clips,
            "seed": context.seed,
            "deterministic": context.deterministic,
            "temperature": context.temperature,
            "bias": context.bias,
            "tolerances": dict(context.tolerances),
            "sweep": dict(context.sweep),
            "run_id": context.run_id,
            "duration_seconds": context.duration(),
        },
    }
    git_commit = _git_commit(repo_root)
    git_dirty = _git_dirty(repo_root)
    metadata["git"] = {
        "commit": git_commit,
        "dirty": git_dirty,
    }
    if metadata_extra:
        metadata["extra"] = dict(metadata_extra)
    sidecar = destination.with_suffix(destination.suffix + ".selection.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    print(
        f"[selection] updated {destination.name} mean_event_f1={result.mean_event_f1:.4f}",
        flush=True,
    )
    return SelectionMetadata(checkpoint=destination, sidecar=sidecar, payload=metadata)


def read_selection_metadata(path: Path) -> Optional[Dict[str, Any]]:
    sidecar = path.with_suffix(path.suffix + ".selection.json")
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def decoder_snapshot_from_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    train_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    metrics_cfg = train_cfg.get("metrics", {}) if isinstance(train_cfg, Mapping) else {}
    decoder_cfg = metrics_cfg.get("decoder", {})
    if isinstance(decoder_cfg, Mapping):
        return copy.deepcopy(decoder_cfg)
    return {}


def tolerance_snapshot_from_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    frame_cfg = dataset_cfg.get("frame_targets", {}) if isinstance(dataset_cfg, Mapping) else {}
    return copy.deepcopy(frame_cfg) if isinstance(frame_cfg, Mapping) else {}
