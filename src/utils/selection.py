from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import copy


FAST_SWEEP_CLIP_MIN = 0.02
FAST_SWEEP_CLIP_MAX = 0.98
FAST_RESULT_MIN = 0.02
FAST_RESULT_MAX = 0.98

DEFAULT_DELTA = 0.05
DEFAULT_LOW_GUARD = 0.10

EVAL_SCRIPT = Path("scripts") / "calib" / "eval_thresholds.py"


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


def _parse_eval_table(lines: Sequence[str]) -> Optional[Dict[str, Any]]:
    decoder_settings: Dict[str, Any] = {}
    for line in lines:
        decoded = _format_decoder_settings_line(line)
        if decoded:
            decoder_settings = decoded
            break

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("onset_thr"):
            header_idx = i
            break
    if header_idx is None:
        return None
    header = lines[header_idx].strip().split("\t")
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
        return None
    best: Optional[Dict[str, Any]] = None
    for raw_line in lines[header_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("["):
            continue
        parts = stripped.split("\t")
        try:
            onset_thr = float(parts[col_idx["onset_thr"]])
            offset_thr = float(parts[col_idx["offset_thr"]])
            onset_f1 = float(parts[col_idx["onset_f1"]])
            offset_f1 = float(parts[col_idx["offset_f1"]])
            onset_pr = float(parts[col_idx["onset_pred_rate"]])
            onset_pos = float(parts[col_idx["onset_pos_rate"]])
            onset_ev = float(parts[col_idx["onset_event_f1"]])
            offset_ev = float(parts[col_idx["offset_event_f1"]])
        except (KeyError, ValueError, IndexError):
            continue
        k_onset = 1
        if "k_onset" in col_idx:
            try:
                k_onset = int(float(parts[col_idx["k_onset"]]))
            except (ValueError, IndexError):
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
        if best is None or row["ev_f1_mean"] > best["ev_f1_mean"] + 1e-9:
            best = row
    if best is not None and decoder_settings:
        best.update(decoder_settings)
        _alias_decoder(best)
    return best


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
    if request.eval_extras:
        cmd.extend(request.eval_extras)
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

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout_target is not None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        with stdout_target.open("w", encoding="utf-8") as fh:
            fh.write(f"# selection run at {timestamp}\n")
            fh.write(stdout)
            if stderr:
                fh.write("\n# stderr\n")
                fh.write(stderr)
    if proc.returncode != 0:
        raise SelectionError(
            f"Evaluation script failed with code {proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    lines = stdout.splitlines()
    parsed = _parse_eval_table(lines)
    if parsed is None:
        raise SelectionError("Could not parse evaluation table from eval_thresholds output")

    result = SelectionResult(
        onset_threshold=float(parsed["onset_thr"]),
        offset_threshold=float(parsed["offset_thr"]),
        k_onset=int(parsed.get("k_onset", 1)),
        onset_event_f1=float(parsed["onset_event_f1"]),
        offset_event_f1=float(parsed["offset_event_f1"]),
        mean_event_f1=float(parsed["ev_f1_mean"]),
        onset_f1=float(parsed["onset_f1"]),
        offset_f1=float(parsed["offset_f1"]),
        onset_pred_rate=float(parsed["onset_pred_rate"]),
        onset_pos_rate=float(parsed["onset_pos_rate"]),
        decoder_kind=parsed.get("decoder_kind"),
        decoder_settings={k: parsed[k] for k in parsed.keys() if k.startswith("decoder_")},
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
