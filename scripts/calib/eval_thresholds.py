#!/usr/bin/env python3
"""TiViT-Piano threshold sweeper and evaluator.

Purpose:
    - Load a checkpoint, sweep onset/offset thresholds (optionally per-head),
      and report frame/event metrics plus Platt-calibrated stats.
    - Optionally dump logits, enforce monotonic sanity checks, and reuse
      cached results when the configuration matches a previous run.
    - Provide decoder/snapping overrides so downstream calibration scripts can
      experiment with gate settings without editing configs.

Key Functions/Classes:
    - ``RuntimeContext`` / ``LoaderContext`` / ``EvalPairContext``: capture
      immutable state needed across the multi-stage evaluation pipeline.
    - ``evaluate_threshold_pair()``: shared core that applies thresholds,
      hysteresis, optional post-processing, and returns metric payloads.
    - ``run_eval_combo()``: drives grid/per-head sweeps, progress logging,
      best-result tracking, and row recording for cache dumps.
    - ``main()``: CLI entry point that orchestrates config loading, model exec,
      sweeps, sanity checks, cache replay, and result persistence.

CLI Arguments:
    --ckpt PATH (default: checkpoints/tivit_best.pt)
        Checkpoint whose state_dict supplies model weights.
    --split {train,val,valid,test} / --max-clips INT / --frames INT / --only ID
        Dataset slice controls passed through to ``make_dataloader``.
    --thresholds FLOAT [FLOAT ...] / --prob_thresholds FLOAT [FLOAT ...]
        Explicit onset thresholds (logit or prob). Use matching ``--offset_*``
        flags to decouple the heads. ``--grid_prob_thresholds`` evaluates the
        Cartesian product; otherwise indices pair up.
    --head {onset,offset}
        Restrict sweeps to one head; requires a fixed threshold (``--fixed_*``)
        or a calibration file for the opposite head.
    --temperature[/-onset/-offset] FLOAT, --bias[/-onset/-offset] FLOAT
        Apply Platt-style calibration scalars before sigmoid; defaults to 1.0/0.0.
    --sanity_thresholds FLOAT [...]
        Verify monotonic onset prediction rates at the provided probability
        thresholds (forces fresh evaluation, bypasses cache).
    --model-return-per-tile / --dump_logits PATH
        Request per-tile logits from the model and/or persist frame logits to
        NPZ for later analysis.
    --decoder {auto,none,hysteresis} plus ``--low_ratio``, ``--min_on``,
        ``--gap_merge``, ``--decoder-<head>-open/hold/min_{on,off}/merge_gap``,
        ``--postproc-mode {never,eval_only,always}``
        Control hysteresis gates and snap/DP post-processing during sweeps and
        the final verification pass.
    --sweep_k_onset, --grid_prob_thresholds, --no_eval_cache,
    --legacy-eval-thresholds
        Additional sweep/caching toggles. Legacy flag routes to the frozen
        pre-refactor implementation for quick regression triage.
    --seed INT / --deterministic / --verbose {quiet,info,debug} / --log-file PATH
        Usual reproducibility and logging controls shared with other TiViT
        scripts.

Usage:
    python scripts/calib/eval_thresholds.py --ckpt checkpoints/tivit_best.pt --split val \\
        --prob_thresholds 0.15 0.2 0.25 --postproc-mode eval_only
"""

import argparse
import sys, json, time, math, os, torch, logging, copy, hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Mapping, MutableMapping, Dict, Any, Sequence, cast

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from decoder.decode import (
    DECODER_DEFAULTS,
    build_threshold_mask,
    decode_hysteresis,
    decoder_notice_text,
    format_decoder_settings,
    normalize_decoder_params,
    pool_roll_BT,
    resolve_decoder_from_config,
    resolve_decoder_gates,
    apply_postprocessing,
)
from tivit.decoder.global_fusion import (
    GlobalFusionConfig,
    FusionDebugState,
    resolve_global_fusion_config,
    build_batch_tile_mask,
    fuse_tile_logits,
)
from tivit.decoder.tile_keymap import TileMaskResult
from utils import load_config, align_pitch_dim, configure_verbosity
from utils.logging_utils import QUIET_INFO_FLAG
from utils.identifiers import canonical_video_id
from utils.time_grid import frame_to_sec
from data import make_dataloader
from models import build_model
from torch.utils.data import DataLoader, Subset
from utils.determinism import configure_determinism, resolve_deterministic_flag, resolve_seed
from utils.registration_refinement import RegistrationRefiner, resolve_registration_cache_path
from theory.key_prior_runtime import (
    KeyPriorRuntimeSettings,
    resolve_key_prior_settings,
    apply_key_prior_to_logits,
)

LOGGER = logging.getLogger("eval_thresholds")
QUIET_EXTRA = {QUIET_INFO_FLAG: True}


def _resolve_backend_label(cfg: Mapping[str, Any] | None) -> str:
    model_cfg = cfg.get("model", {}) if isinstance(cfg, Mapping) else {}
    raw = model_cfg.get("backend", "vivit")
    label = str(raw).strip().lower() if raw is not None else "vivit"
    return label or "vivit"


def _normalise_split(split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    val = split.strip().lower()
    return "val" if val == "valid" else val


# Default probability grid used when sweeping thresholds without an explicit
# list.  We parse lists manually so callers can provide comma-separated values
# without escaping leading minus signs.
DEFAULT_THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

EVAL_CACHE_VERSION = 1
EVAL_CACHE_FILENAME = "eval_cache.json"
_MONO_PROB_LOW = 5e-4
_MONO_PROB_HIGH = 0.99
BAD_CLIP_RETRY_LIMIT = 3


@dataclass
class RuntimeContext:
    """Container for configuration and derived runtime flags."""

    cfg: Dict[str, Any]
    dataset_cfg: Dict[str, Any]
    model_cfg: Dict[str, Any]
    seed: int
    deterministic: bool
    split: str
    decode_fps: float
    hop_seconds: float
    event_tolerance: float
    key_prior_midi_low: int
    key_prior_settings: "KeyPriorRuntimeSettings"
    agg_mode: str
    agg_top_k: int
    agg_tau_sum: float
    default_k_onset: int
    default_k_offset: int
    include_k_column: bool
    k_candidates: List[int]
    preview_prob_threshold: float
    debug_mode: bool
    avlag_disabled: bool
    backend_label: str
    model_tiles: int
    return_per_tile_requested: bool
    fusion: GlobalFusionConfig
    stage_durations: Dict[str, float]
    only_id: Optional[str]


@dataclass
class LoaderContext:
    """Data loader wrapper capturing dataset metadata for progress logs."""

    val_loader: DataLoader
    dataset: Any
    target_clips: Optional[int]
    target_display: str
    registration_refiner: Optional[RegistrationRefiner] = None
    reg_meta_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    reg_meta_path: Optional[Path] = None


@dataclass
class EvalResults:
    """Outputs from the forward pass loop."""

    onset_logits_list: List[torch.Tensor]
    offset_logits_list: List[torch.Tensor]
    pitch_logits_list: List[torch.Tensor]
    onset_probs: List[torch.Tensor]
    offset_probs: List[torch.Tensor]
    onset_tgts: List[torch.Tensor]
    offset_tgts: List[torch.Tensor]
    clips_done: int
    elapsed_data: float
    throughput: float
    lag_ms_samples: List[float]
    lag_source_counter: Counter[str]
    skip_paths: set[str]
    bad_clip_counts: Dict[str, int]
    skipped_batches: int
    tile_preview_stats: Dict[str, Any]
    tile_boundary_hist: Counter[int]
    fusion_debug: Optional[FusionDebugState]


@dataclass
class TileBatchArtifacts:
    mask: Optional[torch.Tensor]
    fusion_targets: Dict[str, torch.Tensor]
    pitch_logits: Optional[torch.Tensor]


@dataclass
class EvalPairContext:
    """Holds immutable sweep-time tensors and decoder settings."""

    agg_mode: str
    agg_top_k: int
    default_k_onset: int
    default_k_offset: int
    hop_seconds: float
    decode_fps: float
    event_tolerance: float
    onset_temperature: float
    offset_temperature: float
    onset_bias: float
    offset_bias: float
    onset_probs: torch.Tensor
    offset_probs: torch.Tensor
    onset_true_bin: torch.Tensor
    offset_true_bin: torch.Tensor
    decoder_kind: str
    decoder_params: Mapping[str, Any]
    onset_decoder: Mapping[str, Any]
    offset_decoder: Mapping[str, Any]
    postproc_modules: Sequence[str]
    postproc_debug: bool
    cfg: Mapping[str, Any]
    decoder_notice_printed: bool = False

    def mark_decoder_notice_printed(self) -> None:
        self.decoder_notice_printed = True


def _format_per_tile_shape(tensor: torch.Tensor) -> str:
    if not torch.is_tensor(tensor):
        return str(type(tensor))
    if tensor.dim() != 4:
        return str(tuple(int(v) for v in tensor.shape))
    b, t, tiles, keys = tensor.shape
    return f"B={b} T={t} tiles={tiles} K={keys}"


def _assert_per_tile_layout(
    tensor: torch.Tensor,
    *,
    label: str,
    expected_tiles: int,
    backend_label: str,
) -> None:
    if not torch.is_tensor(tensor):
        raise ValueError(f"{label} tensor missing for backend '{backend_label}'")
    if tensor.dim() != 4:
        raise ValueError(
            f"{label} logits must be rank-4 (B, T, tiles, K); got shape {tuple(tensor.shape)} "
            f"(backend={backend_label})"
        )
    if expected_tiles > 0 and tensor.shape[2] != expected_tiles:
        raise ValueError(
            f"{label} tensor tiles={tensor.shape[2]} does not match configured tiles={expected_tiles} "
            f"(backend={backend_label})"
        )


def _json_sanitize(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalize_threshold_list(values):
    if values is None:
        return None
    return [round(float(v), 6) for v in values]


def _snapshot_decoder_gates(params: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for head, cfg in params.items():
        if not isinstance(cfg, Mapping):
            continue
        snapshot[head] = {
            "open": round(float(cfg.get("open", 0.0)), 6),
            "hold": round(float(cfg.get("hold", 0.0)), 6),
            "min_on": int(cfg.get("min_on", 0)),
            "min_off": int(cfg.get("min_off", cfg.get("min_on", 0))),
            "merge_gap": int(cfg.get("merge_gap", 0)),
            "median": int(cfg.get("median", 1)),
        }
    return snapshot


def _hash_cache_fingerprint(fingerprint: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_sanitize(fingerprint), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_eval_cache_db(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {"version": EVAL_CACHE_VERSION, "entries": {}}
    except json.JSONDecodeError:
        return {"version": EVAL_CACHE_VERSION, "entries": {}}
    if not isinstance(data, dict):
        return {"version": EVAL_CACHE_VERSION, "entries": {}}
    if data.get("version") != EVAL_CACHE_VERSION:
        return {"version": EVAL_CACHE_VERSION, "entries": {}}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    data["entries"] = entries
    return data


def _persist_eval_cache_db(path: Path, db: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(db, handle, indent=2, sort_keys=True, default=_json_sanitize)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    tmp.replace(path)


def _load_registration_metadata(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[canonical_video_id(str(key))] = value
    return result


def _resolve_clip_ids(paths: Sequence[str], batch_size: int) -> List[Optional[str]]:
    clip_ids: List[Optional[str]] = []
    for idx in range(batch_size):
        clip_id = None
        if idx < len(paths):
            try:
                clip_id = canonical_video_id(Path(paths[idx]).stem)
            except Exception:
                clip_id = canonical_video_id(Path(str(paths[idx])).stem)
        clip_ids.append(clip_id)
    return clip_ids


def _align_batch_targets_to_logits(batch_data: Mapping[str, Any], head_name: str, ref_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    roll_key = f"{head_name}_roll"
    target_tensor = batch_data.get(roll_key)
    if target_tensor is None or not torch.is_tensor(target_tensor):
        return None
    aligned = target_tensor.to(ref_tensor.device).float()
    aligned = pool_roll_BT(aligned, ref_tensor.shape[1])
    aligned = align_pitch_dim(ref_tensor, aligned, head_name)
    if aligned.shape != ref_tensor.shape:
        return None
    return aligned


def _process_per_tile_outputs(
    *,
    batch: Mapping[str, Any],
    paths: Sequence[str],
    onset_tile: torch.Tensor,
    offset_tile: torch.Tensor,
    pitch_tile: torch.Tensor,
    model_tiles: int,
    backend_label: str,
    preview_prob_threshold: float,
    tile_preview_stats: Dict[str, Any],
    tile_key_mask_cache: Dict[str, TileMaskResult],
    tile_key_mask_cushion: int,
    reg_meta_cache: MutableMapping[str, Dict[str, Any]],
    reg_refiner: Optional[RegistrationRefiner],
    fusion_enabled: bool,
    fusion_debug_state: Optional[FusionDebugState],
    comparison_enabled: bool,
    tile_boundary_hist: Counter[int],
    onset_logits_neutral: torch.Tensor,
    offset_logits_neutral: torch.Tensor,
    pitch_logits_neutral: Optional[torch.Tensor],
    per_tile_shape_logged: bool,
    per_tile_target_issue_logged: bool,
) -> Tuple[TileBatchArtifacts, bool, bool]:
    _assert_per_tile_layout(
        onset_tile,
        label="stageA onset_tile",
        expected_tiles=model_tiles,
        backend_label=backend_label,
    )
    _assert_per_tile_layout(
        offset_tile,
        label="stageA offset_tile",
        expected_tiles=model_tiles,
        backend_label=backend_label,
    )
    _assert_per_tile_layout(
        pitch_tile,
        label="stageA pitch_tile",
        expected_tiles=model_tiles,
        backend_label=backend_label,
    )
    if not per_tile_shape_logged:
        print(
            "[per-tile][StageA] layout=(B,T,tiles,K) onset={} offset={} pitch={}".format(
                _format_per_tile_shape(onset_tile),
                _format_per_tile_shape(offset_tile),
                _format_per_tile_shape(pitch_tile),
            ),
            flush=True,
        )
        per_tile_shape_logged = True

    tile_preview_neutral = onset_tile.detach().mean(dim=2)
    preview_abs = (tile_preview_neutral - onset_logits_neutral).abs().max().item()
    tile_preview_stats["max_abs_diff"] = max(tile_preview_stats["max_abs_diff"], float(preview_abs))
    onset_targets_batch = batch["onset_roll"].to(tile_preview_neutral.device).float()
    onset_targets_batch = pool_roll_BT(onset_targets_batch, tile_preview_neutral.shape[1])
    onset_targets_batch = align_pitch_dim(tile_preview_neutral, onset_targets_batch, "onset")
    preview_valid = onset_targets_batch.shape == tile_preview_neutral.shape
    if not preview_valid and not per_tile_target_issue_logged:
        print(
            "[per-tile] skipping preview: target shape {} != preview {} (pooling/pitch mismatch)".format(
                tuple(onset_targets_batch.shape),
                tuple(tile_preview_neutral.shape),
            ),
            flush=True,
        )
        per_tile_target_issue_logged = True
    if preview_valid:
        preview_probs = torch.sigmoid(tile_preview_neutral)
        global_probs_neutral = torch.sigmoid(onset_logits_neutral)
        preview_preds = (preview_probs >= preview_prob_threshold).float()
        global_preds = (global_probs_neutral >= preview_prob_threshold).float()
        preview_f1 = _binary_f1(preview_preds.reshape(-1), onset_targets_batch.reshape(-1)) or 0.0
        global_f1 = _binary_f1(global_preds.reshape(-1), onset_targets_batch.reshape(-1)) or 0.0
        tile_preview_stats["max_f1_delta"] = max(
            tile_preview_stats["max_f1_delta"],
            abs(float(preview_f1) - float(global_f1)),
        )
        tile_preview_stats["count"] += 1

    tile_mask_tensor: Optional[torch.Tensor] = None
    if fusion_enabled or paths:
        clip_ids = _resolve_clip_ids(paths, onset_tile.shape[0])
        key_dim = int(onset_tile.shape[-1])
        tile_mask_batch = build_batch_tile_mask(
            clip_ids,
            reg_meta_cache=reg_meta_cache,
            reg_refiner=reg_refiner,
            mask_cache=tile_key_mask_cache,
            num_tiles=model_tiles,
            cushion_keys=tile_key_mask_cushion,
            n_keys=key_dim,
        )
        tile_mask_tensor = tile_mask_batch.tensor
        records = tile_mask_batch.records
        for idx, record in enumerate(records):
            tile_boundary_hist[record.boundary_keys] += 1
            if fusion_debug_state is not None:
                clip_ref = clip_ids[idx] if idx < len(clip_ids) else None
                fusion_debug_state.record_mask_result(record, clip_id=clip_ref)
    fusion_targets: Dict[str, torch.Tensor] = {}
    if comparison_enabled:
        if preview_valid:
            fusion_targets["onset"] = onset_targets_batch
        else:
            aligned_onset = _align_batch_targets_to_logits(batch, "onset", onset_logits_neutral)
            if aligned_onset is not None:
                fusion_targets["onset"] = aligned_onset
        offset_target = _align_batch_targets_to_logits(batch, "offset", offset_logits_neutral)
        if offset_target is not None:
            fusion_targets["offset"] = offset_target
        if torch.is_tensor(pitch_logits_neutral):
            pitch_target = _align_batch_targets_to_logits(batch, "pitch", pitch_logits_neutral)
            if pitch_target is not None:
                fusion_targets["pitch"] = pitch_target

    artifacts = TileBatchArtifacts(mask=tile_mask_tensor, fusion_targets=fusion_targets, pitch_logits=pitch_logits_neutral)
    return artifacts, per_tile_shape_logged, per_tile_target_issue_logged


def _init_progress_logger(args):
    """Return a log-handle (if any) and a closure for consistent progress prints."""

    log_handle = None
    if args.log_file:
        log_path = Path(args.log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        import atexit

        atexit.register(log_handle.close)

    def _log_progress(msg: str, *, force: bool = False) -> None:
        should_print = force or bool(args.progress)
        if should_print:
            print(msg, flush=True)
        if log_handle is not None:
            log_handle.write(msg + "\n")
            log_handle.flush()

    return log_handle, _log_progress


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _dataset_video_count(ds) -> str:
    if ds is None:
        return "?"
    try:
        if hasattr(ds, "samples"):
            return str(len(getattr(ds, "samples")))
        if hasattr(ds, "videos"):
            videos_attr = getattr(ds, "videos")
            try:
                return str(len(videos_attr))
            except TypeError:
                pass
        return str(len(ds))
    except Exception:
        return "?"


def _filter_batch(batch, keep_indices):
    """Clone ``batch`` keeping only entries at ``keep_indices`` for tensor/list fields."""

    if not keep_indices:
        return None
    paths_field = batch.get("path")
    total = len(paths_field) if isinstance(paths_field, list) else None
    filtered = {}
    for key, value in batch.items():
        if key == "path" and isinstance(paths_field, list):
            filtered[key] = [paths_field[i] for i in keep_indices]
            continue
        if total is not None:
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == total:
                idx_tensor = torch.as_tensor(keep_indices, dtype=torch.long, device=value.device)
                filtered[key] = value.index_select(0, idx_tensor)
                continue
            if isinstance(value, list) and len(value) == total:
                filtered[key] = [value[i] for i in keep_indices]
                continue
            if isinstance(value, tuple) and len(value) == total:
                filtered[key] = [value[i] for i in keep_indices]
                continue
        filtered[key] = value
    return filtered


def _handle_bad_batch(
    paths,
    exc: Exception,
    *,
    skip_paths: set[str],
    bad_clip_counts: Dict[str, int],
    skipped_batches: int,
    log_progress,
) -> int:
    """Update retry bookkeeping for batches that raised during inference."""

    skipped_batches += 1
    safe_paths = [str(p) for p in (paths or []) if p]
    if safe_paths:
        first = Path(safe_paths[0]).name
        extra = len(safe_paths) - 1
        clip_desc = f"{first}+{extra} more" if extra > 0 else first
    else:
        clip_desc = "<unknown>"
    err_type = type(exc).__name__
    log_progress(
        f"[warn] batch failed ({err_type}): clip={clip_desc} error={exc}",
        force=True,
    )
    for path in set(safe_paths):
        bad_clip_counts[path] += 1
        if bad_clip_counts[path] >= BAD_CLIP_RETRY_LIMIT and path not in skip_paths:
            skip_paths.add(path)
            log_progress(
                f"[progress] marked clip as bad after {BAD_CLIP_RETRY_LIMIT} failures: {Path(path).name}",
                force=True,
    )
    return skipped_batches


def _validate_global_pair(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    label: str,
) -> None:
    """Ensure calibration receives global tensors (B,T,P) without extra tile dims."""

    if not torch.is_tensor(logits) or not torch.is_tensor(targets):
        return
    if logits.dim() != targets.dim():
        raise ValueError(
            f"{label} logits/targets rank mismatch: got {tuple(logits.shape)} vs {tuple(targets.shape)}. "
            "Per-head calibration requires reduced global tensors of identical rank."
        )
    if logits.dim() not in (2, 3):
        raise ValueError(
            f"{label} logits have unsupported shape {tuple(logits.shape)}. "
            "Calibration expects (batch, time, keys) or (batch, keys). "
            "If per-tile tensors were requested, reduce them to the global shape before calibration."
        )


def _percentile_tensor(flat: torch.Tensor, q: float) -> float:
    if flat.numel() == 0:
        return 0.0
    try:
        return float(torch.quantile(flat, q).item())
    except (RuntimeError, AttributeError):
        return float(np.quantile(flat.numpy(), q))


def _summarize_probs(
    name: str,
    tensor: torch.Tensor,
    log_progress,
) -> Tuple[float, Dict[float, float]]:
    flat = tensor.reshape(-1).float()
    max_prob = float(flat.max().item()) if flat.numel() else 0.0
    stats = {
        0.95: _percentile_tensor(flat, 0.95),
        0.99: _percentile_tensor(flat, 0.99),
        0.995: _percentile_tensor(flat, 0.995),
    }
    log_progress(
        "[sweep] %s prob stats: max=%.4f p95=%.4f p99=%.4f p99.5=%.4f"
        % (name, max_prob, stats[0.95], stats[0.99], stats[0.995]),
        force=True,
    )
    return max_prob, stats


def _format_float_list(vals: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.3f}" for v in vals) + "]"


def evaluate_threshold_pair(
    on_thr: float,
    off_thr: float,
    use_logits: bool,
    *,
    ctx: EvalPairContext,
    k_onset: Optional[int] = None,
    apply_postproc: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if k_onset is None:
        k_onset = ctx.default_k_onset
    k_onset = max(1, int(k_onset))
    k_offset = max(1, int(ctx.default_k_offset))
    onset_open_thr: Optional[float] = None
    onset_hold_thr: Optional[float] = None
    offset_open_thr: Optional[float] = None
    offset_hold_thr: Optional[float] = None

    onset_probs_tensor = ctx.onset_probs if torch.is_tensor(ctx.onset_probs) else torch.as_tensor(ctx.onset_probs)
    offset_probs_tensor = ctx.offset_probs if torch.is_tensor(ctx.offset_probs) else torch.as_tensor(ctx.offset_probs)
    onset_probs_tensor = onset_probs_tensor.contiguous()
    offset_probs_tensor = offset_probs_tensor.contiguous()
    prob_tensor_for_post = onset_probs_tensor.detach()
    if not prob_tensor_for_post.is_floating_point():
        prob_tensor_for_post = prob_tensor_for_post.float()
    prob_tensor_for_post = prob_tensor_for_post.contiguous()
    fps_eval = 1.0 / ctx.hop_seconds if ctx.hop_seconds > 0 else ctx.decode_fps

    def _resolve_prob_thr(raw_value: float) -> float:
        try:
            raw_float = float(raw_value)
        except (TypeError, ValueError):
            raw_float = float("nan")
        prob_val = _logit_to_probability(raw_float) if use_logits else raw_float
        if not math.isfinite(prob_val):
            prob_val = 0.5
        return max(0.0, min(prob_val, 1.0))

    onset_thr_prob = _resolve_prob_thr(on_thr)
    offset_thr_prob = _resolve_prob_thr(off_thr)
    print(
        "[apply-thr] onset_thr={:.4f} offset_thr={:.4f} "
        "T_on={:.4f} b_on={:.4f} T_off={:.4f} b_off={:.4f}".format(
            onset_thr_prob,
            offset_thr_prob,
            ctx.onset_temperature,
            ctx.onset_bias,
            ctx.offset_temperature,
            ctx.offset_bias,
        ),
        flush=True,
    )

    onset_mask_bool = build_threshold_mask(
        onset_probs_tensor,
        onset_thr_prob,
        mode=ctx.agg_mode,
        cap_count=k_onset,
        top_k=ctx.agg_top_k,
    )
    offset_mask_bool = build_threshold_mask(
        offset_probs_tensor,
        offset_thr_prob,
        mode=ctx.agg_mode,
        cap_count=k_offset,
        top_k=ctx.agg_top_k,
    )

    onset_mask_float = onset_mask_bool.float()
    offset_mask_float = offset_mask_bool.float()
    onset_pred_mask = onset_mask_bool.clone()
    offset_pred_mask = offset_mask_bool.clone()
    onset_pred_bin = onset_mask_float
    offset_pred_bin = offset_mask_float

    if ctx.decoder_kind == "hysteresis":
        if not ctx.decoder_notice_printed:
            print(f"[decoder] {decoder_notice_text(ctx.decoder_kind, ctx.decoder_params)}", flush=True)
            ctx.mark_decoder_notice_printed()
        onset_open_thr, onset_hold_thr = resolve_decoder_gates(
            ctx.onset_decoder,
            fallback_open=onset_thr_prob,
            default_hold=DECODER_DEFAULTS["onset"]["hold"],
        )
        offset_open_thr, offset_hold_thr = resolve_decoder_gates(
            ctx.offset_decoder,
            fallback_open=offset_thr_prob,
            default_hold=DECODER_DEFAULTS["offset"]["hold"],
        )
        masked_onset_probs = (onset_probs_tensor * onset_mask_float).contiguous()
        masked_offset_probs = (offset_probs_tensor * offset_mask_float).contiguous()
        onset_pred_mask = decode_hysteresis(
            masked_onset_probs,
            onset_open_thr,
            onset_hold_thr,
            ctx.onset_decoder["min_on"],
            ctx.onset_decoder["min_off"],
            ctx.onset_decoder["merge_gap"],
            ctx.onset_decoder["median"],
        )
        offset_pred_mask = decode_hysteresis(
            masked_offset_probs,
            offset_open_thr,
            offset_hold_thr,
            ctx.offset_decoder["min_on"],
            ctx.offset_decoder["min_off"],
            ctx.offset_decoder["merge_gap"],
            ctx.offset_decoder["median"],
        )
        onset_pred_bin = onset_pred_mask.to(onset_probs_tensor.dtype)
        offset_pred_bin = offset_pred_mask.to(offset_probs_tensor.dtype)

    post_stats: Dict[str, Any] = {}
    postproc_applied = False
    should_postprocess = apply_postproc and bool(ctx.postproc_modules)
    need_baseline = should_postprocess or ctx.postproc_debug
    baseline_onset_mask = onset_pred_mask.clone() if need_baseline else None
    if should_postprocess:
        onset_post_mask, stage_stats = apply_postprocessing(
            onset_pred_mask,
            prob_tensor_for_post,
            fps=fps_eval,
            cfg=ctx.cfg,
            require_stats=ctx.postproc_debug,
        )
        onset_pred_mask = onset_post_mask
        postproc_applied = True
        if stage_stats:
            post_stats.update(stage_stats)
    onset_pred_bin = onset_pred_mask.to(prob_tensor_for_post.dtype)

    f1_on = _binary_f1(onset_pred_bin.reshape(-1), ctx.onset_true_bin.reshape(-1))
    f1_off = _binary_f1(offset_pred_bin.reshape(-1), ctx.offset_true_bin.reshape(-1))
    ev_f1_on = _event_f1(onset_pred_bin, ctx.onset_true_bin, ctx.hop_seconds, ctx.event_tolerance)
    ev_f1_off = _event_f1(offset_pred_bin, ctx.offset_true_bin, ctx.hop_seconds, ctx.event_tolerance)
    onset_pred_rate = onset_pred_bin.mean().item()
    onset_pos_rate = ctx.onset_true_bin.mean().item()

    f1_on = 0.0 if f1_on is None else f1_on
    f1_off = 0.0 if f1_off is None else f1_off
    ev_f1_on = 0.0 if ev_f1_on is None else ev_f1_on
    ev_f1_off = 0.0 if ev_f1_off is None else ev_f1_off

    if ctx.postproc_debug:
        pre_mask = baseline_onset_mask if baseline_onset_mask is not None else onset_pred_mask
        metrics_payload = {
            "post": _collect_onset_metrics(
                onset_pred_mask.float(),
                ctx.onset_true_bin,
                ctx.hop_seconds,
                ctx.event_tolerance,
                prob_tensor_for_post,
            ),
            "pre": _collect_onset_metrics(
                pre_mask.float(),
                ctx.onset_true_bin,
                ctx.hop_seconds,
                ctx.event_tolerance,
                prob_tensor_for_post,
            ),
        }
        post_stats.setdefault("metrics", metrics_payload)
    result_payload = {
        "onset_thr": float(on_thr),
        "offset_thr": float(off_thr),
        "decoder_onset_open": float(onset_open_thr) if onset_open_thr is not None else None,
        "decoder_onset_hold": float(onset_hold_thr) if onset_hold_thr is not None else None,
        "decoder_offset_open": float(offset_open_thr) if offset_open_thr is not None else None,
        "decoder_offset_hold": float(offset_hold_thr) if offset_hold_thr is not None else None,
        "decoder_kind": ctx.decoder_kind,
        "decoder_onset_min_on": int(ctx.onset_decoder["min_on"]),
        "decoder_onset_merge_gap": int(ctx.onset_decoder["merge_gap"]),
        "decoder_onset_median": int(ctx.onset_decoder["median"]),
        "decoder_offset_min_off": int(ctx.offset_decoder["min_off"]),
        "decoder_offset_merge_gap": int(ctx.offset_decoder["merge_gap"]),
        "decoder_offset_median": int(ctx.offset_decoder["median"]),
        "f1_on": float(f1_on),
        "f1_off": float(f1_off),
        "onset_pred_rate": float(onset_pred_rate),
        "onset_pos_rate": float(onset_pos_rate),
        "ev_f1_on": float(ev_f1_on),
        "ev_f1_off": float(ev_f1_off),
        "k_onset": int(k_onset),
        "use_logits": bool(use_logits),
    }
    result_payload["_postproc_applied"] = postproc_applied
    return result_payload, post_stats


def print_sweep_header(include_k_column: bool, printed_header: bool) -> bool:
    if printed_header:
        return True
    cols = ["onset_thr", "offset_thr"]
    if include_k_column:
        cols.append("k_onset")
    cols.extend(
        [
            "onset_f1",
            "offset_f1",
            "onset_pred_rate",
            "onset_pos_rate",
            "onset_event_f1",
            "offset_event_f1",
        ]
    )
    print("\t".join(cols))
    return True


def print_sweep_row(res: Mapping[str, Any], include_k_column: bool) -> None:
    values = [f"{res['onset_thr']:.2f}", f"{res['offset_thr']:.2f}"]
    if include_k_column:
        values.append(str(res["k_onset"]))
    values.extend(
        [
            f"{res['f1_on']:0.3f}",
            f"{res['f1_off']:0.3f}",
            f"{res['onset_pred_rate']:0.3f}",
            f"{res['onset_pos_rate']:0.3f}",
            f"{res['ev_f1_on']:0.3f}",
            f"{res['ev_f1_off']:0.3f}",
        ]
    )
    print("\t".join(values))


def run_sanity_thresholds(
    args,
    *,
    default_k_onset: int,
    sweep_apply_postproc: bool,
    onset_peak: float,
    offset_peak: float,
    onset_lower_hint: float,
    offset_lower_hint: float,
    eval_ctx: EvalPairContext,
) -> None:
    sanity_vals = args.sanity_thresholds or []
    if not sanity_vals:
        return
    unique_vals = sorted({round(float(v), 6) for v in sanity_vals})
    if not unique_vals:
        return

    def _clamp_prob(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    offset_ref: Optional[float] = None
    if args.offset_prob_thresholds:
        offset_ref = float(args.offset_prob_thresholds[0])
    elif args.fixed_offset_prob is not None:
        offset_ref = float(args.fixed_offset_prob)
    elif args.prob_thresholds:
        offset_ref = float(args.prob_thresholds[0])
    elif args.thresholds:
        offset_ref = _logit_to_probability(float(args.thresholds[0]))
    if offset_ref is None:
        offset_ref = 0.5
    offset_ref = _clamp_prob(offset_ref)
    onset_fmt = "[" + ",".join(f"{_clamp_prob(val):.4f}" for val in unique_vals) + "]"
    print(
        "[sanity] verifying monotonic onset_pred_rate thresholds={} offset_ref={:.4f}".format(
            onset_fmt,
            offset_ref,
        ),
        flush=True,
    )
    prev_rate: Optional[float] = None
    prev_thr: Optional[float] = None
    for idx, thr in enumerate(unique_vals, start=1):
        prob_thr = _clamp_prob(thr)
        res_payload, _ = evaluate_threshold_pair(
            prob_thr,
            offset_ref,
            False,
            ctx=eval_ctx,
            k_onset=default_k_onset,
            apply_postproc=sweep_apply_postproc,
        )
        rate = float(res_payload["onset_pred_rate"])
        print(
            "[sanity] #{idx} onset_thr={thr:.4f} onset_pred_rate={rate:.6f}".format(
                idx=idx,
                thr=prob_thr,
                rate=rate,
            ),
            flush=True,
        )
        if prev_rate is not None and rate > prev_rate + 1e-9:
            message = (
                "[sanity] ERROR: onset_pred_rate increased "
                "thr_prev={prev:.4f} rate_prev={rate_prev:.6f} "
                "thr_curr={curr:.4f} rate_curr={rate_curr:.6f}".format(
                    prev=prev_thr if prev_thr is not None else float("nan"),
                    rate_prev=prev_rate,
                    curr=prob_thr,
                    rate_curr=rate,
                )
            )
            print(message, file=sys.stderr, flush=True)
            raise RuntimeError(message)
        prev_rate = rate
        prev_thr = prob_thr


def replay_cache_entry(
    entry: Mapping[str, Any],
    *,
    include_k_column: bool,
    printed_header: bool,
) -> bool:
    rows = entry.get("rows") or []
    saved_at = entry.get("timestamp") or "unknown"
    key_hint = entry.get("key", "")
    if isinstance(rows, list):
        print(
            "[cache] eval cache hit key={key} combos={count} saved={stamp}".format(
                key=key_hint[:8] if isinstance(key_hint, str) else "n/a",
                count=len(rows),
                stamp=saved_at,
            ),
            flush=True,
        )
        if rows:
            printed_header = print_sweep_header(include_k_column, printed_header)
            for cached_row in rows:
                print_sweep_row(cached_row, include_k_column)
    summary = entry.get("summary_lines") or []
    for line in summary:
        print(line)
    return printed_header


def update_best_result(
    best_result: Optional[dict],
    best_post_stats: Optional[dict],
    *,
    res: Mapping[str, Any],
    stats: Optional[dict],
) -> Tuple[dict | None, dict | None]:
    ev_mean = 0.5 * (res["ev_f1_on"] + res["ev_f1_off"])
    if best_result is None:
        return {**res, "ev_mean": ev_mean}, copy.deepcopy(stats) if stats else None
    best_mean = best_result.get("ev_mean", -1.0)
    if ev_mean > best_mean + 1e-9:
        return {**res, "ev_mean": ev_mean}, copy.deepcopy(stats) if stats else None
    if abs(ev_mean - best_mean) <= 1e-9 and res["ev_f1_on"] > best_result["ev_f1_on"] + 1e-9:
        return {**res, "ev_mean": ev_mean}, copy.deepcopy(stats) if stats else None
    return best_result, best_post_stats


def write_post_logs(
    post_logs_dir: Path,
    *,
    best_row: Mapping[str, Any],
    stats: Mapping[str, Any] | None,
    split: str,
    frames_display: Any,
    max_clips_display: Any,
    decoder_settings_summary: str,
) -> None:
    if not stats:
        return
    try:
        post_logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    summary_payload = {
        "onset_thr": best_row.get("onset_thr"),
        "offset_thr": best_row.get("offset_thr"),
        "k_onset": best_row.get("k_onset"),
        "decoder": {
            "kind": best_row.get("decoder_kind"),
            "onset_open": best_row.get("decoder_onset_open"),
            "onset_hold": best_row.get("decoder_onset_hold"),
            "offset_open": best_row.get("decoder_offset_open"),
            "offset_hold": best_row.get("decoder_offset_hold"),
        },
        "metrics": stats.get("metrics"),
        "snap": stats.get("snap"),
        "dp": stats.get("dp"),
        "context": {
            "split": split,
            "frames": frames_display,
            "max_clips": max_clips_display,
            "decoder_notice": decoder_settings_summary,
        },
    }
    summary_path = post_logs_dir / "post_summary.json"
    per_key_payload: Dict[str, Any] = {}
    snap_stats = stats.get("snap") if isinstance(stats, Mapping) else None
    if isinstance(snap_stats, Mapping) and snap_stats.get("per_key"):
        per_key_payload["snap"] = snap_stats["per_key"]
    dp_stats = stats.get("dp") if isinstance(stats, Mapping) else None
    if isinstance(dp_stats, Mapping) and dp_stats.get("per_key"):
        per_key_payload["dp"] = dp_stats["per_key"]
    try:
        summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
        if per_key_payload:
            (post_logs_dir / "per_key_stats.json").write_text(
                json.dumps(per_key_payload, indent=2, sort_keys=True)
            )
    except OSError:
        pass


def run_eval_combo(
    on_thr: float,
    off_thr: float,
    use_logits: bool,
    *,
    k_onset: Optional[int],
    apply_postproc: bool,
    eval_ctx: EvalPairContext,
    include_k_column: bool,
    row_records: List[Dict[str, Any]],
    best_result: Optional[dict],
    best_post_stats: Optional[dict],
    total_evals: int,
    combo_state: Dict[str, Any],
    args,
    log_progress,
) -> Tuple[dict | None, dict | None, int]:
    res, post_stats = evaluate_threshold_pair(
        on_thr,
        off_thr,
        use_logits,
        ctx=eval_ctx,
        k_onset=k_onset,
        apply_postproc=apply_postproc,
    )
    print_sweep_row(res, include_k_column)
    row_records.append(dict(res))
    best_result, best_post_stats = update_best_result(best_result, best_post_stats, res=res, stats=post_stats)
    total_evals += 1
    combo_state["combo_idx"] += 1
    combo_idx = combo_state["combo_idx"]
    num_combos = combo_state.get("num_combos", 0)
    if args.progress and num_combos > 0:
        now = time.time()
        last_grid_print = combo_state.get("last_grid_print", combo_state.get("start_time", now))
        progress_force = combo_idx == 1 or combo_idx == num_combos
        if progress_force or now - last_grid_print >= args.progress_interval:
            elapsed = now - combo_state.get("start_time", now)
            if combo_idx > 0 and num_combos:
                remaining = max(num_combos - combo_idx, 0)
                eta_seconds = (elapsed / combo_idx) * remaining if combo_idx else 0.0
                eta_display = _format_seconds(eta_seconds)
            else:
                eta_display = "??:??"
            k_display = k_onset if k_onset is not None else eval_ctx.default_k_onset
            log_progress(
                f"[progress] grid {combo_idx}/{num_combos}  onset_thr={on_thr:.3f}  offset_thr={off_thr:.3f}  k_onset={k_display}  elapsed={_format_seconds(elapsed)}  eta≈{eta_display}",
                force=progress_force,
            )
            combo_state["last_grid_print"] = now
    return best_result, best_post_stats, total_evals


def _extract_lag_values(value):
    """Flatten heterogeneous lag_ms structures into a list of floats."""

    vals: List[float] = []
    if value is None:
        return vals
    if torch.is_tensor(value):
        flat = value.detach().cpu().reshape(-1).tolist()
        for item in flat:
            try:
                fval = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fval):
                vals.append(fval)
        return vals
    if isinstance(value, (list, tuple)):
        for item in value:
            vals.extend(_extract_lag_values(item))
        return vals
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return vals
    if math.isfinite(fval):
        vals.append(fval)
    return vals


def _extract_lag_sources(value):
    """Normalize lag source annotations into a flat list of strings."""

    sources: List[str] = []
    if value is None:
        return sources
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item:
                sources.append(item)
    elif isinstance(value, str) and value:
        sources.append(value)
    return sources


def _parse_cli_args(argv: Sequence[str]):
    """Parse CLI args plus derived settings used throughout evaluation."""

    try:
        logit_thrs = _parse_list(argv, "thresholds")
        prob_thrs = _parse_list(argv, "prob_thresholds")
        offset_logit_thrs = _parse_list(argv, "offset_thresholds")
        offset_prob_thrs = _parse_list(argv, "offset_prob_thresholds")
        sanity_prob_thrs = _parse_list(argv, "sanity_thresholds")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
    ap.add_argument("--thresholds", metavar="T", nargs="*", help="Logit threshold values")
    ap.add_argument(
        "--offset_thresholds",
        metavar="T",
        nargs="*",
        help="Logit threshold values for the offset head (default: reuse onset thresholds)",
    )
    ap.add_argument(
        "--prob_thresholds",
        metavar="P",
        nargs="*",
        help="Probability threshold values",
    )
    ap.add_argument(
        "--offset_prob_thresholds",
        metavar="P",
        nargs="*",
        help="Probability threshold values for the offset head (default: reuse onset thresholds)",
    )
    ap.add_argument("--calibration", help="JSON file with calibrated thresholds")
    ap.add_argument("--head", choices=["onset", "offset"], help="Sweep thresholds for only one head")
    ap.add_argument("--fixed_offset_prob", type=float)
    ap.add_argument("--fixed_offset_logit", type=float)
    ap.add_argument("--fixed_onset_prob", type=float)
    ap.add_argument("--fixed_onset_logit", type=float)
    ap.add_argument(
        "--sanity_thresholds",
        metavar="P",
        nargs="*",
        help="Optional onset probability thresholds for monotonic sanity verification",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Scale logits by this temperature before sigmoid; >1 softens predictions",
    )
    ap.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Additive bias applied to logits before sigmoid",
    )
    ap.add_argument(
        "--temperature-onset",
        type=float,
        help="Override onset head temperature prior to sigmoid",
    )
    ap.add_argument(
        "--temperature-offset",
        type=float,
        help="Override offset head temperature prior to sigmoid",
    )
    ap.add_argument(
        "--bias-onset",
        type=float,
        help="Override onset head logit bias prior to sigmoid",
    )
    ap.add_argument(
        "--bias-offset",
        type=float,
        help="Override offset head logit bias prior to sigmoid",
    )
    ap.add_argument("--split", choices=["train", "val", "valid", "test"], help="Dataset split to evaluate")
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--only", help="Restrict evaluation to a single canonical video id")
    ap.add_argument(
        "--model-return-per-tile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request per-tile logits from the model during eval (default: config model.return_per_tile or False)",
    )
    ap.add_argument(
        "--verbose",
        choices=["quiet", "info", "debug"],
        help="Logging verbosity (default: quiet or $TIVIT_VERBOSE)",
    )
    ap.add_argument("--no-avlag", action="store_true", help="Disable audio/video lag estimation for isolation")
    ap.add_argument(
        "--dump_logits",
        default="",
        help="Optional path to save per-frame logits as a compressed NPZ",
    )
    ap.add_argument(
        "--grid_prob_thresholds",
        action="store_true",
        help="Evaluate the Cartesian product of onset/offset probability thresholds",
    )
    ap.add_argument(
        "--no_eval_cache",
        action="store_true",
        help="Disable eval cache reuse and force fresh evaluation",
    )
    ap.add_argument(
        "--sweep_k_onset",
        action="store_true",
        help="When aggregation mode is k_of_p, sweep k_onset over {1,2,3}",
    )
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable periodic progress logging",
    )
    ap.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Minimum number of seconds between progress prints",
    )
    ap.add_argument(
        "--log-file",
        type=str,
        help="Optional file path to tee progress logs",
    )
    ap.add_argument(
        "--legacy-eval-thresholds",
        action="store_true",
        help="Route execution through scripts/calib/eval_thresholds_legacy.py",
    )
    ap.add_argument(
        "--decoder",
        choices=["auto", "none", "hysteresis"],
        default="auto",
        help="Temporal decoder applied during evaluation (default: auto → hysteresis using config)",
    )
    ap.add_argument(
        "--low_ratio",
        type=float,
        default=None,
        help="Multiplier to derive the low hysteresis threshold (default: config or 0.6)",
    )
    ap.add_argument(
        "--min_on",
        type=int,
        default=None,
        help="Drop predicted on-segments shorter than this many frames (default: config or 2)",
    )
    ap.add_argument(
        "--min_off",
        type=int,
        default=None,
        help="Merge gaps shorter than this many frames between ons (default: config or 2)",
    )
    ap.add_argument(
        "--gap_merge",
        type=int,
        default=None,
        help="Merge on-segments separated by gaps <= this many frames (default: config or 1)",
    )
    ap.add_argument(
        "--decoder-onset-open",
        type=float,
        help="Override onset decoder open gate (probability, default: config)",
    )
    ap.add_argument(
        "--decoder-onset-hold",
        type=float,
        help="Override onset decoder hold gate (probability, default: config or derived from low_ratio)",
    )
    ap.add_argument(
        "--decoder-onset-min-on",
        type=int,
        help="Override onset decoder minimum on length in frames (default: config)",
    )
    ap.add_argument(
        "--decoder-onset-merge-gap",
        type=int,
        help="Override onset decoder merge gap in frames (default: config)",
    )
    ap.add_argument(
        "--decoder-offset-open",
        type=float,
        help="Override offset decoder open gate (probability, default: config)",
    )
    ap.add_argument(
        "--decoder-offset-hold",
        type=float,
        help="Override offset decoder hold gate (probability, default: config or derived from low_ratio)",
    )
    ap.add_argument(
        "--decoder-offset-min-off",
        type=int,
        help="Override offset decoder minimum off length in frames (default: config)",
    )
    ap.add_argument(
        "--decoder-offset-merge-gap",
        type=int,
        help="Override offset decoder merge gap in frames (default: config)",
    )
    ap.add_argument(
        "--median",
        type=int,
        default=None,
        help="Odd window size for optional time-axis median smoothing (default: config or 3)",
    )
    ap.add_argument(
        "--postproc-mode",
        choices=["never", "eval_only", "always"],
        default="eval_only",
        help="Control decoder post-processing: never=disable, eval_only=skip sweeps but run on final eval, always=run everywhere",
    )
    ap.add_argument("--seed", type=int, help="Seed for RNGs and dataloader shuffling")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle deterministic torch backends (default: config or enabled)",
    )
    args = ap.parse_args(argv)
    if args.split == "val":
        args.split = "valid"
    if args.legacy_eval_thresholds:
        from scripts.calib import eval_thresholds_legacy as legacy_eval

        legacy_argv = [arg for arg in argv if arg != "--legacy-eval-thresholds"]
        prev_argv = sys.argv
        try:
            sys.argv = [sys.argv[0], *legacy_argv]
            legacy_eval.main()
        finally:
            sys.argv = prev_argv
        sys.exit(0)

    args.verbose = configure_verbosity(args.verbose)
    debug_mode = args.verbose == "debug"
    args.thresholds = logit_thrs
    args.prob_thresholds = prob_thrs
    args.offset_thresholds = offset_logit_thrs
    args.offset_prob_thresholds = offset_prob_thrs
    args.sanity_thresholds = sanity_prob_thrs

    if args.thresholds is not None and args.prob_thresholds is not None:
        print("error: --thresholds and --prob_thresholds are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if args.offset_thresholds is not None and args.offset_prob_thresholds is not None:
        print(
            "error: --offset_thresholds and --offset_prob_thresholds are mutually exclusive",
            file=sys.stderr,
        )
        sys.exit(1)

    onset_logit_list, offset_logit_list, logit_reuse = _resolve_threshold_lists(
        args.thresholds, args.offset_thresholds
    )
    onset_prob_list, offset_prob_list, prob_reuse = _resolve_threshold_lists(
        args.prob_thresholds, args.offset_prob_thresholds
    )

    args.thresholds = onset_logit_list
    args.offset_thresholds = offset_logit_list
    args.prob_thresholds = onset_prob_list
    args.offset_prob_thresholds = offset_prob_list

    if args.head is None:
        has_logit_lists = args.thresholds is not None or args.offset_thresholds is not None
        has_prob_lists = args.prob_thresholds is not None or args.offset_prob_thresholds is not None
        if has_logit_lists and has_prob_lists:
            print(
                "error: specify only logit or probability threshold lists when sweeping both heads",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.thresholds is not None and args.offset_thresholds is not None:
        if len(args.thresholds) != len(args.offset_thresholds):
            print(
                "error: --thresholds and --offset_thresholds must contain the same number of values",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.prob_thresholds is not None and args.offset_prob_thresholds is not None:
        if not args.grid_prob_thresholds and len(args.prob_thresholds) != len(args.offset_prob_thresholds):
            print(
                "error: probability lists must match lengths unless --grid_prob_thresholds is enabled",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.low_ratio is not None and args.low_ratio < 0.0:
        print("error: --low_ratio must be non-negative", file=sys.stderr)
        sys.exit(1)
    if args.min_on is not None and args.min_on < 0:
        print("error: --min_on must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.min_off is not None and args.min_off < 0:
        print("error: --min_off must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.gap_merge is not None and args.gap_merge < 0:
        print("error: --gap_merge must be >= 0", file=sys.stderr)
        sys.exit(1)
    if args.median is not None and (args.median < 1 or args.median % 2 == 0):
        print("error: --median must be an odd integer >= 1", file=sys.stderr)
        sys.exit(1)

    base_temperature = args.temperature if args.temperature is not None else 1.0
    base_bias = args.bias if args.bias is not None else 0.0
    onset_temperature = float(args.temperature_onset if args.temperature_onset is not None else base_temperature)
    offset_temperature = float(args.temperature_offset if args.temperature_offset is not None else base_temperature)
    onset_bias = float(args.bias_onset if args.bias_onset is not None else base_bias)
    offset_bias = float(args.bias_offset if args.bias_offset is not None else base_bias)
    onset_platt_stats = _init_platt_stats(_platt_stats_enabled(onset_temperature, onset_bias))
    offset_platt_stats = _init_platt_stats(_platt_stats_enabled(offset_temperature, offset_bias))

    if args.thresholds is None and args.prob_thresholds is None:
        if args.head is not None or not args.calibration:
            args.prob_thresholds = DEFAULT_THRESHOLDS.copy()

    return (
        args,
        debug_mode,
        onset_temperature,
        offset_temperature,
        onset_bias,
        offset_bias,
        onset_platt_stats,
        offset_platt_stats,
        prob_reuse,
        logit_reuse,
    )


def _prepare_runtime(args, debug_mode: bool, stage_durations: Dict[str, float]) -> RuntimeContext:
    """Load config, resolve dataset/model settings, and capture derived fields."""

    cfg = dict(load_config("configs/config_pianovam_fast.yaml"))
    backend_label = _resolve_backend_label(cfg)
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        cfg["model"] = model_cfg
    decoder_cfg_runtime = cfg.get("decoder", {}) or {}
    fusion_cfg = resolve_global_fusion_config(decoder_cfg_runtime)
    if fusion_cfg.enabled:
        head_desc = ", ".join(fusion_cfg.apply_to)
        print(
            f"[fusion] enabled (mode={fusion_cfg.mode}, heads={head_desc}, cushion={fusion_cfg.cushion_keys})",
            flush=True,
        )
        if fusion_cfg.consistency_check:
            print(
                f"[fusion] consistency check active for first {fusion_cfg.consistency_batches} batches",
                flush=True,
            )
    if args.model_return_per_tile is not None:
        model_cfg["return_per_tile"] = bool(args.model_return_per_tile)
    return_per_tile_requested = bool(model_cfg.get("return_per_tile"))
    if fusion_cfg.needs_per_tile and not return_per_tile_requested:
        model_cfg["return_per_tile"] = True
        return_per_tile_requested = True
        print("[fusion] forcing per-tile logits for global fusion", flush=True)

    dataset_raw = cfg.get("dataset")
    dataset_cfg = dict(dataset_raw) if isinstance(dataset_raw, dict) else {}
    cfg["dataset"] = dataset_cfg

    if args.max_clips is not None:
        dataset_cfg["max_clips"] = args.max_clips
    if args.frames is not None:
        dataset_cfg["frames"] = args.frames

    only_id = canonical_video_id(args.only) if args.only else None
    if only_id:
        dataset_cfg["only_video"] = only_id

    env_disable = str(os.environ.get("AVSYNC_DISABLE", "")).strip().lower()
    avlag_disabled = bool(args.no_avlag) or env_disable in {"1", "true", "yes", "on"}
    if avlag_disabled:
        dataset_cfg["avlag_disabled"] = True

    if debug_mode:
        dataset_cfg["num_workers"] = 0
        dataset_cfg["persistent_workers"] = False
        dataset_cfg["pin_memory"] = False
        print("[debug] num_workers=0, persistent_workers=False, pin_memory=False", flush=True)

    seed = resolve_seed(args.seed, cfg)
    deterministic = resolve_deterministic_flag(args.deterministic, cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["deterministic"] = deterministic
    configure_determinism(seed, deterministic)
    print(
        f"[determinism] seed={seed} deterministic={'on' if deterministic else 'off'}",
        flush=True,
    )

    try:
        model_tiles = int(model_cfg.get("tiles", dataset_cfg.get("tiles", 3)))
    except Exception:
        model_tiles = 3

    decode_fps = float(dataset_cfg.get("decode_fps", 0.0) or 0.0)
    hop_seconds = float(dataset_cfg.get("hop_seconds", 0.0) or 0.0)
    if hop_seconds <= 0.0 and decode_fps > 0.0:
        hop_seconds = 1.0 / decode_fps
    if decode_fps <= 0.0 and hop_seconds > 0.0:
        decode_fps = 1.0 / hop_seconds
    if decode_fps <= 0.0:
        decode_fps = 30.0
    if hop_seconds <= 0.0:
        hop_seconds = 1.0 / decode_fps

    frame_targets_cfg = dataset_cfg.get("frame_targets", {}) or {}
    event_tolerance = float(frame_targets_cfg.get("tolerance", hop_seconds))
    midi_low_cfg = frame_targets_cfg.get("note_min")
    key_prior_midi_low = int(midi_low_cfg) if isinstance(midi_low_cfg, (int, float)) else 21
    split = args.split or dataset_cfg.get("split_val") or dataset_cfg.get("split") or "valid"

    frames_display = dataset_cfg.get("frames")
    max_clips_display = dataset_cfg.get("max_clips")
    only_display = only_id or "-"
    frame_text = frames_display if frames_display is not None else "?"
    max_clips_text = max_clips_display if max_clips_display is not None else "?"
    print(
        f"[progress] starting (split={split}, frames={frame_text}, max_clips={max_clips_text}, only={only_display})",
        flush=True,
    )

    metrics_cfg = cfg.get("training", {}).get("metrics", {}) or {}
    agg_cfg = metrics_cfg.get("aggregation", {}) or {}
    preview_prob_threshold = metrics_cfg.get("prob_threshold_onset", metrics_cfg.get("prob_threshold", 0.5))
    try:
        preview_prob_threshold = float(preview_prob_threshold)
    except (TypeError, ValueError):
        preview_prob_threshold = 0.5
    agg_mode = str(agg_cfg.get("mode", "any")).lower()
    agg_top_k = int(agg_cfg.get("top_k", 0) or 0)
    agg_tau_sum = float(agg_cfg.get("tau_sum", 0.0) or 0.0)
    agg_k_cfg = agg_cfg.get("k", {}) or {}
    default_k_onset = int(agg_k_cfg.get("onset", 1) or 1)
    default_k_offset = int(agg_k_cfg.get("offset", 1) or 1)

    key_prior_settings = resolve_key_prior_settings(decoder_cfg_runtime.get("key_prior"))
    if key_prior_settings.enabled:
        print(
            f"[decoder] key prior enabled (ref_head={key_prior_settings.ref_head}, apply_to={', '.join(key_prior_settings.apply_to)})",
            flush=True,
        )

    include_k_column = agg_mode == "k_of_p"
    if include_k_column and args.sweep_k_onset and args.head is None:
        k_candidates = sorted({default_k_onset, 1, 2, 3})
    else:
        k_candidates = [default_k_onset]

    return RuntimeContext(
        cfg=cfg,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        seed=seed,
        deterministic=deterministic,
        split=split,
        decode_fps=decode_fps,
        hop_seconds=hop_seconds,
        event_tolerance=event_tolerance,
        key_prior_midi_low=key_prior_midi_low,
        key_prior_settings=key_prior_settings,
        agg_mode=agg_mode,
        agg_top_k=agg_top_k,
        agg_tau_sum=agg_tau_sum,
        default_k_onset=default_k_onset,
        default_k_offset=default_k_offset,
        include_k_column=include_k_column,
        k_candidates=k_candidates,
        preview_prob_threshold=preview_prob_threshold,
        debug_mode=debug_mode,
        avlag_disabled=avlag_disabled,
        backend_label=backend_label,
        model_tiles=model_tiles,
        return_per_tile_requested=return_per_tile_requested,
        fusion=fusion_cfg,
        stage_durations=stage_durations,
        only_id=only_id,
    )


def _build_eval_loader(args, runtime: RuntimeContext, log_progress) -> LoaderContext:
    """Materialize the evaluation dataloader and compute target clip counts."""

    cfg = runtime.cfg
    split = runtime.split
    seed = runtime.seed
    stage_durations = runtime.stage_durations

    t_dataset_build0 = time.time()
    val_loader = make_dataloader(cfg, split=split, seed=seed)
    if isinstance(val_loader, dict):
        val_loader = val_loader.get(split, next(iter(val_loader.values())))
    if isinstance(val_loader, (list, tuple)):
        val_loader = val_loader[0]

    dataset = getattr(val_loader, "dataset", None)
    dataset_name = dataset.__class__.__name__ if dataset is not None else type(val_loader).__name__
    ds_len: Optional[int] = None
    dataset_count = "?"
    ok_videos = 0
    materialize_duration = 0.0
    if dataset is not None:
        materialize_stats = getattr(dataset, "_eval_materialize_stats", {}) or {}
        if isinstance(materialize_stats, dict):
            try:
                ok_videos = int(materialize_stats.get("videos") or 0)
            except (TypeError, ValueError):
                ok_videos = 0
            try:
                materialize_duration = float(materialize_stats.get("duration") or 0.0)
            except (TypeError, ValueError):
                materialize_duration = 0.0
        if materialize_duration == 0.0:
            try:
                materialize_duration = float(getattr(dataset, "_last_materialize_duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                materialize_duration = 0.0
        try:
            ds_len = len(dataset)
            dataset_count = str(ds_len)
        except TypeError:
            ds_len = None
            dataset_count = "?"
    dataset_elapsed = time.time() - t_dataset_build0
    stage_durations["dataset_init"] = dataset_elapsed
    batch_size_val = getattr(val_loader, "batch_size", None)
    batch_display = str(batch_size_val) if batch_size_val is not None else "?"
    worker_count = getattr(val_loader, "num_workers", None)
    worker_display = str(worker_count) if worker_count is not None else "?"
    video_count_display = _dataset_video_count(dataset)
    print(
        f"[progress] dataset ready (videos={video_count_display}, workers={worker_display})",
        flush=True,
    )
    log_progress(
        f"[progress] dataset ready in {_format_seconds(dataset_elapsed)} ({dataset_elapsed:.2f}s) "
        f"backend={dataset_name} len={dataset_count} batch={batch_display}",
        force=True,
    )
    frame_summary = getattr(dataset, "frame_target_summary", None)
    if frame_summary:
        frame_summary_display = frame_summary
        if runtime.avlag_disabled and "lag_source=" in frame_summary_display:
            prefix, suffix = frame_summary_display.split("lag_source=", 1)
            if "," in suffix:
                _, tail = suffix.split(",", 1)
                frame_summary_display = f"{prefix}lag_source=no_avlag,{tail}"
            else:
                frame_summary_display = f"{prefix}lag_source=no_avlag"
        log_progress(f"[progress] {frame_summary_display}", force=True)

    if ds_len == 0 and ok_videos > 0:
        print("[error] dataset len is 0 after audit ok>0 – eval entries were not materialized", flush=True)
        sys.exit(1)

    if ds_len is not None:
        resolved_cap = args.max_clips if args.max_clips is not None else ds_len
        target_clips = int(min(ds_len, int(resolved_cap)))
    else:
        target_clips = int(args.max_clips) if args.max_clips is not None else None

    if dataset is not None and target_clips is not None:
        try:
            base_len = len(dataset)
        except TypeError:
            base_len = None
        if base_len is not None:
            subset_cap = min(base_len, int(target_clips))
            subset_indices = list(range(subset_cap))
            dataset = Subset(dataset, subset_indices)
            num_workers = getattr(val_loader, "num_workers", 0)
            persistent_workers = getattr(val_loader, "persistent_workers", False)
            if num_workers <= 0:
                persistent_workers = False
            loader_kwargs = {
                "batch_size": getattr(val_loader, "batch_size", 1),
                "shuffle": False,
                "num_workers": num_workers,
                "pin_memory": getattr(val_loader, "pin_memory", False),
                "drop_last": getattr(val_loader, "drop_last", False),
                "collate_fn": getattr(val_loader, "collate_fn", None),
                "persistent_workers": persistent_workers,
                "timeout": getattr(val_loader, "timeout", 0),
            }
            prefetch_factor = getattr(val_loader, "prefetch_factor", None)
            if num_workers > 0 and prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = prefetch_factor
            pin_memory_device = getattr(val_loader, "pin_memory_device", None)
            if pin_memory_device:
                loader_kwargs["pin_memory_device"] = pin_memory_device
            worker_init_fn = getattr(val_loader, "worker_init_fn", None)
            if worker_init_fn is not None:
                loader_kwargs["worker_init_fn"] = worker_init_fn
            generator = getattr(val_loader, "generator", None)
            if generator is not None:
                loader_kwargs["generator"] = generator
            multiprocessing_context = getattr(val_loader, "multiprocessing_context", None)
            if multiprocessing_context is not None:
                loader_kwargs["multiprocessing_context"] = multiprocessing_context
            val_loader = DataLoader(dataset, **loader_kwargs)

    if materialize_duration > 0:
        stage_durations["materialize"] = materialize_duration

    target_display = str(target_clips) if target_clips is not None else "?"
    base_dataset = dataset
    if isinstance(base_dataset, Subset):
        base_dataset = base_dataset.dataset
    registration_refiner = None
    reg_meta_path: Optional[Path] = None
    if base_dataset is not None:
        registration_refiner = getattr(base_dataset, "registration_refiner", None)
        reg_path_attr = getattr(base_dataset, "registration_cache_path", None)
        if reg_path_attr:
            reg_meta_path = Path(reg_path_attr)
    if isinstance(registration_refiner, RegistrationRefiner):
        reg_meta_cache = registration_refiner.export_geometry_cache()
    else:
        registration_refiner = None
        reg_meta_cache = {}
    if (not reg_meta_cache) or reg_meta_path is None:
        reg_meta_path = reg_meta_path or resolve_registration_cache_path(os.environ.get("TIVIT_REG_REFINED"))
        if reg_meta_path:
            from_file = _load_registration_metadata(reg_meta_path)
            if from_file:
                reg_meta_cache = from_file

    return LoaderContext(
        val_loader=val_loader,
        dataset=dataset,
        target_clips=target_clips,
        target_display=target_display,
        registration_refiner=registration_refiner,
        reg_meta_cache=reg_meta_cache,
        reg_meta_path=reg_meta_path,
    )


def _setup_model(args, cfg: Mapping[str, Any]):
    """Instantiate TiViT-Piano and load checkpoint weights."""

    model = build_model(cfg)
    cfg_backend = _resolve_backend_label(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("config") if isinstance(ckpt, Mapping) else None
    ckpt_backend = _resolve_backend_label(ckpt_cfg) if isinstance(ckpt_cfg, Mapping) else "vivit"
    if ckpt_backend != cfg_backend:
        raise RuntimeError(
            f"Checkpoint backend '{ckpt_backend}' mismatches config backend '{cfg_backend}'. "
            "Set model.backend accordingly before running evaluation."
        )
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    return model


def _collect_logits(
    *,
    model: torch.nn.Module,
    loader_ctx: LoaderContext,
    args,
    runtime: RuntimeContext,
    onset_platt_stats: Dict[str, Any],
    offset_platt_stats: Dict[str, Any],
    onset_temperature: float,
    offset_temperature: float,
    onset_bias: float,
    offset_bias: float,
    log_progress,
) -> EvalResults:
    """Run the forward pass over the eval loader and collect logits/targets."""

    val_loader = loader_ctx.val_loader
    dataset = loader_ctx.dataset
    target_clips = loader_ctx.target_clips
    target_display = loader_ctx.target_display
    decode_fps = runtime.decode_fps
    hop_seconds = runtime.hop_seconds
    key_prior_midi_low = runtime.key_prior_midi_low
    key_prior_settings = runtime.key_prior_settings
    stage_durations = runtime.stage_durations
    avlag_disabled = runtime.avlag_disabled
    preview_prob_threshold = runtime.preview_prob_threshold
    model_tiles = runtime.model_tiles
    return_per_tile_requested = runtime.return_per_tile_requested
    fusion_cfg = runtime.fusion
    fusion_enabled = fusion_cfg.enabled
    fusion_debug_state = FusionDebugState(model_tiles) if fusion_enabled else None
    comparison_enabled = fusion_cfg.consistency_check and fusion_cfg.consistency_batches > 0
    comparison_batches_used = 0

    onset_logits_list, offset_logits_list = [], []
    pitch_logits_list: List[torch.Tensor] = []
    onset_probs, offset_probs = [], []
    onset_tgts, offset_tgts = [], []
    lag_ms_samples: List[float] = []
    lag_source_counter: Counter[str] = Counter()
    skip_paths: set[str] = set()
    bad_clip_counts: Dict[str, int] = Counter()
    skipped_batches = 0

    tile_boundary_hist: Counter[int] = Counter()
    tile_key_mask_cache: Dict[str, TileMaskResult] = {}
    tile_preview_stats = {"max_abs_diff": 0.0, "max_f1_delta": 0.0, "count": 0}
    tile_key_mask_cushion = fusion_cfg.cushion_keys if fusion_enabled else 2
    per_tile_shape_logged = False
    per_tile_target_issue_logged = False
    reg_refiner = loader_ctx.registration_refiner
    reg_meta_cache: Dict[str, Dict[str, Any]] = loader_ctx.reg_meta_cache or {}
    backend_label = runtime.backend_label
    if (return_per_tile_requested or fusion_enabled) and not reg_meta_cache:
        reg_cache_path = loader_ctx.reg_meta_path or resolve_registration_cache_path(
            os.environ.get("TIVIT_REG_REFINED")
        )
        if reg_cache_path:
            reg_meta_cache = _load_registration_metadata(reg_cache_path)
            if not reg_meta_cache:
                LOGGER.debug("per-tile: registration cache %s unavailable or empty", reg_cache_path)

    onset_temperature = float(onset_temperature)
    offset_temperature = float(offset_temperature)
    onset_bias = float(onset_bias)
    offset_bias = float(offset_bias)

    heartbeat_interval = max(10.0, float(args.progress_interval or 10.0))
    t_data0 = time.time()
    last_clip_print = t_data0
    last_heartbeat = t_data0
    last_clip_name = "-"
    first_batch_time = None
    clips_done = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if target_clips is not None and clips_done >= target_clips:
                break
            raw_paths = batch.get("path")
            if isinstance(raw_paths, (list, tuple)):
                paths = [str(p) for p in raw_paths]
            elif raw_paths is None:
                paths = []
            else:
                paths = [str(raw_paths)]
            if skip_paths and paths:
                keep_indices = [idx for idx, p in enumerate(paths) if p not in skip_paths]
                if len(keep_indices) != len(paths):
                    filtered_batch = _filter_batch(batch, keep_indices)
                    if filtered_batch is None:
                        blocked = ", ".join(Path(p).name for p in paths if p) or "<unknown>"
                        log_progress(
                            f"[progress] skipped batch (all blocked clips): {blocked}",
                            force=True,
                        )
                        continue
                    batch = filtered_batch
                    raw_paths = batch.get("path")
                    if isinstance(raw_paths, (list, tuple)):
                        paths = [str(p) for p in raw_paths]
                    elif raw_paths is None:
                        paths = []
                    else:
                        paths = [str(raw_paths)]
            try:
                x = batch["video"]
                if first_batch_time is None:
                    first_batch_time = time.time()
                    first_wait = first_batch_time - t_data0
                    stage_durations["first_batch"] = first_wait
                    log_progress(
                        f"[progress] first batch ready in {_format_seconds(first_wait)} ({first_wait:.2f}s) – includes decode/A/V lag warmup",
                        force=True,
                    )
                out = model(x, return_per_tile=return_per_tile_requested)

                onset_logits_raw = out["onset_logits"] if "onset_logits" in out else out.get("onset")
                offset_logits_raw = out["offset_logits"] if "offset_logits" in out else out.get("offset")
                pitch_logits = out.get("pitch_logits")
                pitch_logits_neutral = pitch_logits.detach() if torch.is_tensor(pitch_logits) else None
                onset_tile: Optional[torch.Tensor] = out.get("onset_tile") if return_per_tile_requested else None
                offset_tile: Optional[torch.Tensor] = out.get("offset_tile") if return_per_tile_requested else None
                pitch_tile: Optional[torch.Tensor] = out.get("pitch_tile") if return_per_tile_requested else None

                if onset_logits_raw is None or offset_logits_raw is None:
                    raise RuntimeError("Model output missing onset/offset logits")
                onset_logits_neutral = onset_logits_raw.detach()
                offset_logits_neutral = offset_logits_raw.detach()

                tile_mask_tensor: Optional[torch.Tensor] = None
                fusion_targets: Dict[str, torch.Tensor] = {}

                if return_per_tile_requested:
                    if onset_tile is None or offset_tile is None or pitch_tile is None:
                        raise RuntimeError("return_per_tile enabled but model did not return per-tile logits")
                    artifacts, per_tile_shape_logged, per_tile_target_issue_logged = _process_per_tile_outputs(
                        batch=batch,
                        paths=paths,
                        onset_tile=onset_tile,
                        offset_tile=offset_tile,
                        pitch_tile=pitch_tile,
                        model_tiles=model_tiles,
                        backend_label=runtime.backend_label,
                        preview_prob_threshold=preview_prob_threshold,
                        tile_preview_stats=tile_preview_stats,
                        tile_key_mask_cache=tile_key_mask_cache,
                        tile_key_mask_cushion=tile_key_mask_cushion,
                        reg_meta_cache=reg_meta_cache,
                        reg_refiner=reg_refiner,
                        fusion_enabled=fusion_enabled,
                        fusion_debug_state=fusion_debug_state,
                        comparison_enabled=comparison_enabled,
                        tile_boundary_hist=tile_boundary_hist,
                        onset_logits_neutral=onset_logits_neutral,
                        offset_logits_neutral=offset_logits_neutral,
                        pitch_logits_neutral=pitch_logits_neutral,
                        per_tile_shape_logged=per_tile_shape_logged,
                        per_tile_target_issue_logged=per_tile_target_issue_logged,
                    )
                    tile_mask_tensor = artifacts.mask
                    fusion_targets = artifacts.fusion_targets
                elif fusion_enabled:
                    raise RuntimeError("global fusion enabled but model did not emit per-tile logits")

                if fusion_enabled:
                    if tile_mask_tensor is None:
                        raise RuntimeError("global fusion enabled but tile mask tensor is unavailable")
                    primary_tile = onset_tile if onset_tile is not None else offset_tile
                    mask_device = primary_tile.device if primary_tile is not None else onset_logits_neutral.device
                    mask_dtype = primary_tile.dtype if primary_tile is not None else onset_logits_neutral.dtype
                    mask_tensor_device = tile_mask_tensor.to(mask_device, dtype=mask_dtype)
                    apply_heads = {head.lower() for head in fusion_cfg.apply_to}
                    comparison_recorded = False
                    if return_per_tile_requested and onset_tile is not None and "onset" in apply_heads:
                        fused_onset = fuse_tile_logits(onset_tile, mask_tensor_device, mode=fusion_cfg.mode)
                        if fusion_debug_state is not None:
                            fusion_debug_state.record_shape("onset", onset_tile.shape, fused_onset.shape)
                        if (
                            comparison_enabled
                            and fusion_debug_state is not None
                            and comparison_batches_used < fusion_cfg.consistency_batches
                        ):
                            target = fusion_targets.get("onset")
                            if target is not None and fusion_debug_state.record_comparison(
                                "onset",
                                baseline_logits=onset_logits_neutral,
                                fused_logits=fused_onset,
                                targets=target,
                                prob_threshold=preview_prob_threshold,
                                f1_fn=_binary_f1,
                            ):
                                comparison_recorded = True
                        onset_logits_neutral = fused_onset
                    if return_per_tile_requested and offset_tile is not None and "offset" in apply_heads:
                        fused_offset = fuse_tile_logits(offset_tile, mask_tensor_device, mode=fusion_cfg.mode)
                        if fusion_debug_state is not None:
                            fusion_debug_state.record_shape("offset", offset_tile.shape, fused_offset.shape)
                        if (
                            comparison_enabled
                            and fusion_debug_state is not None
                            and comparison_batches_used < fusion_cfg.consistency_batches
                        ):
                            target = fusion_targets.get("offset")
                            if target is not None and fusion_debug_state.record_comparison(
                                "offset",
                                baseline_logits=offset_logits_neutral,
                                fused_logits=fused_offset,
                                targets=target,
                                prob_threshold=preview_prob_threshold,
                                f1_fn=_binary_f1,
                            ):
                                comparison_recorded = True
                        offset_logits_neutral = fused_offset
                    if return_per_tile_requested and "pitch" in apply_heads and torch.is_tensor(pitch_tile) and torch.is_tensor(pitch_logits_neutral):
                        fused_pitch = fuse_tile_logits(pitch_tile, mask_tensor_device, mode=fusion_cfg.mode)
                        if fusion_debug_state is not None:
                            fusion_debug_state.record_shape("pitch", pitch_tile.shape, fused_pitch.shape)
                        if (
                            comparison_enabled
                            and fusion_debug_state is not None
                            and comparison_batches_used < fusion_cfg.consistency_batches
                        ):
                            target = fusion_targets.get("pitch")
                            if target is not None and fusion_debug_state.record_comparison(
                                "pitch",
                                baseline_logits=pitch_logits_neutral,
                                fused_logits=fused_pitch,
                                targets=target,
                                prob_threshold=preview_prob_threshold,
                                f1_fn=_binary_f1,
                            ):
                                comparison_recorded = True
                        pitch_logits = fused_pitch
                        pitch_logits_neutral = fused_pitch
                    if comparison_recorded:
                        comparison_batches_used += 1

                if onset_platt_stats["enabled"]:
                    onset_prob_neutral = torch.sigmoid(onset_logits_neutral)
                else:
                    onset_prob_neutral = None
                if offset_platt_stats["enabled"]:
                    offset_prob_neutral = torch.sigmoid(offset_logits_neutral)
                else:
                    offset_prob_neutral = None

                onset_logits = onset_logits_neutral / onset_temperature + onset_bias
                offset_logits = offset_logits_neutral / offset_temperature + offset_bias

                pitch_prior_tensor = None
                pitch_was_2d = False
                if torch.is_tensor(pitch_logits):
                    pitch_prior_tensor = pitch_logits
                    if pitch_prior_tensor.dim() == 2:
                        pitch_prior_tensor = pitch_prior_tensor.unsqueeze(1)
                        pitch_was_2d = True

                prior_inputs = {"onset": onset_logits, "offset": offset_logits}
                if torch.is_tensor(pitch_prior_tensor):
                    prior_inputs["pitch"] = pitch_prior_tensor
                prior_outputs = apply_key_prior_to_logits(
                    prior_inputs,
                    key_prior_settings,
                    fps=decode_fps,
                    midi_low=key_prior_midi_low,
                    midi_high=None,
                )
                if "onset" in prior_outputs:
                    onset_logits = prior_outputs["onset"]
                if "offset" in prior_outputs:
                    offset_logits = prior_outputs["offset"]
                if pitch_prior_tensor is not None and "pitch" in prior_outputs:
                    pitch_prior_tensor = prior_outputs["pitch"]
                    pitch_logits = pitch_prior_tensor.squeeze(1) if pitch_was_2d else pitch_prior_tensor

                onset_prob = torch.sigmoid(onset_logits)
                offset_prob = torch.sigmoid(offset_logits)

                if onset_prob_neutral is not None:
                    _update_platt_stats(onset_platt_stats, onset_prob.detach() - onset_prob_neutral)
                if offset_prob_neutral is not None:
                    _update_platt_stats(offset_platt_stats, offset_prob.detach() - offset_prob_neutral)

                onset_logits_list.append(onset_logits.detach().cpu())
                offset_logits_list.append(offset_logits.detach().cpu())
                onset_probs.append(onset_prob.detach().cpu())
                offset_probs.append(offset_prob.detach().cpu())

                if pitch_logits is not None:
                    if pitch_logits.dim() == 2:
                        pitch_logits = pitch_logits.unsqueeze(1)
                    pitch_logits_list.append(pitch_logits.detach().cpu())

                onset_tgts.append(batch["onset_roll"].float().cpu())
                offset_tgts.append(batch["offset_roll"].float().cpu())

                lag_vals = _extract_lag_values(batch.get("lag_ms"))
                if lag_vals:
                    lag_ms_samples.extend(lag_vals)
                if not avlag_disabled:
                    lag_sources = _extract_lag_sources(batch.get("lag_source"))
                    if lag_sources:
                        lag_source_counter.update(lag_sources)

                if runtime.debug_mode and len(onset_logits_list) == 1:
                    print("[DEBUG] batch video", x.shape, "onset_logits", onset_logits.shape)
                    print(
                        "[DEBUG] onset_roll nonzero=",
                        int(batch["onset_roll"].sum().item()),
                        "offset_roll nonzero=",
                        int(batch["offset_roll"].sum().item()),
                    )

                batch_size = int(x.shape[0]) if hasattr(x, "shape") and x.shape else 1
                clips_done += batch_size
                now = time.time()
                if paths:
                    candidate_id = canonical_video_id(Path(paths[-1]).name)
                    last_clip_name = candidate_id or Path(paths[-1]).name
                if now - last_heartbeat >= heartbeat_interval:
                    elapsed = now - t_data0
                    log_progress(
                        f"[progress] data pass heartbeat: processed_clips={clips_done} skips={len(skip_paths)} last_clip={last_clip_name} elapsed={_format_seconds(elapsed)}",
                        force=True,
                    )
                    last_heartbeat = now
                if args.progress:
                    progress_force = i == 0 or (target_clips is not None and clips_done >= target_clips)
                    if progress_force or now - last_clip_print >= args.progress_interval:
                        elapsed = now - t_data0
                        if target_clips is not None and target_clips > 0:
                            pct = min(100.0, 100.0 * clips_done / float(target_clips))
                            pct_display = f"{pct:5.1f}"
                        else:
                            pct_display = "?"
                        if clips_done == 0:
                            eta_display = "--:--"
                        elif target_clips is None or target_clips <= 0:
                            eta_display = "--:--"
                        else:
                            remaining = max(target_clips - clips_done, 0)
                            if remaining == 0:
                                eta_display = "00:00"
                            else:
                                eta_seconds = (elapsed / clips_done) * remaining
                                eta_display = _format_seconds(eta_seconds)
                        processed_display = clips_done if target_clips is None else min(clips_done, target_clips)
                        log_progress(
                            f"[progress] clips {processed_display}/{target_display}  ({pct_display}%)  elapsed={_format_seconds(elapsed)}  eta≈{eta_display}",
                            force=progress_force,
                        )
                        last_clip_print = now
                if target_clips is not None and clips_done >= target_clips:
                    break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                skipped_batches = _handle_bad_batch(
                    paths,
                    exc,
                    skip_paths=skip_paths,
                    bad_clip_counts=bad_clip_counts,
                    skipped_batches=skipped_batches,
                    log_progress=log_progress,
                )
                continue

    elapsed_data = time.time() - t_data0
    stage_durations["data_pass"] = elapsed_data
    throughput = clips_done / elapsed_data if elapsed_data > 0 else 0.0

    return EvalResults(
        onset_logits_list=onset_logits_list,
        offset_logits_list=offset_logits_list,
        pitch_logits_list=pitch_logits_list,
        onset_probs=onset_probs,
        offset_probs=offset_probs,
        onset_tgts=onset_tgts,
        offset_tgts=offset_tgts,
        clips_done=clips_done,
        elapsed_data=elapsed_data,
        throughput=throughput,
        lag_ms_samples=lag_ms_samples,
        lag_source_counter=lag_source_counter,
        skip_paths=skip_paths,
        bad_clip_counts=bad_clip_counts,
        skipped_batches=skipped_batches,
        tile_preview_stats=tile_preview_stats,
        tile_boundary_hist=tile_boundary_hist,
        fusion_debug=fusion_debug_state,
    )


def _summarize_data_pass(
    results: EvalResults,
    runtime: RuntimeContext,
    target_clips: Optional[int],
    log_progress,
):
    """Emit post-pass summaries including throughput, lag stats, and per-tile checks."""

    if runtime.fusion.enabled and results.fusion_debug is not None:
        for line in results.fusion_debug.summary_lines():
            print(line, flush=True)
        if runtime.fusion.consistency_check and not results.fusion_debug.comparison:
            print("[fusion] consistency check enabled but no comparison samples recorded", flush=True)
    elif runtime.return_per_tile_requested:
        stats = results.tile_preview_stats
        if stats["count"] > 0:
            print(
                "[per-tile] preview max_abs_diff={:.3e} max_f1_delta={:.3e} batches={}".format(
                    stats["max_abs_diff"],
                    stats["max_f1_delta"],
                    stats["count"],
                ),
                flush=True,
            )
        if results.tile_boundary_hist:
            hist_desc = ", ".join(f"{k}:{v}" for k, v in sorted(results.tile_boundary_hist.items()))
            print(f"[per-tile] boundary-keys histogram {{{hist_desc}}}", flush=True)

    processed_display = results.clips_done if target_clips is None else min(results.clips_done, target_clips or 0)
    skipped_display = len(results.skip_paths)
    elapsed_display = _format_seconds(results.elapsed_data)
    expected_display = target_clips if target_clips is not None else "?"
    log_progress(
        f"[progress] data pass done: clips={processed_display}, expected={expected_display}, skipped={skipped_display}, elapsed={elapsed_display}",
        force=True,
    )
    log_progress(
        f"[progress] throughput: {results.throughput:.2f} clips/s ({results.elapsed_data:.2f}s)",
        force=True,
    )
    if runtime.avlag_disabled:
        log_progress("[progress] A/V lag ms stats: disabled (all zero).", force=True)
    elif results.lag_ms_samples:
        lag_arr = np.asarray(results.lag_ms_samples, dtype=np.float32)
        lag_mean = float(lag_arr.mean())
        lag_median = float(np.median(lag_arr))
        lag_p95 = float(np.percentile(lag_arr, 95))
        log_progress(
            "[progress] A/V lag ms stats: mean={:.1f} median={:.1f} p95={:.1f} samples={}".format(
                lag_mean,
                lag_median,
                lag_p95,
                lag_arr.size,
            ),
            force=True,
        )
    if results.lag_source_counter and not runtime.avlag_disabled:
        top_sources = ", ".join(f"{src}:{cnt}" for src, cnt in results.lag_source_counter.most_common(3))
        log_progress(f"[progress] lag sources top: {top_sources}", force=True)
    if results.skipped_batches:
        log_progress(f"[progress] batches skipped due to errors: {results.skipped_batches}", force=True)
    if results.bad_clip_counts:
        summary_bits = ", ".join(f"{Path(p).name}:{count}" for p, count in results.bad_clip_counts.items())
        log_progress(f"[progress] bad clip retries: {summary_bits}", force=True)
    if results.skip_paths:
        skip_names = ", ".join(Path(p).name for p in sorted(results.skip_paths))
        log_progress(f"[progress] permanently skipped clips: {skip_names}", force=True)


def _assert_monotonic_rates(
    probs: torch.Tensor,
    *,
    label: str,
    agg_mode: str,
    cap_count: int,
    top_k: int,
) -> None:
    if probs.numel() == 0:
        raise RuntimeError(f"[sanity] {label} probability tensor is empty; cannot verify monotonicity")
    low_mask = build_threshold_mask(
        probs,
        _MONO_PROB_LOW,
        mode=agg_mode,
        cap_count=cap_count,
        top_k=top_k,
    )
    high_mask = build_threshold_mask(
        probs,
        _MONO_PROB_HIGH,
        mode=agg_mode,
        cap_count=cap_count,
        top_k=top_k,
    )
    low_rate = float(low_mask.float().mean().item())
    high_rate = float(high_mask.float().mean().item())
    if low_rate <= high_rate + 1e-9:
        raise RuntimeError(
            "[sanity] {} pred_rate did not decrease as threshold increased "
            "(low_thr={:.6f} rate={:.6f}, high_thr={:.2f} rate={:.6f})".format(
                label,
                _MONO_PROB_LOW,
                low_rate,
                _MONO_PROB_HIGH,
                high_rate,
            )
        )
    print(
        "[sanity] {} monotonic check passed: rate_low={:.6f} > rate_high={:.6f}".format(
            label, low_rate, high_rate
        ),
        flush=True,
    )


def _platt_stats_enabled(temp: float, bias: float, tol: float = 1e-9) -> bool:
    return abs(float(temp) - 1.0) > tol or abs(float(bias)) > tol


def _init_platt_stats(enabled: bool) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "count": 0,
        "sum": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def _update_platt_stats(stats: MutableMapping[str, Any], delta: torch.Tensor) -> None:
    if not stats.get("enabled"):
        return
    if delta is None or delta.numel() == 0:
        return
    delta_cpu = delta.detach().cpu()
    count = int(delta_cpu.numel())
    if count == 0:
        return
    stats["count"] = int(stats.get("count", 0)) + count
    stats["sum"] = float(stats.get("sum", 0.0)) + float(delta_cpu.sum().item())
    delta_min = float(delta_cpu.min().item())
    delta_max = float(delta_cpu.max().item())
    stats["min"] = float(min(float(stats.get("min", float("inf"))), delta_min))
    stats["max"] = float(max(float(stats.get("max", float("-inf"))), delta_max))


def _report_platt_stats(label: str, stats: Mapping[str, Any]) -> None:
    if not stats.get("enabled"):
        return
    count = int(stats.get("count", 0))
    if count == 0:
        return
    total = float(stats.get("sum", 0.0))
    mean = total / count if count else 0.0
    delta_min = float(stats.get("min", 0.0))
    delta_max = float(stats.get("max", 0.0))
    print(
        "[platt-shift] {label} Δprob min={min_val:.6f} mean={mean:.6f} max={max_val:.6f}".format(
            label=label,
            min_val=delta_min,
            mean=mean,
            max_val=delta_max,
        ),
        flush=True,
    )


def _format_summary_lines(
    best_result: Mapping[str, Any],
    *,
    final_postproc_applied: bool,
    postproc_modules: Sequence[str],
    args,
    onset_decoder: Mapping[str, Any],
    offset_decoder: Mapping[str, Any],
) -> List[str]:
    lines: List[str] = []
    modules_label = ",".join(postproc_modules) if postproc_modules else "none"
    status = "RAN" if final_postproc_applied else "SKIPPED"
    lines.append(f"POSTPROC: {status} during final eval (mode={args.postproc_mode}) modules={modules_label}")
    lines.append(
        "[best-event] mean_event_f1={:.3f} onset_event_f1={:.3f} offset_event_f1={:.3f} k_onset={}".format(
            best_result["ev_mean"],
            best_result["ev_f1_on"],
            best_result["ev_f1_off"],
            best_result.get("k_onset", 1),
        )
    )
    if not best_result.get("use_logits", False):
        lines.append(
            "[best-yaml] onset_prob_threshold={:.2f}, offset_prob_threshold={:.2f}, k_onset={}".format(
                best_result["onset_thr"],
                best_result["offset_thr"],
                best_result.get("k_onset", 1),
            )
        )
    onset_open_best = best_result.get("decoder_onset_open")
    onset_hold_best = best_result.get("decoder_onset_hold")
    onset_min_on = best_result.get("decoder_onset_min_on", onset_decoder.get("min_on", 0))
    onset_merge_gap = best_result.get("decoder_onset_merge_gap", onset_decoder.get("merge_gap", 0))
    onset_median = best_result.get("decoder_onset_median", onset_decoder.get("median", 1))
    offset_open_best = best_result.get("decoder_offset_open")
    offset_hold_best = best_result.get("decoder_offset_hold")
    offset_min_off = best_result.get("decoder_offset_min_off", offset_decoder.get("min_off", 0))
    offset_merge_gap = best_result.get("decoder_offset_merge_gap", offset_decoder.get("merge_gap", 0))
    offset_median = best_result.get("decoder_offset_median", offset_decoder.get("median", 1))
    if (
        onset_open_best is not None
        and onset_hold_best is not None
        and offset_open_best is not None
        and offset_hold_best is not None
    ):
        lines.append(
            "[best-decoder] onset_open={:.3f} hold={:.3f} min_on={} merge_gap={} | "
            "offset_open={:.3f} hold={:.3f} min_off={} merge_gap={}".format(
                onset_open_best,
                onset_hold_best,
                int(onset_min_on),
                int(onset_merge_gap),
                offset_open_best,
                offset_hold_best,
                int(offset_min_off),
                int(offset_merge_gap),
            )
        )
    temp_val = float(args.temperature) if args.temperature is not None else 1.0
    bias_val = float(args.bias) if args.bias is not None else 0.0
    temp_on_val = float(args.temperature_onset) if args.temperature_onset is not None else temp_val
    temp_off_val = float(args.temperature_offset) if args.temperature_offset is not None else temp_val
    bias_on_val = float(args.bias_onset) if args.bias_onset is not None else bias_val
    bias_off_val = float(args.bias_offset) if args.bias_offset is not None else bias_val
    compare_line = (
        "[compare-config] onset_thr_used={:.4f} offset_thr_used={:.4f} "
        "T_on={:.4f} b_on={:.4f} T_off={:.4f} b_off={:.4f} "
        "onset_gate(open={:.4f}, hold={:.4f}, min_on={}, gap={}, median={}) "
        "offset_gate(open={:.4f}, hold={:.4f}, min_off={}, gap={}, median={})"
    ).format(
        best_result["onset_thr"],
        best_result["offset_thr"],
        temp_on_val,
        bias_on_val,
        temp_off_val,
        bias_off_val,
        float(onset_open_best if onset_open_best is not None else onset_decoder.get("open", 0.0)),
        float(onset_hold_best if onset_hold_best is not None else onset_decoder.get("hold", 0.0)),
        int(onset_min_on),
        int(onset_merge_gap),
        int(onset_median),
        float(offset_open_best if offset_open_best is not None else offset_decoder.get("open", 0.0)),
        float(offset_hold_best if offset_hold_best is not None else offset_decoder.get("hold", 0.0)),
        int(offset_min_off),
        int(offset_merge_gap),
        int(offset_median),
    )
    lines.append(compare_line)
    return lines

def _prepare_logits_for_dump(tensor: torch.Tensor) -> np.ndarray:
    """Flatten a tensor to (T,P) and return a contiguous float64 numpy array."""

    if tensor is None:
        raise ValueError("Expected tensor, got None")
    if tensor.ndim < 2:
        raise ValueError(f"Logits tensor must have at least 2 dims, got {tensor.ndim}")

    tensor = tensor.contiguous()
    last_dim = tensor.shape[-1]
    tensor = tensor.reshape(-1, last_dim).contiguous()

    if tensor.ndim != 2:
        raise ValueError(f"Logits tensor reshape result must be 2D, got {tensor.ndim}D")
    if not tensor.is_contiguous():
        raise ValueError("Expected contiguous tensor after reshape")

    array = np.ascontiguousarray(tensor.numpy(), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"NumPy logits array must be 2D, got {array.ndim}D")
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected contiguous NumPy array for logits dump")
    return array


def _parse_list(argv, name):
    """Extract ``--<name>`` from ``argv`` allowing comma/space separation."""
    flag = f"--{name}"
    for i, arg in enumerate(list(argv)):
        if arg.startswith(flag):
            if arg == flag:
                j = i + 1
                vals = []
                while j < len(argv) and not argv[j].startswith("--"):
                    vals.append(argv[j])
                    j += 1
                if not vals:
                    raise ValueError(f"{flag} expects at least one value")
                del argv[i:j]
                arg_str = " ".join(vals)
            else:  # handle --flag=... form
                arg_str = arg.split("=", 1)[1]
                del argv[i]
            arg_str = arg_str.replace(",", " ")
            return [float(v) for v in arg_str.split() if v]
    return None


def _resolve_threshold_lists(onset_vals, offset_vals):
    """Return onset/offset lists, cloning inputs and reusing values when missing."""

    reuse_flags = {"onset_from_offset": False, "offset_from_onset": False}
    if onset_vals is None and offset_vals is None:
        return None, None, reuse_flags

    if onset_vals is None:
        onset_vals = list(offset_vals)
        reuse_flags["onset_from_offset"] = True
    else:
        onset_vals = list(onset_vals)

    if offset_vals is None:
        offset_vals = list(onset_vals)
        reuse_flags["offset_from_onset"] = True
    else:
        offset_vals = list(offset_vals)

    return onset_vals, offset_vals, reuse_flags


def _logit_to_probability(value: float) -> float:
    """Convert an arbitrary logit to its probability-domain equivalent."""

    val = float(value)
    if math.isnan(val):
        return 0.5
    if val >= 0.0:
        z = math.exp(-val)
        prob = 1.0 / (1.0 + z)
    else:
        z = math.exp(val)
        prob = z / (1.0 + z)
    if not math.isfinite(prob):
        return 0.5
    return max(0.0, min(prob, 1.0))


def _binary_f1(pred, target, eps=1e-8):
    """Binary F1 score for tensors in {0,1}."""
    if target.sum().item() == 0 and pred.sum().item() == 0:
        return 0.0
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)
   
   
def _event_f1(pred, target, hop_seconds: float, tol_sec: float, eps=1e-8):
    """Event-level F1 score with time tolerance."""
    pred_pos = pred.nonzero(as_tuple=False)
    true_pos = target.nonzero(as_tuple=False)
    if pred_pos.numel() == 0 and true_pos.numel() == 0:
        return None

    # ``frame_to_sec`` returns ``TensorLike`` which can be ``int`` from Pylance's
    # perspective.  Explicitly convert to tensors so static analyzers know these
    # support indexing and broadcasting.
    pred_times = torch.as_tensor(frame_to_sec(pred_pos[:, 0], hop_seconds))
    true_times = torch.as_tensor(frame_to_sec(true_pos[:, 0], hop_seconds))
    pred_pitch = pred_pos[:, 1]
    true_pitch = true_pos[:, 1]

    used = torch.zeros(true_pos.shape[0], dtype=torch.bool)
    tp = 0
    for i in range(pred_pos.shape[0]):
        p = pred_pitch[i]
        t = pred_times[i]
        mask = (true_pitch == p) & (~used)
        if mask.any():
            cand_idx = torch.where(mask)[0]
            diffs = torch.abs(true_times[cand_idx] - t)
            min_diff, j = torch.min(diffs, dim=0)
            if min_diff.item() <= tol_sec:
                tp += 1
                used[cand_idx[j]] = True
    fp = pred_pos.shape[0] - tp
    fn = true_pos.shape[0] - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)


def _compute_ece(probs: torch.Tensor, targets: torch.Tensor, bins: int = 15) -> float:
    if probs.numel() == 0:
        return 0.0
    probs_flat = probs.reshape(-1).detach().cpu().numpy()
    targets_flat = targets.reshape(-1).detach().cpu().numpy()
    if probs_flat.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = probs_flat.size
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (probs_flat >= low) & (probs_flat <= high)
        else:
            mask = (probs_flat >= low) & (probs_flat < high)
        count = mask.sum()
        if count == 0:
            continue
        prob_mean = probs_flat[mask].mean()
        true_mean = targets_flat[mask].mean()
        ece += abs(prob_mean - true_mean) * (count / total)
    return float(ece)


def _collect_onset_metrics(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    hop_seconds: float,
    loose_tol: float,
    probs: torch.Tensor,
) -> dict:
    strict_tol = max(loose_tol * 0.5, hop_seconds / 2.0)
    frame_f1 = _binary_f1(pred_mask.reshape(-1).float(), true_mask.reshape(-1).float()) or 0.0
    ev_loose = _event_f1(pred_mask, true_mask, hop_seconds, loose_tol) or 0.0
    ev_strict = _event_f1(pred_mask, true_mask, hop_seconds, strict_tol) or 0.0
    return {
        "frame_f1": float(frame_f1),
        "event_f1_loose": float(ev_loose),
        "event_f1_strict": float(ev_strict),
        "pred_rate": float(pred_mask.float().mean().item()),
        "pos_rate": float(true_mask.float().mean().item()),
        "ece": _compute_ece(probs, true_mask),
        "tolerance_loose": float(loose_tol),
        "tolerance_strict": float(strict_tol),
    }


def main():
    argv = sys.argv[1:]
    t_main_start = time.time()
    (
        args,
        debug_mode,
        onset_temperature,
        offset_temperature,
        onset_bias,
        offset_bias,
        onset_platt_stats,
        offset_platt_stats,
        prob_reuse,
        logit_reuse,
    ) = _parse_cli_args(argv)

    stage_durations: Dict[str, float] = {}
    runtime = _prepare_runtime(args, debug_mode, stage_durations)
    log_handle, _log_progress = _init_progress_logger(args)
    _ = log_handle  # handle kept alive for atexit cleanup

    onset_probs_final = list(args.prob_thresholds) if args.prob_thresholds is not None else []
    offset_probs_final = list(args.offset_prob_thresholds) if args.offset_prob_thresholds is not None else []
    if args.grid_prob_thresholds:
        combos = len(onset_probs_final) * len(offset_probs_final)
    else:
        combos = max(len(onset_probs_final), len(offset_probs_final))
    onset_display = "[" + ",".join(f"{p:.4f}" for p in onset_probs_final) + "]"
    offset_display = "[" + ",".join(f"{p:.4f}" for p in offset_probs_final) + "]"
    LOGGER.info(
        "[grid] onset_probs=%s offset_probs=%s (final) combos=%d",
        onset_display,
        offset_display,
        combos,
        extra=QUIET_EXTRA,
    )

    if args.head is None and args.prob_thresholds is not None and prob_reuse.get("offset_from_onset"):
        print("[eval] offset probability thresholds not provided; reusing onset list", flush=True)
    if args.head is None and args.thresholds is not None and logit_reuse.get("offset_from_onset"):
        print("[eval] offset logit thresholds not provided; reusing onset list", flush=True)

    loader_ctx = _build_eval_loader(args, runtime, _log_progress)
    if loader_ctx.target_clips == 0:
        _log_progress("[progress] target_clips resolved to 0; exiting early.", force=True)
        return

    model = _setup_model(args, runtime.cfg)

    eval_results = _collect_logits(
        model=model,
        loader_ctx=loader_ctx,
        args=args,
        runtime=runtime,
        onset_platt_stats=onset_platt_stats,
        offset_platt_stats=offset_platt_stats,
        onset_temperature=onset_temperature,
        offset_temperature=offset_temperature,
        onset_bias=onset_bias,
        offset_bias=offset_bias,
        log_progress=_log_progress,
    )

    _summarize_data_pass(eval_results, runtime, loader_ctx.target_clips, _log_progress)

    if not eval_results.onset_logits_list:
        _log_progress("[progress] no valid clips processed; aborting.", force=True)
        print("error: no valid clips processed; aborting.", file=sys.stderr)
        return

    onset_logits_list = eval_results.onset_logits_list
    offset_logits_list = eval_results.offset_logits_list
    pitch_logits_list = eval_results.pitch_logits_list
    onset_probs = eval_results.onset_probs
    offset_probs = eval_results.offset_probs
    onset_tgts = eval_results.onset_tgts
    offset_tgts = eval_results.offset_tgts
    skip_paths = eval_results.skip_paths
    bad_clip_counts = eval_results.bad_clip_counts
    lag_ms_samples = eval_results.lag_ms_samples
    lag_source_counter = eval_results.lag_source_counter
    skipped_batches = eval_results.skipped_batches

    cfg = runtime.cfg
    dataset_cfg = runtime.dataset_cfg
    model_cfg = runtime.model_cfg
    decode_fps = runtime.decode_fps
    hop_seconds = runtime.hop_seconds
    event_tolerance = runtime.event_tolerance
    key_prior_midi_low = runtime.key_prior_midi_low
    key_prior_settings = runtime.key_prior_settings
    agg_mode = runtime.agg_mode
    agg_top_k = runtime.agg_top_k
    agg_tau_sum = runtime.agg_tau_sum
    default_k_onset = runtime.default_k_onset
    default_k_offset = runtime.default_k_offset
    include_k_column = runtime.include_k_column
    k_candidates = runtime.k_candidates
    preview_prob_threshold = runtime.preview_prob_threshold
    avlag_disabled = runtime.avlag_disabled
    return_per_tile_requested = runtime.return_per_tile_requested
    split = runtime.split
    stage_durations = runtime.stage_durations
    # Unless a calibration file is provided and no head is specified, default to
    # sweeping over probability thresholds when none were specified explicitly.
    if args.thresholds is None and args.prob_thresholds is None:
        if args.head is not None or not args.calibration:
            args.prob_thresholds = DEFAULT_THRESHOLDS.copy()


    cfg = dict(load_config("configs/config_pianovam_fast.yaml"))
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        cfg["model"] = model_cfg
    if args.model_return_per_tile is not None:
        model_cfg["return_per_tile"] = bool(args.model_return_per_tile)
    return_per_tile_requested = bool(model_cfg.get("return_per_tile"))

    seed = resolve_seed(args.seed, cfg)
    deterministic = resolve_deterministic_flag(args.deterministic, cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["deterministic"] = deterministic
    configure_determinism(seed, deterministic)
    print(
        f"[determinism] seed={seed} deterministic={'on' if deterministic else 'off'}",
        flush=True,
    )
    dataset_raw = cfg.get("dataset")
    if isinstance(dataset_raw, dict):
        dataset_cfg = dict(dataset_raw)
    else:
        dataset_cfg = {}
    cfg["dataset"] = dataset_cfg
    try:
        model_tiles = int(model_cfg.get("tiles", dataset_cfg.get("tiles", 3)))
    except Exception:
        model_tiles = 3
    if args.max_clips is not None:
        dataset_cfg["max_clips"] = args.max_clips
    if args.frames is not None:
        dataset_cfg["frames"] = args.frames
    only_id = canonical_video_id(args.only) if args.only else None
    if only_id:
        dataset_cfg["only_video"] = only_id
    env_disable = str(os.environ.get("AVSYNC_DISABLE", "")).strip().lower()
    avlag_disabled = bool(args.no_avlag) or env_disable in {"1", "true", "yes", "on"}
    if avlag_disabled:
        dataset_cfg["avlag_disabled"] = True
    if debug_mode:
        dataset_cfg["num_workers"] = 0
        dataset_cfg["persistent_workers"] = False
        dataset_cfg["pin_memory"] = False
        print("[debug] num_workers=0, persistent_workers=False, pin_memory=False", flush=True)
    decode_fps = float(dataset_cfg.get("decode_fps", 0.0) or 0.0)
    hop_seconds = float(dataset_cfg.get("hop_seconds", 0.0) or 0.0)
    if hop_seconds <= 0.0 and decode_fps > 0.0:
        hop_seconds = 1.0 / decode_fps
    if decode_fps <= 0.0 and hop_seconds > 0.0:
        decode_fps = 1.0 / hop_seconds
    if decode_fps <= 0.0:
        decode_fps = 30.0
    if hop_seconds <= 0.0:
        hop_seconds = 1.0 / decode_fps
    frame_targets_cfg = dataset_cfg.get("frame_targets", {}) or {}
    event_tolerance = float(frame_targets_cfg.get("tolerance", hop_seconds))
    midi_low_cfg = frame_targets_cfg.get("note_min")
    key_prior_midi_low = int(midi_low_cfg) if isinstance(midi_low_cfg, (int, float)) else 21
    split = args.split or dataset_cfg.get("split_val") or dataset_cfg.get("split") or "valid"

    frames_display = dataset_cfg.get("frames")
    max_clips_display = dataset_cfg.get("max_clips")
    only_display = only_id or "-"
    frame_text = frames_display if frames_display is not None else "?"
    max_clips_text = max_clips_display if max_clips_display is not None else "?"
    print(
        f"[progress] starting (split={split}, frames={frame_text}, max_clips={max_clips_text}, only={only_display})",
        flush=True,
    )

    metrics_cfg = cfg.get("training", {}).get("metrics", {}) or {}
    agg_cfg = metrics_cfg.get("aggregation", {}) or {}
    sweep_cfg = metrics_cfg.get("sweep", {}) or {}
    floor_band_raw = sweep_cfg.get("floor_band", [0.20, 0.30, 0.40])
    floor_band: List[float] = []
    if isinstance(floor_band_raw, (list, tuple)):
        for item in floor_band_raw:
            try:
                val = float(item)
            except (TypeError, ValueError):
                continue
            if 0.0 <= val <= 1.0:
                floor_band.append(val)
    else:
        try:
            val = float(floor_band_raw)
        except (TypeError, ValueError):
            val = None
        if val is not None and 0.0 <= val <= 1.0:
            floor_band.append(val)
    if not floor_band:
        floor_band = [0.20, 0.30, 0.40]

    calibration_data = None
    if args.calibration:
        with open(args.calibration) as f:
            calibration_data = json.load(f)

    k_candidates = runtime.k_candidates
    per_head_sweep_vals = None
    per_head_use_logits = False
    per_head_mode = "prob"
    if args.head is not None:
        if args.head == "onset":
            if args.thresholds is not None:
                per_head_sweep_vals = args.thresholds
                per_head_use_logits = True
                per_head_mode = "logit"
            else:
                per_head_sweep_vals = args.prob_thresholds
                per_head_use_logits = False
                per_head_mode = "prob"
        else:
            if args.offset_thresholds is not None:
                per_head_sweep_vals = args.offset_thresholds
                per_head_use_logits = True
                per_head_mode = "logit"
            else:
                per_head_sweep_vals = args.offset_prob_thresholds
                per_head_use_logits = False
                per_head_mode = "prob"

        if per_head_sweep_vals is None:
            per_head_sweep_vals = args.thresholds if args.thresholds is not None else args.prob_thresholds
            per_head_use_logits = args.thresholds is not None
            per_head_mode = "logit" if per_head_use_logits else "prob"

    calib_pairs: int = 0
    logit_pairs: int = 0
    prob_pairs: int = 0
    num_prob_combos: int = 0

    if args.head is None:
        calib_pairs = 0
        if calibration_data:
            on_cal = calibration_data.get("onset", {})
            off_cal = calibration_data.get("offset", {})
            if "best_logit" in on_cal and "best_logit" in off_cal:
                calib_pairs = 1
            elif "best_prob" in on_cal and "best_prob" in off_cal:
                calib_pairs = 1
        logit_pairs = len(args.thresholds) if args.thresholds else 0
        onset_prob_list = args.prob_thresholds or []
        offset_prob_list = args.offset_prob_thresholds or onset_prob_list
        if args.grid_prob_thresholds:
            prob_pairs = len(onset_prob_list) * len(offset_prob_list)
        else:
            prob_pairs = len(onset_prob_list)
        num_prob_combos = prob_pairs * len(k_candidates)
        num_thr_pairs = calib_pairs + logit_pairs + prob_pairs
        num_combos = calib_pairs + logit_pairs + num_prob_combos
        num_k = len(k_candidates) if prob_pairs > 0 and len(k_candidates) > 1 else 1
        thr_parts = []
        if logit_pairs:
            thr_parts.append(f"logit:{logit_pairs}")
        if prob_pairs:
            if args.grid_prob_thresholds and onset_prob_list and offset_prob_list:
                thr_parts.append(f"prob_grid:{len(onset_prob_list)}x{len(offset_prob_list)}")
            else:
                thr_parts.append(f"prob_pairs:{prob_pairs}")
        if calib_pairs:
            thr_parts.append(f"calib:{calib_pairs}")
        if not thr_parts:
            thr_parts.append("none")
        thr_desc = ",".join(thr_parts)
        k_sweep_state = "on" if len(k_candidates) > 1 else "off"
    else:
        sweep_len = len(per_head_sweep_vals) if per_head_sweep_vals is not None else 0
        num_thr_pairs = sweep_len
        num_combos = sweep_len
        num_k = 1
        thr_desc = str(sweep_len)
        k_sweep_state = "off"

    target_display = loader_ctx.target_display
    if args.postproc_mode == "eval_only" and num_combos:
        print("POSTPROC: SKIPPED during sweeps (mode=eval_only)")
    _log_progress(
        f"[progress] starting: clips={target_display} combos={num_combos} (thr={thr_desc}, k_sweep={k_sweep_state})",
        force=True,
    )


    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    pitch_logits = torch.cat(pitch_logits_list, dim=0) if pitch_logits_list else None
    onset_probs = torch.cat(onset_probs, dim=0)
    offset_probs = torch.cat(offset_probs, dim=0)
    onset_tgts = torch.cat(onset_tgts, dim=0)
    offset_tgts = torch.cat(offset_tgts, dim=0)

    _validate_global_pair(onset_logits, onset_tgts, label="onset")
    _validate_global_pair(offset_logits, offset_tgts, label="offset")

    T_logits, P_logits = onset_probs.shape[1], onset_probs.shape[2]
    if onset_tgts.shape[1] != T_logits:
        onset_tgts = pool_roll_BT(onset_tgts, T_logits)
        offset_tgts = pool_roll_BT(offset_tgts, T_logits)
    onset_tgts = align_pitch_dim(onset_probs, onset_tgts, "onset")
    offset_tgts = align_pitch_dim(offset_probs, offset_tgts, "offset")
    
    if debug_mode:
        print(
            "[DEBUG] aligned shapes logits=",
            onset_logits.shape,
            "targets=",
            onset_tgts.shape,
        )
        print(
            "[DEBUG] targets nonzero onset=",
            int(onset_tgts.sum().item()),
            "offset=",
            int(offset_tgts.sum().item()),
        )
        diff = (torch.sigmoid(onset_logits) - onset_probs).abs().max().item()
        print(f"[DEBUG] sigmoid max abs diff={diff:.3e}")
    
    dump_path = Path(args.dump_logits).expanduser() if args.dump_logits else None
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)

        dump_arrays = {}
        onset_np = _prepare_logits_for_dump(onset_logits)
        offset_np = _prepare_logits_for_dump(offset_logits)
        dump_arrays["onset_logits"] = onset_np
        dump_arrays["offset_logits"] = offset_np

        if pitch_logits is not None:
            dump_arrays["pitch_logits"] = _prepare_logits_for_dump(pitch_logits)

        pitch_bins = next((arr.shape[1] for arr in dump_arrays.values() if arr is not None), None)
        frame_cfg = cfg.get("dataset", {}).get("frame_targets", {}) or {}
        midi_low = int(frame_cfg.get("note_min", 21))
        midi_high_cfg = frame_cfg.get("note_max")
        midi_high = int(midi_high_cfg) if midi_high_cfg is not None else midi_low
        if pitch_bins is not None:
            if midi_high - midi_low + 1 != pitch_bins:
                midi_low = 21
                midi_high = midi_low + pitch_bins - 1
            else:
                midi_high = midi_low + pitch_bins - 1

        meta = {
            "fps": decode_fps,
            "midi_low": midi_low,
            "midi_high": midi_high,
        }
        dump_arrays["meta"] = json.dumps(meta, sort_keys=True)

        np.savez_compressed(dump_path, **dump_arrays)
        print(f"[eval] dumped logits -> {dump_path}")
        
    # diagnostic prints
    print(f"[OVERALL onset probs] mean={onset_probs.mean():.3f} min={onset_probs.min():.3f} max={onset_probs.max():.3f}")
    print(f"[OVERALL offset probs] mean={offset_probs.mean():.3f} min={offset_probs.min():.3f} max={offset_probs.max():.3f}")
    _report_platt_stats("onset", onset_platt_stats)
    _report_platt_stats("offset", offset_platt_stats)

    _assert_monotonic_rates(
        onset_probs,
        label="onset",
        agg_mode=agg_mode,
        cap_count=default_k_onset,
        top_k=agg_top_k,
    )
    _assert_monotonic_rates(
        offset_probs,
        label="offset",
        agg_mode=agg_mode,
        cap_count=default_k_offset,
        top_k=agg_top_k,
    )

    offset_lower_hint = 0.10  # base guardrail

    onset_max_prob, onset_stats = _summarize_probs("onset", onset_probs, _log_progress)
    offset_max_prob, offset_stats = _summarize_probs("offset", offset_probs, _log_progress)
    onset_peak = max(onset_max_prob, float(onset_stats.get(0.99, onset_max_prob)))
    offset_peak = max(offset_max_prob, float(offset_stats.get(0.99, offset_max_prob)))
    onset_lower_hint = float(max(0.0, min(onset_peak - 0.10, 0.95)))
    offset_lower_hint = float(max(0.0, min(offset_peak - 0.10, 0.95)))

    if args.head is None:
        if args.prob_thresholds:
            using_grid = bool(args.grid_prob_thresholds)
            onset_list = list(args.prob_thresholds)
            if args.offset_prob_thresholds is not None:
                offset_list = list(args.offset_prob_thresholds)
            else:
                offset_list = list(onset_list)

            lowest_onset_thr = min(onset_list) if onset_list else None
            lowest_offset_thr = min(offset_list) if offset_list else None
            onset_extend_needed = (
                lowest_onset_thr is not None
                and onset_peak + 1e-9 < lowest_onset_thr
            )
            offset_extend_needed = (
                lowest_offset_thr is not None
                and offset_peak + 1e-9 < lowest_offset_thr
            )
            inserted_lower_onset = None
            inserted_lower_offset = None

            if using_grid:
                onset_set = set(onset_list)
                offset_set = set(offset_list)
                onset_set.update(floor_band)
                offset_set.update(floor_band)
                if onset_extend_needed:
                    onset_set.add(onset_lower_hint)
                    inserted_lower_onset = onset_lower_hint
                if offset_extend_needed:
                    offset_set.add(offset_lower_hint)
                    inserted_lower_offset = offset_lower_hint
                args.prob_thresholds = sorted(onset_set)
                args.offset_prob_thresholds = sorted(offset_set)
            else:
                pairs = [(float(o), float(off)) for o, off in zip(onset_list, offset_list)]
                pair_map = {(round(o, 6), round(off, 6)): (o, off) for o, off in pairs}

                def _add_pair(on_val: float, off_val: float) -> None:
                    key = (round(on_val, 6), round(off_val, 6))
                    if key not in pair_map:
                        pair_map[key] = (float(on_val), float(off_val))

                for val in floor_band:
                    _add_pair(val, val)
                if onset_extend_needed:
                    _add_pair(onset_lower_hint, onset_lower_hint)
                    inserted_lower_onset = onset_lower_hint
                if offset_extend_needed:
                    _add_pair(offset_lower_hint, offset_lower_hint)
                    inserted_lower_offset = offset_lower_hint

                pairs = list(pair_map.values())
                pairs.sort(key=lambda item: (item[1], item[0]))
                args.prob_thresholds = [p[0] for p in pairs]
                args.offset_prob_thresholds = [p[1] for p in pairs]

            onset_list = list(args.prob_thresholds)
            offset_list = (
                list(args.offset_prob_thresholds)
                if args.offset_prob_thresholds is not None
                else list(onset_list)
            )
            if using_grid:
                prob_pairs = len(onset_list) * len(offset_list)
            else:
                prob_pairs = len(onset_list)
            num_prob_combos = prob_pairs * len(k_candidates)
            num_thr_pairs = calib_pairs + logit_pairs + prob_pairs
            num_combos = calib_pairs + logit_pairs + num_prob_combos

            if inserted_lower_onset is not None:
                _log_progress(
                    "[sweep] onset peak prob %.4f < min sweep %.4f → added %.4f to sweep list."
                    % (onset_peak, lowest_onset_thr, inserted_lower_onset),
                    force=True,
                )
            if inserted_lower_offset is not None:
                _log_progress(
                    "[sweep] offset peak prob %.4f < min sweep %.4f → added %.4f to sweep list."
                    % (offset_peak, lowest_offset_thr, inserted_lower_offset),
                    force=True,
                )
            _log_progress(
                f"[sweep] ensured floor band {', '.join(f'{v:.2f}' for v in sorted(floor_band))} in probability sweep.",
                force=True,
            )
            if using_grid:
                prob_desc = f"prob_grid:{len(onset_list)}x{len(offset_list)}"
            else:
                prob_desc = f"prob_pairs:{prob_pairs}"
            thr_desc = []
            if logit_pairs:
                thr_desc.append(f"logit:{logit_pairs}")
            thr_desc.append(prob_desc)
            if calib_pairs:
                thr_desc.append(f"calib:{calib_pairs}")
            _log_progress(
                f"[sweep] updated probability grids → onset={_format_float_list(onset_list)} "
                f"offset={_format_float_list(offset_list)} combos={num_combos}",
                force=True,
            )
    else:
        if per_head_mode == "prob" and per_head_sweep_vals is not None:
            values = set(float(v) for v in per_head_sweep_vals)
            values.update(floor_band)
            inserted_lower = None
            if args.head == "offset":
                lowest = min(values) if values else None
                extend_needed = lowest is not None and offset_peak + 1e-9 < lowest
                if extend_needed:
                    values.add(offset_lower_hint)
                    inserted_lower = offset_lower_hint
            elif args.head == "onset":
                lowest = min(values) if values else None
                extend_needed = lowest is not None and onset_peak + 1e-9 < lowest
                if extend_needed:
                    values.add(onset_lower_hint)
                    inserted_lower = onset_lower_hint
            per_head_sweep_vals = sorted(values)
            num_combos = len(per_head_sweep_vals)
            if inserted_lower is not None:
                peak_val = offset_peak if args.head == "offset" else onset_peak
                _log_progress(
                    "[sweep] per-head %s peak prob %.4f → added %.4f to sweep list."
                    % (args.head, peak_val, inserted_lower),
                    force=True,
                )
            _log_progress(
                f"[sweep] per-head sweep ({args.head}) values={_format_float_list(per_head_sweep_vals)} combos={num_combos}",
                force=True,
            )

    # Use all key/time positions rather than collapsing with ``any``.
    # Collapsing across the note dimension causes the predicted rate to be
    # either 0 or 1 for a clip, which in turn makes F1-threshold sweeps
    # uninformative.  Instead we compute metrics over the full pianoroll so
    # that the positive rate varies smoothly with the threshold.
    onset_true_bin = (onset_tgts > 0).float()
    offset_true_bin = (offset_tgts > 0).float()

    decoder_params = resolve_decoder_from_config(metrics_cfg)
    if args.low_ratio is not None:
        decoder_params["onset"]["low_ratio"] = float(args.low_ratio)
        decoder_params["offset"]["low_ratio"] = float(args.low_ratio)
    if args.min_on is not None:
        decoder_params["onset"]["min_on"] = int(args.min_on)
        decoder_params["offset"]["min_on"] = int(args.min_on)
    if args.min_off is not None:
        decoder_params["onset"]["min_off"] = int(args.min_off)
        decoder_params["offset"]["min_off"] = int(args.min_off)
    if args.gap_merge is not None:
        decoder_params["onset"]["merge_gap"] = int(args.gap_merge)
        decoder_params["offset"]["merge_gap"] = int(args.gap_merge)
    if args.median is not None:
        decoder_params["onset"]["median"] = int(args.median)
        decoder_params["offset"]["median"] = int(args.median)
    if args.decoder_onset_open is not None:
        decoder_params["onset"]["open"] = float(args.decoder_onset_open)
        decoder_params["onset"]["open_defined"] = True
    if args.decoder_onset_hold is not None:
        decoder_params["onset"]["hold"] = float(args.decoder_onset_hold)
        decoder_params["onset"]["hold_defined"] = True
    if args.decoder_onset_min_on is not None:
        decoder_params["onset"]["min_on"] = int(args.decoder_onset_min_on)
    if args.decoder_onset_merge_gap is not None:
        decoder_params["onset"]["merge_gap"] = int(args.decoder_onset_merge_gap)
    if args.decoder_offset_open is not None:
        decoder_params["offset"]["open"] = float(args.decoder_offset_open)
        decoder_params["offset"]["open_defined"] = True
    if args.decoder_offset_hold is not None:
        decoder_params["offset"]["hold"] = float(args.decoder_offset_hold)
        decoder_params["offset"]["hold_defined"] = True
    if args.decoder_offset_min_off is not None:
        decoder_params["offset"]["min_off"] = int(args.decoder_offset_min_off)
    if args.decoder_offset_merge_gap is not None:
        decoder_params["offset"]["merge_gap"] = int(args.decoder_offset_merge_gap)
    decoder_params = normalize_decoder_params(decoder_params)
    onset_decoder = decoder_params["onset"]
    offset_decoder = decoder_params["offset"]
    decoder_choice = args.decoder or "auto"
    if decoder_choice == "none":
        print("[decoder] requested decoder=none -> forcing hysteresis with config defaults", flush=True)
    decoder_kind = "hysteresis" if decoder_choice in {"auto", "none"} else decoder_choice
    decoder_settings_summary = format_decoder_settings(decoder_kind, decoder_params)
    print(f"[decoder-settings] {decoder_settings_summary}")
    decoder_post_cfg = cfg.get("decoder", {}).get("post", {}) or {}
    snap_enabled_cfg = bool(decoder_post_cfg.get("snap", {}).get("enabled", False))
    dp_enabled_cfg = bool(decoder_post_cfg.get("dp", {}).get("enabled", False))
    postproc_modules: List[str] = []
    if snap_enabled_cfg:
        postproc_modules.append("snap")
    if dp_enabled_cfg:
        postproc_modules.append("dp")
    postproc_debug = bool(cfg.get("logging", {}).get("postproc_debug", False))
    _SWEEP_POSTPROC_MAP = {"never": False, "eval_only": False, "always": True}
    sweep_apply_postproc = _SWEEP_POSTPROC_MAP[args.postproc_mode]
    final_apply_postproc = args.postproc_mode in ("eval_only", "always")
    post_logs_dir = Path(cfg.get("logging", {}).get("log_dir", "logs") or "logs") / "post"
    row_records: List[Dict[str, Any]] = []

    eval_ctx = EvalPairContext(
        agg_mode=agg_mode,
        agg_top_k=agg_top_k,
        default_k_onset=default_k_onset,
        default_k_offset=default_k_offset,
        hop_seconds=hop_seconds,
        decode_fps=decode_fps,
        event_tolerance=event_tolerance,
        onset_temperature=onset_temperature,
        offset_temperature=offset_temperature,
        onset_bias=onset_bias,
        offset_bias=offset_bias,
        onset_probs=torch.as_tensor(onset_probs),
        offset_probs=torch.as_tensor(offset_probs),
        onset_true_bin=onset_true_bin,
        offset_true_bin=offset_true_bin,
        decoder_kind=decoder_kind,
        decoder_params=decoder_params,
        onset_decoder=onset_decoder,
        offset_decoder=offset_decoder,
        postproc_modules=tuple(postproc_modules),
        postproc_debug=postproc_debug,
        cfg=cfg,
    )

    printed_header = False

    sanity_requested = bool(args.sanity_thresholds)
    eval_cache_enabled = not args.no_eval_cache and not sanity_requested
    if sanity_requested and not args.no_eval_cache:
        print("[cache] disabled because --sanity_thresholds requires fresh evaluation", flush=True)
    cache_db: Optional[Dict[str, Any]] = None
    cache_path: Optional[Path] = None
    cache_key: Optional[str] = None
    cache_fingerprint: Optional[Dict[str, Any]] = None
    if eval_cache_enabled:
        cache_dir = Path(cfg.get("logging", {}).get("log_dir", "logs") or "logs")
        cache_path = cache_dir / EVAL_CACHE_FILENAME
        fingerprint_payload = {
            "ckpt": str(Path(args.ckpt).expanduser().resolve()),
            "split": split,
            "frames_cfg": dataset_cfg.get("frames") if isinstance(dataset_cfg, Mapping) else None,
            "frames_arg": args.frames,
            "max_clips_cfg": dataset_cfg.get("max_clips") if isinstance(dataset_cfg, Mapping) else None,
            "max_clips_arg": args.max_clips,
            "prob_thresholds": _normalize_threshold_list(args.prob_thresholds),
            "offset_prob_thresholds": _normalize_threshold_list(args.offset_prob_thresholds),
            "logit_thresholds": _normalize_threshold_list(args.thresholds),
            "offset_logit_thresholds": _normalize_threshold_list(args.offset_thresholds),
            "grid_prob_thresholds": bool(args.grid_prob_thresholds),
            "k_candidates": [int(k) for k in k_candidates],
            "default_k": {"onset": int(default_k_onset), "offset": int(default_k_offset)},
            "agg_mode": agg_mode,
            "agg_top_k": int(agg_top_k),
            "agg_tau_sum": round(float(agg_tau_sum), 6),
            "temperature": round(float(args.temperature), 6) if args.temperature is not None else 1.0,
            "bias": round(float(args.bias), 6) if args.bias is not None else 0.0,
            "temperature_onset": round(float(args.temperature_onset), 6) if args.temperature_onset is not None else None,
            "temperature_offset": round(float(args.temperature_offset), 6) if args.temperature_offset is not None else None,
            "bias_onset": round(float(args.bias_onset), 6) if args.bias_onset is not None else None,
            "bias_offset": round(float(args.bias_offset), 6) if args.bias_offset is not None else None,
            "decoder": _snapshot_decoder_gates(decoder_params),
            "postproc_mode": args.postproc_mode,
            "postproc_modules": postproc_modules,
            "head": args.head,
            "sweep_k_onset": bool(args.sweep_k_onset),
            "only": only_id,
            "calibration": args.calibration,
            "no_avlag": bool(args.no_avlag),
            "sanity_thresholds": _normalize_threshold_list(args.sanity_thresholds),
        }
        cache_fingerprint = cast(Dict[str, Any], _json_sanitize(fingerprint_payload))
        cache_key = _hash_cache_fingerprint(fingerprint_payload)
        cache_db = _load_eval_cache_db(cache_path)
        entries = cache_db.get("entries", {})
        cached_entry = entries.get(cache_key)
        if cached_entry and cached_entry.get("fingerprint") == cache_fingerprint:
            printed_header = replay_cache_entry(
                cached_entry,
                include_k_column=include_k_column,
                printed_header=printed_header,
            )
            return

    best_result = None
    best_post_stats = None
    total_evals = 0
    summary_lines: List[str] = []

    combo_state = {
        "combo_idx": 0,
        "num_combos": num_combos,
        "start_time": time.time(),
        "last_grid_print": time.time(),
    }
    _log_progress(f"[progress] grid sweep start: combos={num_combos}", force=True)

    if args.head is None:
        # Evaluate at calibrated thresholds if provided.
        if calibration_data:
            on_cal = calibration_data.get("onset", {})
            off_cal = calibration_data.get("offset", {})
            if "best_logit" in on_cal and "best_logit" in off_cal:
                printed_header = print_sweep_header(include_k_column, printed_header)
                best_result, best_post_stats, total_evals = run_eval_combo(
                    on_cal["best_logit"],
                    off_cal["best_logit"],
                    True,
                    k_onset=default_k_onset,
                    apply_postproc=sweep_apply_postproc,
                    eval_ctx=eval_ctx,
                    include_k_column=include_k_column,
                    row_records=row_records,
                    best_result=best_result,
                    best_post_stats=best_post_stats,
                    total_evals=total_evals,
                    combo_state=combo_state,
                    args=args,
                    log_progress=_log_progress,
                )
            elif "best_prob" in on_cal and "best_prob" in off_cal:
                printed_header = print_sweep_header(include_k_column, printed_header)
                best_result, best_post_stats, total_evals = run_eval_combo(
                    on_cal["best_prob"],
                    off_cal["best_prob"],
                    False,
                    k_onset=default_k_onset,
                    apply_postproc=sweep_apply_postproc,
                    eval_ctx=eval_ctx,
                    include_k_column=include_k_column,
                    row_records=row_records,
                    best_result=best_result,
                    best_post_stats=best_post_stats,
                    total_evals=total_evals,
                    combo_state=combo_state,
                    args=args,
                    log_progress=_log_progress,
                )
            else:
                print("Calibration file missing best_logit/best_prob keys", file=sys.stderr)

        # Sweep over provided threshold grids.
        if args.thresholds:
            printed_header = print_sweep_header(include_k_column, printed_header)
            offset_list = args.offset_thresholds if args.offset_thresholds else args.thresholds
            if len(offset_list) != len(args.thresholds):
                print("error: offset logit threshold count must match onset count", file=sys.stderr)
                return
            for on_thr, off_thr in zip(args.thresholds, offset_list):
                best_result, best_post_stats, total_evals = run_eval_combo(
                    on_thr,
                    off_thr,
                    True,
                    k_onset=default_k_onset,
                    apply_postproc=sweep_apply_postproc,
                    eval_ctx=eval_ctx,
                    include_k_column=include_k_column,
                    row_records=row_records,
                    best_result=best_result,
                    best_post_stats=best_post_stats,
                    total_evals=total_evals,
                    combo_state=combo_state,
                    args=args,
                    log_progress=_log_progress,
                )
        if args.prob_thresholds:
            printed_header = print_sweep_header(include_k_column, printed_header)
            onset_list = args.prob_thresholds
            offset_list = args.offset_prob_thresholds if args.offset_prob_thresholds else onset_list
            for k_val in k_candidates:
                if args.grid_prob_thresholds:
                    for on_thr in onset_list:
                        for off_thr in offset_list:
                            best_result, best_post_stats, total_evals = run_eval_combo(
                                on_thr,
                                off_thr,
                                False,
                                k_onset=k_val,
                                apply_postproc=sweep_apply_postproc,
                                eval_ctx=eval_ctx,
                                include_k_column=include_k_column,
                                row_records=row_records,
                                best_result=best_result,
                                best_post_stats=best_post_stats,
                                total_evals=total_evals,
                                combo_state=combo_state,
                                args=args,
                                log_progress=_log_progress,
                            )
                else:
                    if len(onset_list) != len(offset_list):
                        print("error: offset probability thresholds must match onset count", file=sys.stderr)
                        return
                    for idx, on_thr in enumerate(onset_list):
                        off_thr = offset_list[idx]
                        best_result, best_post_stats, total_evals = run_eval_combo(
                            on_thr,
                            off_thr,
                            False,
                            k_onset=k_val,
                            apply_postproc=sweep_apply_postproc,
                            eval_ctx=eval_ctx,
                            include_k_column=include_k_column,
                            row_records=row_records,
                            best_result=best_result,
                            best_post_stats=best_post_stats,
                            total_evals=total_evals,
                            combo_state=combo_state,
                            args=args,
                            log_progress=_log_progress,
                        )
    else:
        # Per-head sweep
        sweep_vals = per_head_sweep_vals
        use_logits = per_head_use_logits
        mode = per_head_mode

        if sweep_vals is None:
            print(
                "error: specify --thresholds or --prob_thresholds for per-head sweep",
                file=sys.stderr,
            )
            return
        
        other_head = "offset" if args.head == "onset" else "onset"
        fixed_thr = None
        source = None

        if calibration_data:
            other_cal = calibration_data.get(other_head, {})
            if use_logits:
                if "best_logit" in other_cal:
                    fixed_thr = other_cal["best_logit"]
                elif "best_prob" in other_cal:
                    fixed_thr = torch.logit(torch.tensor(other_cal["best_prob"])).item()
            else:
                if "best_prob" in other_cal:
                    fixed_thr = other_cal["best_prob"]
                elif "best_logit" in other_cal:
                    fixed_thr = torch.sigmoid(torch.tensor(other_cal["best_logit"])).item()
            if fixed_thr is None:
                print("Calibration file missing threshold for", other_head, file=sys.stderr)
                return
            source = "calibration"
        else:
            if args.head == "onset":
                fixed_thr = args.fixed_offset_logit if use_logits else args.fixed_offset_prob
                flag_name = "--fixed_offset_logit" if use_logits else "--fixed_offset_prob"
            else:
                fixed_thr = args.fixed_onset_logit if use_logits else args.fixed_onset_prob
                flag_name = "--fixed_onset_logit" if use_logits else "--fixed_onset_prob"
            if fixed_thr is None:
                print(
                    f"error: specify {flag_name} or --calibration to fix {other_head} threshold",
                    file=sys.stderr,
                )
                return
            source = "flag"

        print(
            f"Per-head sweep: head={args.head}, mode={mode}, fixed_{other_head}={fixed_thr:.3f} (source={source})"
        )
        printed_header = print_sweep_header(include_k_column, printed_header)
        for t in sweep_vals:
            if args.head == "onset":
                on_thr, off_thr = t, fixed_thr
            else:
                on_thr, off_thr = fixed_thr, t
            best_result, best_post_stats, total_evals = run_eval_combo(
                on_thr,
                off_thr,
                use_logits,
                k_onset=default_k_onset,
                apply_postproc=sweep_apply_postproc,
                eval_ctx=eval_ctx,
                include_k_column=include_k_column,
                row_records=row_records,
                best_result=best_result,
                best_post_stats=best_post_stats,
                total_evals=total_evals,
                combo_state=combo_state,
                args=args,
                log_progress=_log_progress,
            )

    run_sanity_thresholds(
        args,
        default_k_onset=default_k_onset,
        sweep_apply_postproc=sweep_apply_postproc,
        onset_peak=onset_peak,
        offset_peak=offset_peak,
        onset_lower_hint=onset_lower_hint,
        offset_lower_hint=offset_lower_hint,
        eval_ctx=eval_ctx,
    )

    grid_elapsed = time.time() - combo_state["start_time"]
    stage_durations["grid_pass"] = grid_elapsed
    grid_rate = total_evals / grid_elapsed if grid_elapsed > 0 else 0.0
    _log_progress(
        f"[progress] grid pass done: combos={num_combos}, elapsed={_format_seconds(grid_elapsed)} ({grid_elapsed:.2f}s), rate={grid_rate:.2f}/s",
        force=True,
    )
    total_elapsed = time.time() - t_main_start
    stage_order = [
        ("dataset_init", "dataset"),
        ("materialize", "materialize"),
        ("first_batch", "first_batch"),
        ("data_pass", "data_pass"),
        ("grid_pass", "grid_sweep"),
    ]
    stage_parts = []
    for key, label in stage_order:
        if key in stage_durations:
            dur_val = stage_durations[key]
            stage_parts.append(f"{label}={_format_seconds(dur_val)}")
    stage_summary = ", ".join(stage_parts) if stage_parts else "n/a"
    _log_progress(
        f"[progress] stages: {stage_summary} | total={_format_seconds(total_elapsed)} ({total_elapsed:.2f}s)",
        force=True,
    )

    final_postproc_applied = False
    if best_result is not None:
        pre_dp_result = best_result
        pre_post_stats = copy.deepcopy(best_post_stats) if best_post_stats else None
        pre_ev_mean = pre_dp_result.get("ev_mean")
        final_postproc_applied = bool(best_result.get("_postproc_applied", False))
        need_final_eval = final_apply_postproc and bool(postproc_modules) and not final_postproc_applied
        if need_final_eval:
            final_res, final_stats = evaluate_threshold_pair(
                best_result["onset_thr"],
                best_result["offset_thr"],
                best_result.get("use_logits", False),
                ctx=eval_ctx,
                k_onset=best_result.get("k_onset", default_k_onset),
                apply_postproc=True,
            )
            final_ev_mean = 0.5 * (final_res["ev_f1_on"] + final_res["ev_f1_off"])
            degrade = (
                pre_ev_mean is not None
                and math.isfinite(pre_ev_mean)
                and final_ev_mean < pre_ev_mean - 0.02
            )
            if degrade:
                print("[warn] DP hurt metrics; disabling for this round and returning pre-DP events.", flush=True)
                best_result = pre_dp_result
                best_post_stats = pre_post_stats
                final_postproc_applied = False
            else:
                best_result = {**final_res, "ev_mean": final_ev_mean}
                best_post_stats = copy.deepcopy(final_stats) if final_stats else None
                final_postproc_applied = bool(final_res.get("_postproc_applied", False))

    if best_result is not None and total_evals > 0:
        summary_lines = _format_summary_lines(
            best_result,
            final_postproc_applied=final_postproc_applied,
            postproc_modules=postproc_modules,
            args=args,
            onset_decoder=onset_decoder,
            offset_decoder=offset_decoder,
        )
        for line in summary_lines:
            print(line)
        if postproc_debug and best_post_stats:
            write_post_logs(
                post_logs_dir,
                best_row=best_result,
                stats=best_post_stats,
                split=split,
                frames_display=frame_text,
                max_clips_display=max_clips_text,
                decoder_settings_summary=decoder_settings_summary,
            )

    if (
        eval_cache_enabled
        and cache_db is not None
        and cache_path is not None
        and cache_key
        and cache_fingerprint is not None
        and summary_lines
        and row_records
    ):
        cache_entry_payload = {
            "fingerprint": cache_fingerprint,
            "rows": row_records,
            "summary_lines": summary_lines,
            "best_result": best_result,
            "best_post_stats": best_post_stats,
            "final_postproc_applied": final_postproc_applied,
            "decoder_notice": decoder_settings_summary,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "key": cache_key,
        }
        cache_db.setdefault("entries", {})[cache_key] = _json_sanitize(cache_entry_payload)
        cache_db["version"] = EVAL_CACHE_VERSION
        _persist_eval_cache_db(cache_path, cache_db)

if __name__ == "__main__":
    main()
