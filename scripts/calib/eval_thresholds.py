#!/usr/bin/env python3
"""Purpose:
    Sweep onset/offset thresholds and evaluate frame- and event-level metrics,
    optionally dumping logits for further analysis.

Key Functions/Classes:
    - _parse_list(): Custom parser that supports comma- or space-separated CLI
      threshold lists.
    - _event_f1(): Computes event-level F1 scores using tolerance-aware
      matching on the time grid.
    - main(): Parses CLI options, loads a checkpoint, iterates over the
      dataloader, and prints metric summaries for each threshold.

CLI:
    Run ``python scripts/eval_thresholds.py --ckpt <path>`` with optional
    ``--thresholds``/``--prob_thresholds`` lists, ``--split`` to choose a
    dataset split, and ``--dump_logits`` to save logits to NPZ.
"""

import sys, json, time, math, os, torch
from collections import Counter
import numpy as np
import torch.nn.functional as F
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config, align_pitch_dim
from utils.identifiers import canonical_video_id
from utils.time_grid import frame_to_sec
from data import make_dataloader
from models import build_model


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


def _binary_f1(pred, target, eps=1e-8):
    """Binary F1 score for tensors in {0,1}.

    Returns None if both pred and target are all zeros."""
    if target.sum().item() == 0 and pred.sum().item() == 0:
        return None
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


def _pool_roll_BT(x_btP: torch.Tensor, Tprime: int) -> torch.Tensor:
    """Downsample a (B,T,P) pianoroll along time using max pooling.

    This mirrors the alignment logic used during training so that frame-level
    targets match the model's temporal resolution ``Tprime``.
    """
    x = x_btP.permute(0, 2, 1)  # (B,P,T)
    x = F.adaptive_max_pool1d(x, Tprime)  # (B,P,T')
    return x.permute(0, 2, 1).contiguous()  # (B,T',P)
    
def main():
    import argparse

    argv = sys.argv[1:]
    t_main_start = time.time()
    try:
        logit_thrs = _parse_list(argv, "thresholds")
        prob_thrs = _parse_list(argv, "prob_thresholds")
        offset_logit_thrs = _parse_list(argv, "offset_thresholds")
        offset_prob_thrs = _parse_list(argv, "offset_prob_thresholds")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return

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
    # Explicit thresholds for the non-swept head when no calibration is provided
    ap.add_argument("--fixed_offset_prob", type=float)
    ap.add_argument("--fixed_offset_logit", type=float)
    ap.add_argument("--fixed_onset_prob", type=float)
    ap.add_argument("--fixed_onset_logit", type=float)
    # Optional temperature and bias parameters for logit calibration
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
    ap.add_argument("--split", choices=["train", "val", "test"], help="Dataset split to evaluate")
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--only", help="Restrict evaluation to a single canonical video id")
    ap.add_argument("--debug", action="store_true", help="Log extra diagnostics for first batch")
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
    args = ap.parse_args(argv)
    args.thresholds = logit_thrs
    args.prob_thresholds = prob_thrs
    args.offset_thresholds = offset_logit_thrs
    args.offset_prob_thresholds = offset_prob_thrs

    if args.thresholds is not None and args.prob_thresholds is not None:
        print("error: --thresholds and --prob_thresholds are mutually exclusive", file=sys.stderr)
        return
    if args.offset_thresholds is not None and args.offset_prob_thresholds is not None:
        print(
            "error: --offset_thresholds and --offset_prob_thresholds are mutually exclusive",
            file=sys.stderr,
        )
        return

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
            return

    if args.thresholds is not None and args.offset_thresholds is not None:
        if len(args.thresholds) != len(args.offset_thresholds):
            print(
                "error: --thresholds and --offset_thresholds must contain the same number of values",
                file=sys.stderr,
            )
            return

    if args.prob_thresholds is not None and args.offset_prob_thresholds is not None:
        if not args.grid_prob_thresholds and len(args.prob_thresholds) != len(args.offset_prob_thresholds):
            print(
                "error: probability lists must match lengths unless --grid_prob_thresholds is enabled",
                file=sys.stderr,
            )
            return
    log_handle = None

    if args.head is None and args.prob_thresholds is not None and prob_reuse.get("offset_from_onset"):
        print("[eval] offset probability thresholds not provided; reusing onset list", flush=True)
    if args.head is None and args.thresholds is not None and logit_reuse.get("offset_from_onset"):
        print("[eval] offset logit thresholds not provided; reusing onset list", flush=True)
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
    
    stage_durations = {}
    BAD_CLIP_RETRY_LIMIT = 3
    bad_clip_counts = Counter()
    skip_paths = set()
    lag_ms_samples = []
    lag_source_counter = Counter()
    skipped_batches = 0

    def _extract_lag_values(value):
        vals = []
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
        sources = []
        if value is None:
            return sources
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item:
                    sources.append(item)
        elif isinstance(value, str) and value:
            sources.append(value)
        return sources

    def _filter_batch(batch, keep_indices):
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
                if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == total:
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

    def _handle_bad_batch(paths, exc):
        nonlocal skipped_batches
        skipped_batches += 1
        safe_paths = [str(p) for p in (paths or []) if p]
        if safe_paths:
            first = Path(safe_paths[0]).name
            extra = len(safe_paths) - 1
            clip_desc = f"{first}+{extra} more" if extra > 0 else first
        else:
            clip_desc = "<unknown>"
        err_type = type(exc).__name__
        _log_progress(
            f"[warn] batch failed ({err_type}): clip={clip_desc} error={exc}",
            force=True,
        )
        for path in set(safe_paths):
            bad_clip_counts[path] += 1
            if bad_clip_counts[path] >= BAD_CLIP_RETRY_LIMIT and path not in skip_paths:
                skip_paths.add(path)
                _log_progress(
                    f"[progress] marked clip as bad after {BAD_CLIP_RETRY_LIMIT} failures: {Path(path).name}",
                    force=True,
                )
    # Unless a calibration file is provided and no head is specified, default to
    # sweeping over probability thresholds when none were specified explicitly.
    if args.thresholds is None and args.prob_thresholds is None:
        if args.head is not None or not args.calibration:
            args.prob_thresholds = DEFAULT_THRESHOLDS.copy()


    cfg = dict(load_config("configs/config.yaml"))
    dataset_raw = cfg.get("dataset")
    if isinstance(dataset_raw, dict):
        dataset_cfg = dict(dataset_raw)
    else:
        dataset_cfg = {}
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
    if args.debug:
        dataset_cfg["num_workers"] = 0
        dataset_cfg["persistent_workers"] = False
        dataset_cfg["pin_memory"] = False
        print("[debug] num_workers=0, persistent_workers=False, pin_memory=False", flush=True)
    decode_fps = float(dataset_cfg.get("decode_fps", 1.0))
    hop_seconds = float(dataset_cfg.get("hop_seconds", 1.0 / decode_fps))
    event_tolerance = float(
        dataset_cfg.get("frame_targets", {}).get("tolerance", hop_seconds)
    )
    split = args.split or dataset_cfg.get("split_val") or dataset_cfg.get("split") or "val"

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
    agg_mode = str(agg_cfg.get("mode", "any")).lower()
    agg_k_cfg = agg_cfg.get("k", {}) or {}
    default_k_onset = int(agg_k_cfg.get("onset", 1) or 1)

    calibration_data = None
    if args.calibration:
        with open(args.calibration) as f:
            calibration_data = json.load(f)

    include_k_column = agg_mode == "k_of_p"
    if include_k_column and args.sweep_k_onset and args.head is None:
        k_candidates = sorted({default_k_onset, 1, 2, 3})
    else:
        k_candidates = [default_k_onset]

    # build loader
    t_dataset_build0 = time.time()
    val_loader = make_dataloader(cfg, split=split)
    if isinstance(val_loader, dict):
        val_loader = val_loader.get(split, next(iter(val_loader.values())))
    if isinstance(val_loader, (list, tuple)):
        val_loader = val_loader[0]

    dataset = getattr(val_loader, "dataset", None)
    dataset_elapsed = time.time() - t_dataset_build0
    stage_durations["dataset_init"] = dataset_elapsed
    dataset_name = dataset.__class__.__name__ if dataset is not None else type(val_loader).__name__
    dataset_len = None
    dataset_count = "?"
    if dataset is not None:
        try:
            dataset_len = len(dataset)
            dataset_count = str(dataset_len)
        except TypeError:
            dataset_len = None
            dataset_count = "?"
    batch_size_val = getattr(val_loader, "batch_size", None)
    batch_display = str(batch_size_val) if batch_size_val is not None else "?"
    worker_count = getattr(val_loader, "num_workers", None)
    worker_display = str(worker_count) if worker_count is not None else "?"
    video_count_display = _dataset_video_count(dataset)
    print(
        f"[progress] dataset ready (videos={video_count_display}, workers={worker_display})",
        flush=True,
    )
    _log_progress(
        f"[progress] dataset ready in {_format_seconds(dataset_elapsed)} ({dataset_elapsed:.2f}s) "
        f"backend={dataset_name} len={dataset_count} batch={batch_display}",
        force=True,
    )
    frame_summary = getattr(dataset, "frame_target_summary", None)
    if frame_summary:
        frame_summary_display = frame_summary
        if avlag_disabled and "lag_source=" in frame_summary_display:
            prefix, suffix = frame_summary_display.split("lag_source=", 1)
            if "," in suffix:
                _, tail = suffix.split(",", 1)
                frame_summary_display = f"{prefix}lag_source=no_avlag,{tail}"
            else:
                frame_summary_display = f"{prefix}lag_source=no_avlag"
        _log_progress(f"[progress] {frame_summary_display}", force=True)

    if dataset_len is not None:
        max_cap = args.max_clips if args.max_clips is not None else dataset_len
        target_clips = min(dataset_len, max_cap)
    else:
        target_clips = args.max_clips

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

    target_display = str(target_clips) if target_clips is not None else "?"
    _log_progress(
        f"[progress] starting: clips={target_display} combos={num_combos} (thr={thr_desc}, k_sweep={k_sweep_state})",
        force=True,
    )

    # load model + ckpt
    model = build_model(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    # run model once to collect logits/probabilities and targets
    onset_logits_list, offset_logits_list = [], []
    pitch_logits_list = []
    onset_probs, offset_probs = [], []
    onset_tgts, offset_tgts = [], []
    clips_done = 0
    t_data0 = time.time()
    last_clip_print = t_data0
    heartbeat_interval = max(10.0, float(args.progress_interval or 10.0))
    last_heartbeat = t_data0
    last_clip_name = "-"
    first_batch_time = None
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
                        _log_progress(
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
                    _log_progress(
                        f"[progress] first batch ready in {_format_seconds(first_wait)} ({first_wait:.2f}s) – includes decode/A/V lag warmup",
                        force=True,
                    )
                out = model(x)

                # prefer *_logits if present; fallback to old naming
                onset_logits = out["onset_logits"] if "onset_logits" in out else out.get("onset")
                offset_logits = out["offset_logits"] if "offset_logits" in out else out.get("offset")
                pitch_logits = out.get("pitch_logits")
                
                # Apply temperature scaling and bias for calibration
                onset_logits = onset_logits / args.temperature + args.bias
                offset_logits = offset_logits / args.temperature + args.bias
                onset_prob = torch.sigmoid(onset_logits)
                offset_prob = torch.sigmoid(offset_logits)

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

                if args.debug and len(onset_logits_list) == 1:
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
                    _log_progress(
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
                        clips_total_display = target_display
                        _log_progress(
                            f"[progress] clips {processed_display}/{clips_total_display}  ({pct_display}%)  elapsed={_format_seconds(elapsed)}  eta≈{eta_display}",
                            force=progress_force,
                        )
                        last_clip_print = now
                if target_clips is not None and clips_done >= target_clips:
                    break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                _handle_bad_batch(paths, exc)
                continue

    elapsed_data = time.time() - t_data0
    stage_durations["data_pass"] = elapsed_data
    throughput = clips_done / elapsed_data if elapsed_data > 0 else 0.0
    processed_display = clips_done if target_clips is None else min(clips_done, target_clips)
    skipped_display = len(skip_paths)
    elapsed_display = _format_seconds(elapsed_data)
    expected_display = target_clips if target_clips is not None else "?"
    _log_progress(
        f"[progress] data pass done: clips={processed_display}, expected={expected_display}, skipped={skipped_display}, elapsed={elapsed_display}",
        force=True,
    )
    _log_progress(
        f"[progress] throughput: {throughput:.2f} clips/s ({elapsed_data:.2f}s)",
        force=True,
    )
    if avlag_disabled:
        _log_progress("[progress] A/V lag ms stats: disabled (all zero).", force=True)
    elif lag_ms_samples:
        lag_arr = np.asarray(lag_ms_samples, dtype=np.float32)
        lag_mean = float(lag_arr.mean())
        lag_median = float(np.median(lag_arr))
        lag_p95 = float(np.percentile(lag_arr, 95))
        _log_progress(
            "[progress] A/V lag ms stats: mean={:.1f} median={:.1f} p95={:.1f} samples={}".format(
                lag_mean,
                lag_median,
                lag_p95,
                lag_arr.size,
            ),
            force=True,
        )
    if lag_source_counter and not avlag_disabled:
        top_sources = ", ".join(f"{src}:{cnt}" for src, cnt in lag_source_counter.most_common(3))
        _log_progress(f"[progress] lag sources top: {top_sources}", force=True)
    if skipped_batches:
        _log_progress(f"[progress] batches skipped due to errors: {skipped_batches}", force=True)
    if bad_clip_counts:
        summary_bits = ", ".join(f"{Path(p).name}:{count}" for p, count in bad_clip_counts.items())
        _log_progress(f"[progress] bad clip retries: {summary_bits}", force=True)
    if skip_paths:
        skip_names = ", ".join(Path(p).name for p in sorted(skip_paths))
        _log_progress(f"[progress] permanently skipped clips: {skip_names}", force=True)

    if not onset_logits_list:
        _log_progress("[progress] no valid clips processed; aborting.", force=True)
        print("error: no valid clips processed; aborting.", file=sys.stderr)
        return

    onset_logits = torch.cat(onset_logits_list, dim=0)
    offset_logits = torch.cat(offset_logits_list, dim=0)
    pitch_logits = torch.cat(pitch_logits_list, dim=0) if pitch_logits_list else None
    onset_probs = torch.cat(onset_probs, dim=0)
    offset_probs = torch.cat(offset_probs, dim=0)
    onset_tgts = torch.cat(onset_tgts, dim=0)
    offset_tgts = torch.cat(offset_tgts, dim=0)

    T_logits, P_logits = onset_probs.shape[1], onset_probs.shape[2]
    if onset_tgts.shape[1] != T_logits:
        onset_tgts = _pool_roll_BT(onset_tgts, T_logits)
        offset_tgts = _pool_roll_BT(offset_tgts, T_logits)
    onset_tgts = align_pitch_dim(onset_probs, onset_tgts, "onset")
    offset_tgts = align_pitch_dim(offset_probs, offset_tgts, "offset")
    
    if args.debug:
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

    # Use all key/time positions rather than collapsing with ``any``.
    # Collapsing across the note dimension causes the predicted rate to be
    # either 0 or 1 for a clip, which in turn makes F1-threshold sweeps
    # uninformative.  Instead we compute metrics over the full pianoroll so
    # that the positive rate varies smoothly with the threshold.
    onset_true_bin = (onset_tgts > 0).float()
    offset_true_bin = (offset_tgts > 0).float()
    
    def _eval_pair(on_thr, off_thr, use_logits, *, k_onset=None):
        if k_onset is None:
            k_onset = default_k_onset
        if use_logits:
            onset_pred_bin = (onset_logits >= on_thr).float()
            offset_pred_bin = (offset_logits >= off_thr).float()
        else:
            onset_pred_bin = (onset_probs >= on_thr).float()
            offset_pred_bin = (offset_probs >= off_thr).float()

        if agg_mode == "k_of_p" and k_onset > 1:
            counts = onset_pred_bin.sum(dim=-1, keepdim=True)
            keep = (counts >= k_onset).float()
            onset_pred_bin = onset_pred_bin * keep
        
        f1_on = _binary_f1(onset_pred_bin.reshape(-1), onset_true_bin.reshape(-1))
        f1_off = _binary_f1(offset_pred_bin.reshape(-1), offset_true_bin.reshape(-1))
        ev_f1_on = _event_f1(onset_pred_bin, onset_true_bin, hop_seconds, event_tolerance)
        ev_f1_off = _event_f1(offset_pred_bin, offset_true_bin, hop_seconds, event_tolerance)
        onset_pred_rate = onset_pred_bin.mean().item()
        onset_pos_rate = onset_true_bin.mean().item()

        f1_on = 0.0 if f1_on is None else f1_on
        f1_off = 0.0 if f1_off is None else f1_off
        ev_f1_on = 0.0 if ev_f1_on is None else ev_f1_on
        ev_f1_off = 0.0 if ev_f1_off is None else ev_f1_off

        return {
            "onset_thr": float(on_thr),
            "offset_thr": float(off_thr),
            "f1_on": float(f1_on),
            "f1_off": float(f1_off),
            "onset_pred_rate": float(onset_pred_rate),
            "onset_pos_rate": float(onset_pos_rate),
            "ev_f1_on": float(ev_f1_on),
            "ev_f1_off": float(ev_f1_off),
            "k_onset": int(k_onset),
            "use_logits": bool(use_logits),
        }
    
    printed_header = False

    def _header():
        nonlocal printed_header
        if not printed_header:
            cols = [
                "onset_thr",
                "offset_thr",
            ]
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
            printed_header = True
    
    def _print_row(res: dict):
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

    best_result = None
    total_evals = 0

    def _update_best(res: dict):
        nonlocal best_result
        nonlocal total_evals
        total_evals += 1
        ev_mean = 0.5 * (res["ev_f1_on"] + res["ev_f1_off"])
        if best_result is None:
            best_result = {**res, "ev_mean": ev_mean}
            return
        best_mean = best_result.get("ev_mean", -1.0)
        if ev_mean > best_mean + 1e-9:
            best_result = {**res, "ev_mean": ev_mean}
        elif abs(ev_mean - best_mean) <= 1e-9 and res["ev_f1_on"] > best_result["ev_f1_on"] + 1e-9:
            best_result = {**res, "ev_mean": ev_mean}
    
    combo_idx = 0
    t_grid0 = time.time()
    last_grid_print = t_grid0
    _log_progress(f"[progress] grid sweep start: combos={num_combos}", force=True)

    def _run_eval(on_thr, off_thr, use_logits, *, k_onset=None):
        nonlocal combo_idx
        nonlocal last_grid_print
        res = _eval_pair(on_thr, off_thr, use_logits, k_onset=k_onset)
        _print_row(res)
        _update_best(res)
        combo_idx += 1
        if args.progress and num_combos > 0:
            now = time.time()
            progress_force = combo_idx == 1 or combo_idx == num_combos
            if progress_force or now - last_grid_print >= args.progress_interval:
                elapsed = now - t_grid0
                if combo_idx > 0 and num_combos:
                    remaining = max(num_combos - combo_idx, 0)
                    eta_seconds = (elapsed / combo_idx) * remaining if combo_idx else 0.0
                    eta_display = _format_seconds(eta_seconds)
                else:
                    eta_display = "??:??"
                k_display = k_onset if k_onset is not None else default_k_onset
                _log_progress(
                    f"[progress] grid {combo_idx}/{num_combos}  onset_thr={on_thr:.3f}  offset_thr={off_thr:.3f}  k_onset={k_display}  elapsed={_format_seconds(elapsed)}  eta≈{eta_display}",
                    force=progress_force,
                )
                last_grid_print = now
        return res


    if args.head is None:
        # Evaluate at calibrated thresholds if provided.
        if calibration_data:
            on_cal = calibration_data.get("onset", {})
            off_cal = calibration_data.get("offset", {})
            if "best_logit" in on_cal and "best_logit" in off_cal:
                _header()
                _run_eval(
                    on_cal["best_logit"],
                    off_cal["best_logit"],
                    True,
                    k_onset=default_k_onset,
                )
            elif "best_prob" in on_cal and "best_prob" in off_cal:
                _header()
                _run_eval(
                    on_cal["best_prob"],
                    off_cal["best_prob"],
                    False,
                    k_onset=default_k_onset,
                )
            else:
                print("Calibration file missing best_logit/best_prob keys", file=sys.stderr)

        # Sweep over provided threshold grids.
        if args.thresholds:
            _header()
            offset_list = args.offset_thresholds if args.offset_thresholds else args.thresholds
            if len(offset_list) != len(args.thresholds):
                print("error: offset logit threshold count must match onset count", file=sys.stderr)
                return
            for on_thr, off_thr in zip(args.thresholds, offset_list):
                _run_eval(on_thr, off_thr, True, k_onset=default_k_onset)
        if args.prob_thresholds:
            _header()
            onset_list = args.prob_thresholds
            offset_list = args.offset_prob_thresholds if args.offset_prob_thresholds else onset_list
            for k_val in k_candidates:
                if args.grid_prob_thresholds:
                    for on_thr in onset_list:
                        for off_thr in offset_list:
                            _run_eval(on_thr, off_thr, False, k_onset=k_val)
                else:
                    if len(onset_list) != len(offset_list):
                        print("error: offset probability thresholds must match onset count", file=sys.stderr)
                        return
                    for idx, on_thr in enumerate(onset_list):
                        off_thr = offset_list[idx]
                        _run_eval(on_thr, off_thr, False, k_onset=k_val)
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
        _header()
        for t in sweep_vals:
            if args.head == "onset":
                on_thr, off_thr = t, fixed_thr
            else:
                on_thr, off_thr = fixed_thr, t
            _run_eval(on_thr, off_thr, use_logits, k_onset=default_k_onset)

    grid_elapsed = time.time() - t_grid0
    stage_durations["grid_pass"] = grid_elapsed
    grid_rate = total_evals / grid_elapsed if grid_elapsed > 0 else 0.0
    _log_progress(
        f"[progress] grid pass done: combos={num_combos}, elapsed={_format_seconds(grid_elapsed)} ({grid_elapsed:.2f}s), rate={grid_rate:.2f}/s",
        force=True,
    )
    total_elapsed = time.time() - t_main_start
    stage_order = [
        ("dataset_init", "dataset"),
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

    if best_result is not None and total_evals > 0:
        print(
            "[best-event] mean_event_f1={:.3f} onset_event_f1={:.3f} offset_event_f1={:.3f} k_onset={}".format(
                best_result["ev_mean"],
                best_result["ev_f1_on"],
                best_result["ev_f1_off"],
                best_result["k_onset"],
            )
        )
        if not best_result.get("use_logits", False):
            print(
                "[best-yaml] onset_prob_threshold={:.2f}, offset_prob_threshold={:.2f}, k_onset={}".format(
                    best_result["onset_thr"],
                    best_result["offset_thr"],
                    best_result["k_onset"],
                )
            )

if __name__ == "__main__":
    main()
