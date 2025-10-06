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

import sys, json, torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from utils import load_config, align_pitch_dim
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
    try:
        logit_thrs = _parse_list(argv, "thresholds")
        prob_thrs = _parse_list(argv, "prob_thresholds")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/tivit_best.pt")
    ap.add_argument("--thresholds", metavar="T", nargs="*", help="Logit threshold values")
    ap.add_argument(
        "--prob_thresholds",
        metavar="P",
        nargs="*",
        help="Probability threshold values",
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
    ap.add_argument("--debug", action="store_true", help="Log extra diagnostics for first batch")
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
    args = ap.parse_args(argv)
    args.thresholds = logit_thrs
    args.prob_thresholds = prob_thrs

    if args.thresholds is not None and args.prob_thresholds is not None:
        print("error: --thresholds and --prob_thresholds are mutually exclusive", file=sys.stderr)
        return

    # Unless a calibration file is provided and no head is specified, default to
    # sweeping over probability thresholds when none were specified explicitly.
    if args.thresholds is None and args.prob_thresholds is None:
        if args.head is not None or not args.calibration:
            args.prob_thresholds = DEFAULT_THRESHOLDS.copy()


    cfg = load_config("configs/config.yaml")
    if args.max_clips is not None:
        cfg["dataset"]["max_clips"] = args.max_clips
    if args.frames is not None:
        cfg["dataset"]["frames"] = args.frames
    decode_fps = float(cfg["dataset"].get("decode_fps", 1.0))
    hop_seconds = float(cfg["dataset"].get("hop_seconds", 1.0 / decode_fps))
    event_tolerance = float(
        cfg["dataset"].get("frame_targets", {}).get("tolerance", hop_seconds)
    )
    split = args.split or cfg["dataset"].get("split_val") or cfg["dataset"].get("split") or "val"

    metrics_cfg = cfg.get("training", {}).get("metrics", {}) or {}
    agg_cfg = metrics_cfg.get("aggregation", {}) or {}
    agg_mode = str(agg_cfg.get("mode", "any")).lower()
    agg_k_cfg = agg_cfg.get("k", {}) or {}
    default_k_onset = int(agg_k_cfg.get("onset", 1) or 1)

    # build loader
    val_loader = make_dataloader(cfg, split=split)
    if isinstance(val_loader, dict):
        val_loader = val_loader.get(split, next(iter(val_loader.values())))
    if isinstance(val_loader, (list, tuple)):
        val_loader = val_loader[0]

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
    with torch.no_grad():
        for batch in val_loader:
            x = batch["video"]
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

            if args.debug and len(onset_logits_list) == 1:
                print("[DEBUG] batch video", x.shape, "onset_logits", onset_logits.shape)
                print(
                    "[DEBUG] onset_roll nonzero=",
                    int(batch["onset_roll"].sum().item()),
                    "offset_roll nonzero=",
                    int(batch["offset_roll"].sum().item()),
                )
                
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

    include_k_column = agg_mode == "k_of_p"

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

    if agg_mode == "k_of_p" and args.sweep_k_onset:
        k_candidates = sorted({default_k_onset, 1, 2, 3})
    else:
        k_candidates = [default_k_onset]

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


    if args.head is None:
        # Evaluate at calibrated thresholds if provided.
        if args.calibration:
            with open(args.calibration) as f:
                calib = json.load(f)
            on_cal = calib.get("onset", {})
            off_cal = calib.get("offset", {})
            if "best_logit" in on_cal and "best_logit" in off_cal:
                _header()
                res = _eval_pair(
                    on_cal["best_logit"],
                    off_cal["best_logit"],
                    use_logits=True,
                    k_onset=default_k_onset,
                )
                _print_row(res)
                _update_best(res)
            elif "best_prob" in on_cal and "best_prob" in off_cal:
                _header()
                res = _eval_pair(
                    on_cal["best_prob"],
                    off_cal["best_prob"],
                    use_logits=False,
                    k_onset=default_k_onset,
                )
                _print_row(res)
                _update_best(res)
            else:
                print("Calibration file missing best_logit/best_prob keys", file=sys.stderr)

        # Sweep over provided threshold grids.
        if args.thresholds:
            _header()
            for t in args.thresholds:
                res = _eval_pair(t, t, use_logits=True, k_onset=default_k_onset)
                _print_row(res)
                _update_best(res)
        if args.prob_thresholds:
            _header()
            for k_val in k_candidates:
                for on_thr in args.prob_thresholds:
                    off_candidates = (
                        args.prob_thresholds if args.grid_prob_thresholds else [on_thr]
                    )
                    for off_thr in off_candidates:
                        res = _eval_pair(on_thr, off_thr, use_logits=False, k_onset=k_val)
                        _print_row(res)
                        _update_best(res)
    else:
        # Per-head sweep
        if args.thresholds is not None:
            sweep_vals = args.thresholds
            use_logits = True
            mode = "logit"
        else:
            sweep_vals = args.prob_thresholds
            use_logits = False
            mode = "prob"

        if sweep_vals is None:
            print(
                "error: specify --thresholds or --prob_thresholds for per-head sweep",
                file=sys.stderr,
            )
            return
        
        other_head = "offset" if args.head == "onset" else "onset"
        fixed_thr = None
        source = None

        if args.calibration:
            with open(args.calibration) as f:
                calib = json.load(f)
            other_cal = calib.get(other_head, {})
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
            res = _eval_pair(on_thr, off_thr, use_logits, k_onset=default_k_onset)
            _print_row(res)
            _update_best(res)

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

