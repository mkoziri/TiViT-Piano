"""Purpose:
    Train TiViT-Piano models by loading configuration, preparing dataloaders,
    running the optimization loop, and logging metrics/checkpoints.

Key Functions/Classes:
    - compute_loss() / compute_loss_frame(): Implement clip- and frame-level
      loss calculations with focal/BCE options and auxiliary regularizers.
    - train(): Orchestrates the optimization loop (defined later in the file)
      including gradient scaling, logging, and checkpointing helpers.
    - main(): Parses CLI arguments, initializes logging, and kicks off training
      using the selected configuration.

CLI:
    Run ``python scripts/train.py --config configs/config.yaml`` with optional
    overrides such as ``--epochs`` or dataset adjustments defined within the
    configuration file. Use ``--seed``/``--deterministic`` to control global
    reproducibility settings.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple


import argparse
import faulthandler
import json
import os
import signal
import threading
import time
from pathlib import Path
from time import perf_counter
from datetime import datetime

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import make_dataloader
from models import build_model
from utils import load_config, configure_verbosity, get_logger
from utils.determinism import configure_determinism, resolve_deterministic_flag, resolve_seed
from utils.logging_utils import QUIET_INFO_FLAG

logger = get_logger(__name__)

INNER_EVAL_WARMUP_WINDOWS = 120
INNER_EVAL_CACHE_PATH = Path("runs/cache/inner_eval_stats.json")
INNER_EVAL_CACHE_TOLERANCE = 0.20
INNER_EVAL_N_POS_MIN = 2000
INNER_EVAL_MIN_CAP = 300
INNER_EVAL_MAX_CAP = 5000
INNER_EVAL_SAFETY = 1.10
INNER_EVAL_BUDGET_DEFAULT = 900  # seconds
INNER_EVAL_BUDGET_MAX = 2700  # seconds

# Toggle for per-epoch debug metrics; set to False to mute quickly.
DEBUG_EVAL_METRICS = True

def ensure_dirs(cfg: Mapping[str, Any]) -> Tuple[Path, Path]:
    log_cfg = cfg.get("logging", {})
    ckpt = Path(log_cfg.get("checkpoint_dir", "./checkpoints")).expanduser()
    logd = Path(log_cfg.get("log_dir", "./logs")).expanduser()
    ckpt.mkdir(parents=True, exist_ok=True)
    logd.mkdir(parents=True, exist_ok=True)
    return ckpt, logd

def make_criterions():
    #heads
    # - pitch_logits: (B, P)   -> BCEWithLogits
    # - onset_logits: (B, P)   -> BCEWithLogits
    # - offset_logits: (B, P)  -> BCEWithLogits
    # - hand_logits:  (B, 2)   -> CrossEntropy
    # - clef_logits:  (B, 3)   -> CrossEntropy
    return {
        "pitch": nn.BCEWithLogitsLoss(),
        "onset": nn.BCEWithLogitsLoss(),
        "offset": nn.BCEWithLogitsLoss(),
        "hand": nn.CrossEntropyLoss(),
        "clef": nn.CrossEntropyLoss(),
    }

# --- diagnostics helpers (easy to strip post-debugging) --------------------
def _prediction_stats_from_logits(logits: torch.Tensor) -> Dict[str, float]:
    """Compute basic statistics on sigmoid probabilities for monitoring."""
    if logits is None:
        return {}
    probs = torch.sigmoid(logits.detach())
    if probs.numel() == 0:
        return {}
    stats = {
        "mean": float(probs.mean().item()),
        "std": float(probs.std(unbiased=False).item()),
        "min": float(probs.amin().item()),
        "max": float(probs.amax().item()),
    }
    return stats

def _log_prediction_stats(prefix: str, stats: Dict[str, Dict[str, float]], *, writer=None, step: Optional[int] = None) -> None:
    """Log nested onset/offset stat dict to logger/TensorBoard (diagnostic)."""
    if not stats:
        return
    msg_bits = []
    for head, vals in stats.items():
        if not vals:
            continue
        msg_bits.append(
            f"{head}: "
            + ", ".join(f"{k}={v:.3f}" for k, v in vals.items() if isinstance(v, float))
        )
        if writer is not None and step is not None:
            for k, v in vals.items():
                if isinstance(v, float):
                    writer.add_scalar(f"{prefix}/{head}_{k}", v, step)
    if msg_bits:
        logger.info("[pred-stats:%s] %s", prefix, " | ".join(msg_bits))
# --- End of diagnostics helpers (easy to strip post-debugging)---

def _print_head_grad_norms(model, step_tag=""):
    import math
    head_norms = {}
    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad: 
            continue
        lname = name.lower()
        if any(k in lname for k in ["onset", "offset"]):
            g = p.grad.detach()
            # L2 norm
            n = float(g.norm(2).cpu())
            head_norms[name] = n
    if head_norms:
        top = sorted(head_norms.items(), key=lambda kv: kv[1], reverse=True)[:6]
        logger.debug(
            "[GRAD%s] onset/offset param norms (top): %s",
            f":{step_tag}" if step_tag else "",
            ", ".join(f"{k}={v:.4e}" for k, v in top),
        )
    else:
        logger.debug(
            "[GRAD%s] no onset/offset grads found",
            f":{step_tag}" if step_tag else "",
        )

def _pool_roll_BT(x_btP, Tprime):
    # (B,T,P) -> (B,T',P) with "any positive in window" preserved via max
    # used for pitch/onset/offset rolls when aligning T -> T'
    x = x_btP.permute(0, 2, 1)            # (B,P,T)
    x = F.adaptive_max_pool1d(x, Tprime)  # (B,P,T')
    return x.permute(0, 2, 1).contiguous() # (B,T',P)

def _interp_labels_BT(x_bt, Tprime):
    # (B,T) -> (B,T') for integer class labels (nearest)
    x = x_bt.float().unsqueeze(1)          # (B,1,T)
    x = F.interpolate(x, size=Tprime, mode="nearest")
    return x.squeeze(1).long()             # (B,T')

# --- helper: normalize DataLoader returns -----------------------------------
def _pick_loader(obj, split_key=None):
    """
    Accepts:
      - DataLoader (iterable)
      - dict of loaders (use split_key if present, else first value)
      - list/tuple of loaders (use first)
    """
    if isinstance(obj, dict):
        if split_key and split_key in obj:
            return obj[split_key]
        return next(iter(obj.values()))
    if isinstance(obj, (list, tuple)):
        if not obj:
            raise ValueError("Empty loader list/tuple.")
        return obj[0]
    return obj

def _targets_summary(loader) -> Optional[str]:
    dl = _pick_loader(loader)
    dataset = getattr(dl, 'dataset', None)
    if dataset is None:
        return None
    summary = getattr(dataset, 'frame_target_summary', None)
    if isinstance(summary, str):
        return summary
    spec = getattr(dataset, 'frame_target_spec', None)
    if spec is not None:
        return spec.summary()
    return None


def _time_pool_out_to_clip(out: dict) -> dict:
    """
    If logits are time-distributed (B,T,...) but we're about to use clip losses,
    reduce over time with mean so shapes match clip targets.
    """
    pooled = dict(out)  # shallow copy
    if out["pitch_logits"].dim() == 3:
        pooled["pitch_logits"]  = out["pitch_logits"].mean(dim=1)      # (B,P)
    if out["onset_logits"].dim() == 3:
        pooled["onset_logits"]  = out["onset_logits"].mean(dim=1)      # (B,P)
    if out["offset_logits"].dim() == 3:
        pooled["offset_logits"] = out["offset_logits"].mean(dim=1)     # (B,P)
    if out["hand_logits"].dim() == 3:
        pooled["hand_logits"]   = out["hand_logits"].mean(dim=1)       # (B,2)
    if out["clef_logits"].dim() == 3:
        pooled["clef_logits"]   = out["clef_logits"].mean(dim=1)       # (B,3)
    return pooled
    
class FocalBCE(nn.Module):
    """
    Binary focal loss on logits. Targets in {0,1}, logits are raw (pre-sigmoid).
    gamma controls down-weighting easy examples; alpha biases class weighting.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.10, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        p  = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # per-element
        pt = targets * p + (1.0 - targets) * (1.0 - p)   # prob of the true class
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def _set_onoff_head_bias(model, prior: float = 0.02):
    """Calibrate onset/offset last-layer biases using a Bernoulli prior mean."""
    import math
    from typing import Optional
    import torch
    import torch.nn as nn

    prior = float(prior)
    if not (0.0 < prior < 1.0):
        raise ValueError(f"onset/offset bias prior must be in (0,1); got {prior}")

    def _seed_bias(module: Optional[nn.Module]) -> None:
        if not isinstance(module, nn.Sequential) or not module:
            return
        last = module[-1]
        bias = getattr(last, "bias", None)
        if isinstance(bias, torch.Tensor):
            nn.init.constant_(bias, b0)

    b0 = math.log(prior / (1.0 - prior))
    logger.info("[bias_seed] onset/offset bias prior=%.4f (logit=%.4f)", prior, b0)
    with torch.no_grad():
        _seed_bias(getattr(model, "head_onset", None))
        _seed_bias(getattr(model, "head_offset", None))

def _dynamic_pos_weighted_bce(logits: torch.Tensor, targets: torch.Tensor, base_crit: nn.BCEWithLogitsLoss):
    """Compute BCEWithLogits with adaptive pos_weight derived from current batch."""

    eps = 1e-6
    target_float = targets.float()
    positive_rate = target_float.mean().clamp(min=eps, max=1.0 - eps)
    pos_weight = ((1.0 - positive_rate) / positive_rate).clamp(1.0, 100.0)

    reduction = getattr(base_crit, "reduction", "mean")
    weight = getattr(base_crit, "weight", None)
    if weight is not None:
        weight = weight.to(device=logits.device, dtype=logits.dtype)

    pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)

    return F.binary_cross_entropy_with_logits(
        logits,
        target_float,
        weight=weight,
        pos_weight=pos_weight,
        reduction=reduction,
    )

def compute_loss(out: dict, tgt: dict, crit: dict, weights: dict):
    # Guard: if logits are time-distributed but we're in clip-loss, pool over time
    if out["pitch_logits"].dim() == 3:  # (B,T,P)
        out = _time_pool_out_to_clip(out)
        
    loss_pitch = _dynamic_pos_weighted_bce(out["pitch_logits"], tgt["pitch"], crit["pitch"]) * weights["pitch"]
    loss_onset = _dynamic_pos_weighted_bce(out["onset_logits"], tgt["onset"], crit["onset"]) * weights["onset"]
    loss_offset = _dynamic_pos_weighted_bce(out["offset_logits"], tgt["offset"], crit["offset"]) * weights["offset"]
    loss_hand   = crit["hand"](out["hand_logits"],    tgt["hand"])    * weights["hand"]
    loss_clef   = crit["clef"](out["clef_logits"],    tgt["clef"])    * weights["clef"]

    total = loss_pitch + loss_onset + loss_offset + loss_hand + loss_clef
    parts = {
        "total": total.item(),
        "pitch": loss_pitch.item(),
        "onset": loss_onset.item(),
        "offset": loss_offset.item(),
        "hand": loss_hand.item(),
        "clef": loss_clef.item(),
    }
    return total, parts
    
def compute_loss_frame(out: dict, batch: dict, weights: dict):
    """
    Frame-level objective with the repo's time alignment:
      - Align targets from T (labels) to T' (model logits) using:
          * rolls (pitch/onset/offset): adaptive_max_pool1d  (ANY-over-window)
          * class labels (hand/clef):   nearest interpolation
      - Onset/Offset loss is selectable from YAML:
          * "bce_pos": BCEWithLogitsLoss(pos_weight=...) with adaptive or fixed pos_weight
          * "focal"  : focal BCE on logits (your original default)
      - Optional negative-class label smoothing for on/off (onoff_neg_smooth)
      - Pitch uses BCEWithLogitsLoss with gentle per-pitch pos_weight (sqrt, clamped)
      - Hand/Clef use CrossEntropy per frame
      - Optional activation-prior regularizer on mean(sigmoid) for on/off
    Returns:
      total_loss (tensor scalar), parts (dict of floats)
    """
    device = out["onset_logits"].device

    # --- logits (at T') ---
    pitch_logit  = out["pitch_logits"]     # (B,T',P)
    onset_logit  = out["onset_logits"]    # (B,T',P)
    offset_logit = out["offset_logits"]   # (B,T',P)
    hand_logit   = out["hand_logits"]     # (B,T',C_hand=2)
    clef_logit   = out["clef_logits"]     # (B,T',C_clef=3)

    B, T_logits, P = pitch_logit.shape

    # --- targets (at original T) ---
    pitch_roll   = batch["pitch_roll"].float()   # (B,T,P)
    onset_roll   = batch["onset_roll"].float()   # (B,T,P)
    offset_roll  = batch["offset_roll"].float()  # (B,T,P)
    hand_frame   = batch["hand_frame"].long()    # (B,T)
    clef_frame   = batch["clef_frame"].long()    # (B,T)
    T_targets = pitch_roll.shape[1]

    # --- time alignment: T -> T' (keep your repo behavior) ---
    if T_targets != T_logits:
        pitch_roll  = _pool_roll_BT(pitch_roll,  T_logits)
        onset_roll  = _pool_roll_BT(onset_roll,  T_logits)
        offset_roll = _pool_roll_BT(offset_roll, T_logits)
        hand_frame  = _interp_labels_BT(hand_frame, T_logits)
        clef_frame  = _interp_labels_BT(clef_frame, T_logits)
    # (this matches the alignment already used elsewhere in your code). :contentReference[oaicite:2]{index=2}

    # --- ensure pitch dimension matches model head ---
    P_tgt = pitch_roll.shape[-1]
    if P_tgt != P:
        if P_tgt == 128 and P == 88:
            # map MIDI 0-127 targets to the 88-key piano range (21-108)
            start = 21
            pitch_roll  = pitch_roll[..., start:start + P]
            onset_roll  = onset_roll[..., start:start + P]
            offset_roll = offset_roll[..., start:start + P]
        else:
            raise ValueError(f"Target pitch dim {P_tgt} does not match model dim {P}")

    # --- optional negative-class smoothing for on/off targets ---
    neg_smooth = float(weights.get("onoff_neg_smooth", 0.0))
    if neg_smooth > 0.0:
        onset_roll  = onset_roll  * (1.0 - neg_smooth) + neg_smooth * (1.0 - onset_roll)
        offset_roll = offset_roll * (1.0 - neg_smooth) + neg_smooth * (1.0 - offset_roll)

    # --- pitch loss: gentle per-pitch pos_weight (sqrt + clamp) ---
    eps = 1e-6
    pos_rate_pitch = pitch_roll.reshape(-1, pitch_roll.shape[-1]).mean(dim=0).clamp_min(eps)  # (P,)
    pos_w_pitch = ((1.0 - pos_rate_pitch) / (pos_rate_pitch + eps)).sqrt().clamp(1.0, 50.0).to(device)
    bce_pitch = nn.BCEWithLogitsLoss(pos_weight=pos_w_pitch)
    loss_pitch = bce_pitch(pitch_logit, pitch_roll) * float(weights.get("pitch", 1.0))

    # --- Helper for adaptive pos_weight calculation ---
    def _adaptive_pos_weight(roll, P, eps=1e-6):
        """
        Computes an adaptive positive class weight for BCE loss based on the mean positive rate
        per pitch dimension in the provided roll tensor.

        Args:
            roll (Tensor): Target tensor of shape (..., P) containing binary labels.
            P (int): Number of pitch classes (last dimension of roll).
            eps (float, optional): Small value to avoid division by zero. Default is 1e-6.

        Returns:
            Tensor: Per-pitch positive class weights (shape: [P]) for use in BCEWithLogitsLoss.
        """
        p = roll.reshape(-1, P).mean(dim=0).clamp_min(eps)
        return ((1.0 - p) / (p + eps)).clamp(1.0, 100.0).detach()

    # --- onset/offset loss: "bce_pos" (adaptive/fixed) OR "focal" ---
    onoff_mode = str(weights.get("onoff_loss", "focal")).lower()  # "bce_pos" | "focal"
    if onoff_mode == "bce_pos":
        pw_mode = str(weights.get("onoff_pos_weight_mode", "adaptive")).lower()  # "adaptive" | "fixed"
        pos_w_on = None
        pos_w_off = None
        if pw_mode == "fixed":
            pw_val = float(weights.get("onoff_pos_weight", 0.0))
            if pw_val > 0.0:
                pos_w_on  = torch.tensor([pw_val], device=device)
                pos_w_off = torch.tensor([pw_val], device=device)
            else:
                # fallback to adaptive if fixed value is not positive
                pos_w_on  = _adaptive_pos_weight(onset_roll, P, eps)
                pos_w_off = _adaptive_pos_weight(offset_roll, P, eps)
        elif pw_mode == "adaptive":
            pos_w_on  = _adaptive_pos_weight(onset_roll, P, eps)
            pos_w_off = _adaptive_pos_weight(offset_roll, P, eps)
        else:
            # fallback to adaptive if unknown mode
            pos_w_on  = _adaptive_pos_weight(onset_roll, P, eps)
            pos_w_off = _adaptive_pos_weight(offset_roll, P, eps)

        bce_on  = nn.BCEWithLogitsLoss(pos_weight=pos_w_on)
        bce_off = nn.BCEWithLogitsLoss(pos_weight=pos_w_off)
        loss_onset  = bce_on(onset_logit,  onset_roll)
        loss_offset = bce_off(offset_logit, offset_roll)

    else:
        # focal BCE on logits (your original path)
        gamma = float(weights.get("focal_gamma", 2.0))
        alpha = float(weights.get("focal_alpha", 0.15))
        bce_on  = F.binary_cross_entropy_with_logits(onset_logit,  onset_roll,  reduction="none")
        bce_off = F.binary_cross_entropy_with_logits(offset_logit, offset_roll, reduction="none")
        p_on    = torch.sigmoid(onset_logit)
        p_off   = torch.sigmoid(offset_logit)
        w_on    = (alpha * (1.0 - p_on).pow(gamma)).detach()
        w_off   = (alpha * (1.0 - p_off).pow(gamma)).detach()
        loss_onset  = (w_on  * bce_on ).mean()
        loss_offset = (w_off * bce_off).mean()

    loss_onset  = loss_onset  * float(weights.get("onset",  1.0))
    loss_offset = loss_offset * float(weights.get("offset", 1.0))

    # --- hand / clef CE at T' ---
    ce = nn.CrossEntropyLoss()
    loss_hand = ce(hand_logit.reshape(B*T_logits, -1), hand_frame.reshape(B*T_logits)) * float(weights.get("hand", 1.0))
    loss_clef = ce(clef_logit.reshape(B*T_logits, -1), clef_frame.reshape(B*T_logits)) * float(weights.get("clef", 1.0))

    # --- total + optional activation prior ---
    total = loss_pitch + loss_onset + loss_offset + loss_hand + loss_clef
    parts = {
        "pitch":  float(loss_pitch.detach().cpu()),
        "onset":  float(loss_onset.detach().cpu()),
        "offset": float(loss_offset.detach().cpu()),
        "hand":   float(loss_hand.detach().cpu()),
        "clef":   float(loss_clef.detach().cpu()),
    }

    prior_w = float(weights.get("onoff_prior_weight", 0.0))
    if prior_w > 0.0:
        prior_target = float(weights.get("onoff_prior_mean", 0.12))
        p_on_mean  = torch.sigmoid(onset_logit).mean()
        p_off_mean = torch.sigmoid(offset_logit).mean()
        reg_onoff = prior_w * (p_on_mean - prior_target).abs() + prior_w * (p_off_mean - prior_target).abs()
        total = total + reg_onoff
        parts["reg_onoff"] = float(reg_onoff.detach().cpu())

    parts["total"] = float(total.detach().cpu())
    return total, parts



def fabricate_dummy_targets(batch_size: int):
    # Simple random targets to exercise the loop
    device = "cpu"
    P = 88
    tgt = {
        "pitch": torch.randint(0, 2, (batch_size, P), device=device, dtype=torch.float32),
        "onset": torch.randint(0, 2, (batch_size, P), device=device, dtype=torch.float32),
        "offset": torch.randint(0, 2, (batch_size, P), device=device, dtype=torch.float32),
        "hand": torch.randint(0, 2, (batch_size,), device=device),
        "clef": torch.randint(0, 3, (batch_size,), device=device),
    }
    return tgt
    
def _binary_f1(pred, target, eps=1e-8):
    # pred, target: float tensors in {0,1}, same shape
    # If there are no positives in target and in pred -> return None (to skip)
    if target.sum().item() == 0 and pred.sum().item() == 0:
        return None
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1

def _acc_from_logits(logits, target):
    # logits: (B, C), target: (B,) long
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()

def _binarize_sigmoid(logits, threshold):
    # logits: (B,), returns 0/1 float tensor
    probs = torch.sigmoid(logits)
    return (probs >= threshold).float()


def _get_onoff_calibration(cfg: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    metrics_cfg = cfg.get("training", {}).get("metrics", {})

    default_thr = float(metrics_cfg.get("prob_threshold", 0.5))
    default_temp = float(metrics_cfg.get("prob_temperature", 1.0))
    default_bias = float(metrics_cfg.get("prob_logit_bias", 0.0))

    onset = {
        "threshold": float(metrics_cfg.get("prob_threshold_onset", default_thr)),
        "temperature": float(metrics_cfg.get("prob_temperature_onset", default_temp)),
        "bias": float(metrics_cfg.get("prob_logit_bias_onset", default_bias)),
    }
    offset = {
        "threshold": float(metrics_cfg.get("prob_threshold_offset", default_thr)),
        "temperature": float(metrics_cfg.get("prob_temperature_offset", default_temp)),
        "bias": float(metrics_cfg.get("prob_logit_bias_offset", default_bias)),
    }

    # Guard against degenerate temperature values that would explode logits.
    eps = 1e-6
    if abs(onset["temperature"]) < eps:
        logger.warning(
            "[metrics] onset temperature %.3g too small; clamping to %.1e",
            onset["temperature"],
            eps,
        )
        onset["temperature"] = eps
    if abs(offset["temperature"]) < eps:
        logger.warning(
            "[metrics] offset temperature %.3g too small; clamping to %.1e",
            offset["temperature"],
            eps,
        )
        offset["temperature"] = eps

    return {"onset": onset, "offset": offset}


def _get_onoff_thresholds(cfg: Mapping[str, Any]):
    cal = _get_onoff_calibration(cfg)
    return cal["onset"]["threshold"], cal["offset"]["threshold"]


def _apply_sigmoid_calibration(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    bias: float = 0.0,
) -> torch.Tensor:
    if temperature != 1.0:
        logits = logits / temperature
    if bias != 0.0:
        logits = logits + bias
    return torch.sigmoid(logits)



def _get_aggregation_config(cfg: Mapping[str, Any]):
    metrics_cfg = cfg.get("training", {}).get("metrics", {})
    agg_cfg = metrics_cfg.get("aggregation", {}) or {}
    mode = str(agg_cfg.get("mode", "any")).lower()
    top_k = int(agg_cfg.get("top_k", 0) or 0)
    tau_sum = float(agg_cfg.get("tau_sum", 0.0) or 0.0)
    k_cfg = agg_cfg.get("k", {}) or {}
    k_onset = int(k_cfg.get("onset", 1) or 1)
    k_offset = int(k_cfg.get("offset", 1) or 1)
    return {
        "mode": mode,
        "top_k": max(0, top_k),
        "tau_sum": max(0.0, tau_sum),
        "k_onset": max(1, k_onset),
        "k_offset": max(1, k_offset),
    }


def _aggregate_onoff_predictions(
    logits: torch.Tensor,
    threshold: float,
    *,
    mode: str,
    k: int,
    top_k: int,
    tau_sum: float,
    temperature: float = 1.0,
    bias: float = 0.0,
    cap_count: Optional[int] = None,
    return_mask: bool = False,
):
    squeeze_time = logits.dim() == 2
    if squeeze_time:
        logits = logits.unsqueeze(1)

    probs = _apply_sigmoid_calibration(logits, temperature=temperature, bias=bias)

    if mode == "top_k_cap" and top_k > 0:
        P = probs.shape[-1]
        if top_k < P:
            topk_idx = probs.topk(top_k, dim=-1).indices
            keep_mask = torch.zeros_like(probs, dtype=torch.bool)
            keep_mask.scatter_(-1, topk_idx, True)
            probs = torch.where(keep_mask, probs, torch.zeros_like(probs))

    key_mask = probs >= threshold

    if cap_count is not None:
        cap_eff = int(cap_count)
        if cap_eff <= 0:
            key_mask = torch.zeros_like(key_mask)
        else:
            P = probs.shape[-1]
            cap_eff = min(cap_eff, P)
            if cap_eff < P:
                topk_idx = probs.topk(cap_eff, dim=-1).indices
                cap_mask = torch.zeros_like(key_mask)
                cap_mask.scatter_(-1, topk_idx, True)
                key_mask = key_mask & cap_mask

    key_counts = key_mask.sum(dim=-1).float()

    if mode == "k_of_p":
        P = probs.shape[-1]
        k_eff = max(1, min(int(k), P))
        pred = key_counts >= k_eff
    elif mode == "sum_prob":
        if tau_sum <= 0.0:
            pred = key_mask.any(dim=-1)
        else:
            summed = probs.sum(dim=-1)
            pred = summed >= tau_sum
    else:  # "any" and fallback, including "top_k_cap"
        pred = key_mask.any(dim=-1)

    if squeeze_time:
        key_counts = key_counts.squeeze(1)
        pred = pred.squeeze(1)
        if return_mask:
            key_mask = key_mask.squeeze(1)

    if return_mask:
        return pred, key_counts, key_mask
    return pred, key_counts


def _accumulate_pred_key_histogram(hist_bins: list[float], counts: torch.Tensor):
    flat_counts = counts.reshape(-1).to(torch.int64)
    hist_bins[0] += (flat_counts == 0).sum().item()
    hist_bins[1] += (flat_counts == 1).sum().item()
    hist_bins[2] += (flat_counts == 2).sum().item()
    hist_bins[3] += (flat_counts == 3).sum().item()
    hist_bins[4] += (flat_counts >= 4).sum().item()


def _init_eval_metric_counts() -> Dict[str, Any]:
    return {
        "pitch_acc": 0.0,
        "hand_acc": 0.0,
        "clef_acc": 0.0,
        "onset_f1": 0.0,
        "offset_f1": 0.0,
        "onset_pos_rate": 0.0,
        "offset_pos_rate": 0.0,
        "onset_pred_rate": 0.0,
        "offset_pred_rate": 0.0,
        "onset_key_sum": 0.0,
        "offset_key_sum": 0.0,
        "onset_frame_count": 0,
        "offset_frame_count": 0,
        "onset_hist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "offset_hist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "n_on": 0,
        "n_off": 0,
    }


def _align_key_mask_to(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    if pred_mask.shape == gt_mask.shape:
        return gt_mask
    if pred_mask.dim() != 3 or gt_mask.dim() != 3:
        raise ValueError("Expected 3D masks for alignment")
    Bp, Tp, Pp = pred_mask.shape
    Bg, Tg, Pg = gt_mask.shape
    if Bp != Bg or Pp != Pg:
        raise ValueError(
            f"Cannot align masks with shapes pred={pred_mask.shape} and gt={gt_mask.shape}"
        )
    if Tg != Tp:
        gt_mask = gt_mask.permute(0, 2, 1)
        gt_mask = F.adaptive_max_pool1d(gt_mask, Tp)
        gt_mask = gt_mask.permute(0, 2, 1).contiguous()
    return gt_mask


def _format_mmss(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    minutes = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


class SafeEvalCollate:
    def __init__(self, base_collate):
        self._base_collate = base_collate

    def __call__(self, batch):
        filtered = [item for item in batch if item is not None]
        if not filtered:
            return None
        if self._base_collate is None:
            return filtered
        return self._base_collate(filtered)


def _unwrap_dataset(dataset):
    current = dataset
    visited = set()
    while hasattr(current, "dataset"):
        key = id(current)
        if key in visited:
            break
        visited.add(key)
        current = current.dataset
    return current


def _resolved_verbosity() -> str:
    return os.environ.get("TIVIT_VERBOSE", "quiet").strip().lower() or "quiet"


def _load_inner_eval_cache(path: Path = INNER_EVAL_CACHE_PATH) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):  # pragma: no cover - defensive
        return None
    if not isinstance(data, dict):
        return None
    return data


def _store_inner_eval_cache(stats: Mapping[str, Any], path: Path = INNER_EVAL_CACHE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = dict(stats)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    except OSError:  # pragma: no cover - best effort persistence
        logger.debug("[train:eval] failed to persist eval cache", exc_info=True)


def _materialized_eval_density(base_dataset) -> Tuple[int, int]:
    snapshot_flags = getattr(base_dataset, "_eval_snapshot_flags", None)
    total_pos = 0
    total_windows = 0
    if isinstance(snapshot_flags, list) and snapshot_flags:
        total_pos = sum(1 for flag in snapshot_flags if flag)
        total_windows = len(snapshot_flags)
    else:
        stats = getattr(base_dataset, "_eval_materialize_stats", {}) or {}
        try:
            total_pos = int(stats.get("positives") or 0)
        except (TypeError, ValueError):
            total_pos = 0
        try:
            total_neg = int(stats.get("negatives") or 0)
        except (TypeError, ValueError):
            total_neg = 0
        total_windows = total_pos + total_neg
    snapshot_entries = getattr(base_dataset, "eval_indices_snapshot", None)
    if isinstance(snapshot_entries, list) and snapshot_entries:
        total_windows = max(total_windows, len(snapshot_entries))
    return total_pos, total_windows


def _build_stratified_selection(base_dataset, cap: int, min_per_video: int) -> Tuple[List[int], Dict[int, int]]:
    cap = max(0, cap)
    snapshot = getattr(base_dataset, "eval_indices_snapshot", None)
    if not isinstance(snapshot, list) or not snapshot:
        return list(range(cap)), {}

    cap = min(cap, len(snapshot))
    video_map: Dict[int, List[int]] = {}
    for idx, (record_idx, _) in enumerate(snapshot):
        video_map.setdefault(record_idx, []).append(idx)
    if not video_map:
        return list(range(cap)), {}

    per_video_seed: Dict[int, int] = {}
    for vid, indices in video_map.items():
        per_video_seed[vid] = min(len(indices), min_per_video)

    alloc_total = sum(per_video_seed.values())
    if alloc_total > cap:
        # Scale down fairly when cap is smaller than requested base allocation.
        vids_desc = sorted(per_video_seed, key=lambda vid: per_video_seed[vid], reverse=True)
        while alloc_total > cap and vids_desc:
            for vid in vids_desc:
                if alloc_total <= cap:
                    break
                if per_video_seed[vid] <= 0:
                    continue
                per_video_seed[vid] -= 1
                alloc_total -= 1

    remaining = cap - alloc_total
    available_map: Dict[int, int] = {
        vid: max(0, len(indices) - per_video_seed.get(vid, 0))
        for vid, indices in video_map.items()
    }
    total_available = sum(available_map.values())
    extras: Dict[int, int] = {vid: 0 for vid in video_map}
    fractions: List[Tuple[float, int]] = []
    if remaining > 0 and total_available > 0:
        for vid, available in available_map.items():
            if available <= 0:
                continue
            share = (available / total_available) * remaining
            extra = int(math.floor(share))
            extras[vid] = extra
            fractions.append((share - extra, vid))
        distributed = sum(extras.values())
        leftover = remaining - distributed
        if leftover > 0:
            for _, vid in sorted(fractions, key=lambda item: item[0], reverse=True):
                if leftover <= 0:
                    break
                if available_map[vid] <= extras[vid]:
                    continue
                extras[vid] += 1
                leftover -= 1

    per_video_total: Dict[int, int] = {}
    for vid, base_alloc in per_video_seed.items():
        total = base_alloc + extras.get(vid, 0)
        total = min(total, len(video_map[vid]))
        per_video_total[vid] = total

    selected: List[int] = []
    for vid, count in per_video_total.items():
        if count <= 0:
            continue
        selected.extend(video_map[vid][:count])

    selected = sorted(set(selected))[:cap]
    return selected, per_video_total


def _compute_eval_cap(total_windows: int, pos_per_window: float, throughput: Optional[float]) -> Dict[str, Any]:
    pos_rate = max(pos_per_window, 1e-6)
    cap_pos = int(math.ceil(INNER_EVAL_N_POS_MIN / pos_rate))

    cap_time_value: Optional[int] = None
    if throughput is not None and throughput > 0:
        cap_time_value = int(math.floor(throughput * INNER_EVAL_BUDGET_DEFAULT * 0.90))
        cap_time_value = max(cap_time_value, 0)

    effective_time_cap = cap_time_value if cap_time_value is not None else float("inf")
    raw_cap = min(cap_pos, effective_time_cap)
    bounded_cap = int(
        min(
            total_windows,
            max(
                INNER_EVAL_MIN_CAP,
                min(INNER_EVAL_MAX_CAP, raw_cap),
            ),
        )
    )

    driver = "pos" if cap_pos <= effective_time_cap else "time"
    if cap_time_value is None:
        driver = "pos"

    return {
        "cap": max(0, bounded_cap),
        "cap_time": cap_time_value,
        "cap_pos": cap_pos,
        "driver": driver,
    }


def _prepare_inner_eval_loader(cfg: Mapping[str, Any], loader: Optional[DataLoader]) -> Optional[DataLoader]:
    if loader is None:
        return None
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader

    base_dataset = _unwrap_dataset(dataset)
    materialize_fn = getattr(base_dataset, "materialize_eval_entries_from_labels", None)
    snapshot = getattr(base_dataset, "eval_indices_snapshot", None)
    if callable(materialize_fn) and not snapshot:
        try:
            materialize_fn()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[train:eval] failed to materialize eval entries: %s", exc)

    def _coerce_positive_int(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    cap_candidates: List[Any] = []
    for env_key in ("AUTOPILOT_CALIB_MAX_CLIPS", "CALIB_MAX_CLIPS"):
        env_val = os.environ.get(env_key)
        if env_val is not None and str(env_val).strip():
            cap_candidates.append(env_val)

    autopilot_cfg = cfg.get("autopilot")
    if isinstance(autopilot_cfg, Mapping):
        cap_candidates.append(autopilot_cfg.get("calib_max_clips"))

    calibration_cfg = cfg.get("calibration")
    if isinstance(calibration_cfg, Mapping):
        cap_candidates.append(calibration_cfg.get("max_clips"))

    dataset_cfg = cfg.get("dataset")
    if isinstance(dataset_cfg, Mapping):
        cap_candidates.append(dataset_cfg.get("max_clips"))

    dataset_attr = getattr(base_dataset, "args_max_clips_or_None", None)
    cap_candidates.append(dataset_attr)

    autopilot_cap = next(
        (
            value
            for value in (_coerce_positive_int(candidate) for candidate in cap_candidates)
            if value is not None
        ),
        None,
    )
    default_cap = 2000
    cap = autopilot_cap if autopilot_cap is not None else default_cap

    try:
        base_len = int(len(base_dataset))
    except Exception:
        base_len = 0

    if base_len > 0:
        target = min(base_len, max(cap, 0))
    else:
        target = max(cap, 0)

    subset_len = base_len if base_len > 0 else target
    subset_len = min(subset_len, target)
    subset_len = max(subset_len, 0)
    subset_indices = list(range(subset_len))
    eval_dataset = Subset(base_dataset, subset_indices)

    safe_collate = SafeEvalCollate(getattr(loader, "collate_fn", None))
    batch_size = getattr(loader, "batch_size", 1)

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
        collate_fn=safe_collate,
        persistent_workers=False,
        timeout=120,
        prefetch_factor=2,
    )
    setattr(eval_loader, "_base_dataset", base_dataset)
    setattr(eval_loader, "_target_total", int(subset_len))
    setattr(eval_loader, "_target_cap", cap)

    snapshot_flags = getattr(base_dataset, "_eval_snapshot_flags", None)
    if isinstance(snapshot_flags, list) and snapshot_flags:
        total_pos = sum(1 for flag in snapshot_flags if flag)
        total_windows = len(snapshot_flags)
    else:
        stats = getattr(base_dataset, "_eval_materialize_stats", {}) or {}
        total_pos = int(stats.get("positives") or 0)
        total_windows = int(stats.get("positives") or 0) + int(stats.get("negatives") or 0)
    snapshot_entries = getattr(base_dataset, "eval_indices_snapshot", None)
    if isinstance(snapshot_entries, list) and snapshot_entries:
        total_windows = max(total_windows, len(snapshot_entries))

    setattr(eval_loader, "_materialized_total_pos", int(total_pos))
    setattr(eval_loader, "_materialized_total_windows", int(total_windows))
    return eval_loader


def _spawn_eval_heartbeat(progress: Dict[str, int], start_time: float, interval: float, stop_event: threading.Event) -> threading.Thread:
    def _runner():
        while not stop_event.wait(interval):
            elapsed = time.perf_counter() - start_time
            processed = progress.get("count", 0)
            total = progress.get("total")
            total_repr = total if isinstance(total, int) and total >= 0 else "?"
            logger.info(
                "[train:eval] heartbeat processed=%s/%s elapsed=%s",
                processed,
                total_repr,
                _format_mmss(elapsed),
            )

    thread = threading.Thread(target=_runner, name="eval-heartbeat", daemon=True)
    thread.start()
    return thread


def _spawn_eval_watchdog(start_time: float, interval: float, stop_event: threading.Event) -> threading.Thread:
    def _runner():
        while not stop_event.wait(interval):
            elapsed = time.perf_counter() - start_time
            logger.info(
                "[train:eval] watchdog alive elapsed=%s",
                _format_mmss(elapsed),
            )

    thread = threading.Thread(target=_runner, name="eval-watchdog", daemon=True)
    thread.start()
    return thread


def _join_thread(thread: Optional[threading.Thread], timeout: float = 1.0) -> None:
    if thread is None:
        return
    try:
        thread.join(timeout=timeout)
    except RuntimeError:
        pass


def _first_parameter_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_optimizer_state(optimizer, device: torch.device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                if value.device != device:
                    state[key] = value.to(device)


def _compute_percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if percentile <= 0:
        return float(sorted_values[0])
    if percentile >= 100:
        return float(sorted_values[-1])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    lower_val = float(sorted_values[lower])
    upper_val = float(sorted_values[upper])
    return lower_val + (upper_val - lower_val) * fraction


def _atomic_write_metrics(metrics: Mapping[str, Any], output_path: Path, lock_timeout: float = 1.0) -> bool:
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    deadline = time.perf_counter() + max(0.0, lock_timeout)
    acquired = False
    while not acquired:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            if time.perf_counter() >= deadline:
                return False
            time.sleep(0.05)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        payload: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                payload[str(key)] = float(value)
            else:
                payload[str(key)] = value
        with tmp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)
        os.replace(tmp_path, output_path)
        return True
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    
# ----------------------- train loop -----------------------

def train_one_epoch(model, train_loader, optimizer, cfg, writer=None, epoch=1):
    summary = _targets_summary(train_loader)
    if summary:
        logger.info(summary)
    model.train()
    crit = make_criterions()  # used only in clip-mode
    w = cfg["training"]["loss_weights"]
    onoff_cal = _get_onoff_calibration(cfg)
    onset_cal = onoff_cal["onset"]
    offset_cal = onoff_cal["offset"]
    thr_on = onset_cal["threshold"]
    thr_off = offset_cal["threshold"]
    agg_cfg = _get_aggregation_config(cfg)

    use_amp = bool(cfg["training"].get("amp", False))
    scaler = GradScaler(enabled=use_amp)

    accum_steps = int(cfg.get("train", {}).get("accumulate_steps", 1))
    grad_clip = float(cfg["optim"].get("grad_clip", 1.0))

    sums = {"total": 0.0, "pitch": 0.0, "onset": 0.0, "offset": 0.0, "hand": 0.0, "clef": 0.0}
    n = 0

    optimizer.zero_grad(set_to_none=True)

    for it, batch in enumerate(train_loader):
        x = batch["video"]

        # Clip-level targets (fallback path)
        have_all = all(k in batch for k in ("pitch", "onset", "offset", "hand", "clef"))
        use_dummy = bool(cfg["training"].get("debug_dummy_labels", False))
        if have_all and not use_dummy:
            tgt = {
                "pitch":  batch["pitch"].long(),
                "hand":   batch["hand"].long(),
                "clef":   batch["clef"].long(),
                "onset":  batch["onset"].float(),
                "offset": batch["offset"].float(),
            }
        else:
            tgt = fabricate_dummy_targets(x.shape[0])

        with autocast(enabled=use_amp):
            out = model(x)

            # Route: frame loss iff model is in frame mode AND batch has frame targets
            use_frame = (
                getattr(model, "head_mode", "clip") == "frame"
                and all(k in batch for k in ("pitch_roll", "onset_roll", "offset_roll", "hand_frame", "clef_frame"))
            )

            if use_frame:
                loss, parts = compute_loss_frame(out, batch, weights=w)
            else:
                # Guard: if model outputs (B,T,...) but we're using clip loss, pool over time
                if out["pitch_logits"].dim() == 3:
                    out = _time_pool_out_to_clip(out)
                loss, parts = compute_loss(out, tgt, crit, w)

        # Backprop / step
        scaler.scale(loss / accum_steps).backward()
        if (it + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            _print_head_grad_norms(model, step_tag=f"e{epoch}_it{it}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Accumulate loss parts
        for k in sums:
            if k in parts:
                sums[k] += parts[k]
        n += 1

        # --- OPTIONAL: train-side predicted positive rates (only frame mode) ---
        if use_frame and "onset_logits" in out and "offset_logits" in out:
            onset_probs = _apply_sigmoid_calibration(
                out["onset_logits"],
                temperature=onset_cal["temperature"],
                bias=onset_cal["bias"],
            )
            offset_probs = _apply_sigmoid_calibration(
                out["offset_logits"],
                temperature=offset_cal["temperature"],
                bias=offset_cal["bias"],
            )
            onset_pred = (onset_probs >= onset_cal["threshold"]).float()
            offset_pred = (offset_probs >= offset_cal["threshold"]).float()
            if "train_onset_pred_rate" not in sums:
                sums["train_onset_pred_rate"] = 0.0
                sums["train_offset_pred_rate"] = 0.0
            sums["train_onset_pred_rate"]  += onset_pred.mean().item()
            sums["train_offset_pred_rate"] += offset_pred.mean().item()

    if (n % accum_steps) != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Average over batches
    avg = {k: (sums[k] / max(1, n)) for k in sums}

    # Log to TensorBoard
    if writer is not None:
        for k, v in avg.items():
            writer.add_scalar(f"train/{k}", v, epoch)

    return avg


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: Mapping[str, Any], best_val: float | None = None):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "best_val": best_val,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    
def evaluate_one_epoch(model, loader, cfg, *, optimizer=None, timeout_minutes: int = 15):
    summary = _targets_summary(loader)
    if summary:
        logger.info(summary)

    verbosity = _resolved_verbosity()
    dataset = getattr(loader, "dataset", None)
    base_dataset = getattr(loader, "_base_dataset", _unwrap_dataset(dataset))
    try:
        raw_total = len(dataset) if dataset is not None else len(loader)
    except Exception:
        raw_total = 0
    if not isinstance(raw_total, int):
        raw_total = 0
    raw_total = max(0, raw_total)

    total_pos = 0
    total_windows = raw_total
    if base_dataset is not None:
        total_pos, total_windows = _materialized_eval_density(base_dataset)
    materialized_total = getattr(loader, "_materialized_total_windows", 0)
    if materialized_total:
        try:
            total_windows = max(total_windows, int(materialized_total))
        except (TypeError, ValueError):
            pass
    if total_windows <= 0:
        total_windows = raw_total

    pos_per_window = (total_pos / total_windows) if total_windows > 0 else 0.0
    pos_per_window = float(pos_per_window)

    if verbosity != "quiet":
        logger.info(
            "[train:eval] pos_per_window=%.2f (total_pos=%d over %d windows)",
            pos_per_window,
            total_pos,
            total_windows,
        )

    cache_data = _load_inner_eval_cache()
    cached_throughput: Optional[float] = None
    if cache_data:
        try:
            cached_throughput = float(cache_data.get("throughput_win_per_s"))
        except (TypeError, ValueError):
            cached_throughput = None
        if cached_throughput is not None and (not math.isfinite(cached_throughput) or cached_throughput <= 0):
            cached_throughput = None

    cap_components = _compute_eval_cap(total_windows, pos_per_window, cached_throughput)
    planned_cap = cap_components["cap"]
    cap_time_est = cap_components["cap_time"]
    cap_pos = cap_components["cap_pos"]
    cap_driver = cap_components["driver"]

    stratified_indices: List[int]
    per_video_alloc: Dict[int, int] = {}
    if isinstance(dataset, Subset) and base_dataset is not None:
        stratified_indices, per_video_alloc = _build_stratified_selection(base_dataset, planned_cap, 10)
        dataset.indices = stratified_indices
        setattr(loader, "_target_total", len(stratified_indices))
        if verbosity != "quiet":
            logger.info(
                "[train:eval] stratified sampling: videos=%d  min_per_video=%d",
                len(per_video_alloc),
                10,
            )
        if verbosity == "debug" and per_video_alloc:
            logger.debug("[train:eval] per-video allocations: %s", per_video_alloc)
    else:
        stratified_indices = list(range(planned_cap))

    total_clips = len(stratified_indices)
    if total_clips <= 0:
        total_clips = planned_cap if planned_cap > 0 else raw_total

    cap_active = min(cap_components["cap"], total_clips)
    cap_driver_active = cap_driver
    cap_time_active = cap_time_est
    throughput_used = cached_throughput
    throughput_live: Optional[float] = None
    cap_logged = False
    total_clips = cap_active

    eval_wall_budget = INNER_EVAL_BUDGET_DEFAULT
    if cached_throughput is not None and cap_time_est is not None and cap_pos > cap_time_est:
        budget_needed = int(math.ceil(cap_pos / max(cached_throughput, 1e-6) * INNER_EVAL_SAFETY))
        if budget_needed > INNER_EVAL_BUDGET_DEFAULT:
            if budget_needed > INNER_EVAL_BUDGET_MAX:
                logger.warning(
                    "[train:eval] requested budget %ds exceeds cap %ds; clamped. F1 variance may be high.",
                    budget_needed,
                    INNER_EVAL_BUDGET_MAX,
                )
                budget_needed = INNER_EVAL_BUDGET_MAX
            logger.warning(
                "[train:eval] increasing timeout to %ds (was %ds) to satisfy N_POS_MIN=%d; pos_per_window=%.2f, throughput=%.2f win/s",
                budget_needed,
                eval_wall_budget,
                INNER_EVAL_N_POS_MIN,
                pos_per_window,
                cached_throughput,
            )
            eval_wall_budget = budget_needed

    print(f"[train:eval] start (clips={total_clips})", flush=True)

    eval_start = perf_counter()
    stop_event = threading.Event()
    progress = {"count": 0, "total": total_clips}
    heartbeat_thread = _spawn_eval_heartbeat(progress, eval_start, 10.0, stop_event)
    watchdog_thread = _spawn_eval_watchdog(eval_start, 15.0, stop_event)

    timeout_seconds = float(eval_wall_budget)
    warmup_target = INNER_EVAL_WARMUP_WINDOWS
    warmup_processed = 0
    warmup_start_time: Optional[float] = None
    warmup_elapsed: Optional[float] = None
    warmup_logged = False
    skip_empty_logged = False
    timed_out = False

    lag_ms_values: list[float] = []
    lag_low_corr = 0
    lag_timeouts = 0

    prev_training = model.training
    original_device = _first_parameter_device(model)
    cpu_device = torch.device("cpu")
    moved_to_cpu = original_device.type != "cpu"
    if moved_to_cpu:
        model.to(cpu_device)
        _move_optimizer_state(optimizer, cpu_device)

    num_threads_prev = torch.get_num_threads()
    interop_prev = None
    if hasattr(torch, "get_num_interop_threads"):
        try:
            interop_prev = torch.get_num_interop_threads()
        except RuntimeError:
            interop_prev = None
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            interop_prev = None

    model.eval()
    crit = make_criterions()
    w = cfg["training"]["loss_weights"]
    metrics_cfg = cfg.get("training", {}).get("metrics", {})
    thr_pitch = float(metrics_cfg.get("prob_threshold", 0.5))
    onoff_cal = _get_onoff_calibration(cfg)
    onset_cal = onoff_cal["onset"]
    offset_cal = onoff_cal["offset"]
    thr_on = onset_cal["threshold"]
    thr_off = offset_cal["threshold"]
    agg_cfg = _get_aggregation_config(cfg)

    logger.info(
        "[eval-thresholds] onset=%.3f offset=%.3f (default=%.3f) | temps (on=%.2f, off=%.2f) bias (on=%.3f, off=%.3f)",
        thr_on,
        thr_off,
        thr_pitch,
        onset_cal["temperature"],
        offset_cal["temperature"],
        onset_cal["bias"],
        offset_cal["bias"],
    )

    onset_logit_stats: Optional[dict] = None
    offset_logit_stats: Optional[dict] = None
    onset_prob_stats: Optional[dict] = None
    offset_prob_stats: Optional[dict] = None
    onset_k_effective: Optional[int] = None
    offset_k_effective: Optional[int] = None

    if DEBUG_EVAL_METRICS:

        def _init_stats() -> dict:
            return {"min": math.inf, "max": -math.inf, "sum": 0.0, "count": 0}

        def _update_stats(stats: dict, tensor: Optional[torch.Tensor]) -> None:
            if tensor is None:
                return
            t = tensor.detach()
            if t.numel() == 0:
                return
            t_min = t.min().item()
            t_max = t.max().item()
            stats["min"] = min(stats["min"], t_min)
            stats["max"] = max(stats["max"], t_max)
            stats["sum"] += t.sum().item()
            stats["count"] += t.numel()

        def _finalize_stats(stats: dict) -> Tuple[float, float, float]:
            if stats["count"] == 0:
                nan = float("nan")
                return nan, nan, nan
            mean = stats["sum"] / stats["count"]
            return stats["min"], mean, stats["max"]

        onset_logit_stats = _init_stats()
        offset_logit_stats = _init_stats()
        onset_prob_stats = _init_stats()
        offset_prob_stats = _init_stats()

    sums = {"total": 0.0, "pitch": 0.0, "onset": 0.0, "offset": 0.0, "hand": 0.0, "clef": 0.0}
    n_batches = 0

    # metric accumulators
    loose_counts = _init_eval_metric_counts()
    strict_counts = _init_eval_metric_counts()
    legacy_counts = {
        "onset_f1": 0.0,
        "offset_f1": 0.0,
        "onset_pred_rate": 0.0,
        "offset_pred_rate": 0.0,
        "n_on": 0,
        "n_off": 0,
        "metric_n": 0,
    }
    metric_n = 0
    valid_clip_counter = 0

    def _resolve_clip_label(batch_dict: Mapping[str, Any], idx: int) -> str:
        paths = batch_dict.get("path") if isinstance(batch_dict, Mapping) else None
        if isinstance(paths, (list, tuple)) and idx < len(paths):
            return str(paths[idx])
        for key in ("clip_id", "clip_ids", "video_id", "video_ids"):
            if not isinstance(batch_dict, Mapping):
                break
            value = batch_dict.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                if value.ndim == 0:
                    items = [value.item()]
                else:
                    items = value.tolist()
            else:
                items = value
            if isinstance(items, (list, tuple)) and idx < len(items):
                return str(items[idx])
        return f"idx={idx}"

    def _reason_to_label(reasons: set[str]) -> str:
        if any("nonfinite" in r for r in reasons):
            return "non-finite preds"
        if any(any(term in r for term in ("missing", "shape", "empty")) for r in reasons):
            return "invalid preds"
        return "invalid preds"

    def _validate_onoff_outputs(
        onset_logits: Optional[torch.Tensor],
        offset_logits: Optional[torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, List[set[str]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        valid_mask = torch.ones(batch_size, dtype=torch.bool)
        invalid_reasons: List[set[str]] = [set() for _ in range(batch_size)]

        def _mark_all(reason: str) -> None:
            if batch_size == 0:
                return
            valid_mask[:] = False
            for slots in invalid_reasons:
                slots.add(reason)

        def _check_tensor(name: str, tensor: Optional[torch.Tensor]) -> bool:
            if not torch.is_tensor(tensor):
                _mark_all(f"{name}:missing")
                return False
            if tensor.ndim == 0:
                _mark_all(f"{name}:empty")
                return False
            if tensor.shape[0] != batch_size:
                _mark_all(f"{name}:shape")
                return False
            reshaped = tensor.reshape(batch_size, -1)
            if reshaped.shape[1] == 0:
                _mark_all(f"{name}:empty")
                return False
            finite_mask = torch.isfinite(reshaped).all(dim=1)
            finite_list = finite_mask.tolist()
            for idx, ok in enumerate(finite_list):
                if not ok:
                    invalid_reasons[idx].add(f"{name}:nonfinite")
                    valid_mask[idx] = False
            return True

        onset_probs: Optional[torch.Tensor] = None
        offset_probs: Optional[torch.Tensor] = None
        onset_ok = _check_tensor("onset logits", onset_logits)
        offset_ok = _check_tensor("offset logits", offset_logits)

        if onset_ok and torch.is_tensor(onset_logits):
            onset_probs = _apply_sigmoid_calibration(
                onset_logits.detach(),
                temperature=onset_cal["temperature"],
                bias=onset_cal["bias"],
            )
            _check_tensor("onset probs", onset_probs)
        if offset_ok and torch.is_tensor(offset_logits):
            offset_probs = _apply_sigmoid_calibration(
                offset_logits.detach(),
                temperature=offset_cal["temperature"],
                bias=offset_cal["bias"],
            )
            _check_tensor("offset probs", offset_probs)

        return valid_mask, invalid_reasons, onset_probs, offset_probs

    want = ("pitch", "onset", "offset", "hand", "clef")
    class _StopEvalLoop(Exception):
        pass
    try:
        with torch.inference_mode():
            for batch in loader:
                if batch is None:
                    if not skip_empty_logged:
                        print("[train:eval] skip empty batch (all invalid)", flush=True)
                        skip_empty_logged = True
                    continue

                elapsed = perf_counter() - eval_start
                if timeout_seconds > 0 and elapsed > timeout_seconds:
                    timed_out = True
                    break

                if "video" not in batch:
                    raise KeyError("Eval batch missing 'video' tensor.")

                x = batch["video"]
                if torch.is_tensor(x):
                    batch_size = int(x.shape[0])
                else:
                    batch_size = len(batch.get("path", []))

                if warmup_start_time is None:
                    warmup_start_time = perf_counter()
                warmup_processed += int(batch_size)

                progress["count"] += int(batch_size)
                if cap_active > 0:
                    progress["count"] = min(progress["count"], cap_active)
                    if progress["count"] >= cap_active:
                        if warmup_elapsed is None and warmup_start_time is not None:
                            warmup_elapsed = perf_counter() - warmup_start_time
                        raise _StopEvalLoop()

                if (
                    warmup_elapsed is None
                    and warmup_target > 0
                    and warmup_processed >= warmup_target
                ):
                    warmup_elapsed = max(
                        (perf_counter() - warmup_start_time) if warmup_start_time is not None else 0.0,
                        1e-6,
                    )
                    throughput_live = warmup_processed / warmup_elapsed
                    if verbosity != "quiet":
                        logger.info(
                            "[train:eval] warmup throughput=%.2f win/s (%d in %.1fs)",
                            throughput_live,
                            warmup_processed,
                            warmup_elapsed,
                        )
                    warmup_logged = True

                    cap_payload = _compute_eval_cap(total_windows, pos_per_window, throughput_live)
                    new_cap = min(cap_payload["cap"], len(stratified_indices))
                    old_cap = cap_active
                    cap_driver_active = cap_payload["driver"]
                    cap_time_active = cap_payload["cap_time"]

                    if throughput_used is None:
                        cap_active = new_cap
                    else:
                        if throughput_used > 0:
                            delta_pct = (throughput_live - throughput_used) / throughput_used * 100.0
                        else:
                            delta_pct = None
                        if delta_pct is not None and abs(delta_pct) > INNER_EVAL_CACHE_TOLERANCE * 100:
                            cap_active = min(new_cap, cap_active)
                            if verbosity != "quiet" and cap_active != old_cap:
                                logger.info(
                                    "[train:eval] adjusted cap from %d→%d (live throughput %+d%%)",
                                    old_cap,
                                    cap_active,
                                    int(round(delta_pct)),
                                )
                        # retain original cap when difference within tolerance

                    throughput_used = throughput_live

                    cap_active = max(0, min(cap_active, len(stratified_indices)))
                    total_clips = cap_active
                    progress["total"] = cap_active
                    progress["count"] = min(progress["count"], cap_active)

                    if (
                        throughput_used
                        and throughput_used > 0
                        and cap_time_active is not None
                        and cap_pos > cap_time_active
                    ):
                        orig_budget = eval_wall_budget
                        budget_needed_live = int(
                            math.ceil(cap_pos / max(throughput_used, 1e-6) * INNER_EVAL_SAFETY)
                        )
                        if budget_needed_live > orig_budget:
                            if budget_needed_live > INNER_EVAL_BUDGET_MAX:
                                logger.warning(
                                    "[train:eval] requested budget %ds exceeds cap %ds; clamped. F1 variance may be high.",
                                    budget_needed_live,
                                    INNER_EVAL_BUDGET_MAX,
                                )
                                budget_needed_live = INNER_EVAL_BUDGET_MAX
                            logger.warning(
                                "[train:eval] increasing timeout to %ds (was %ds) to satisfy N_POS_MIN=%d; pos_per_window=%.2f, throughput=%.2f win/s",
                                budget_needed_live,
                                orig_budget,
                                INNER_EVAL_N_POS_MIN,
                                pos_per_window,
                                throughput_used,
                            )
                            eval_wall_budget = budget_needed_live
                            timeout_seconds = float(eval_wall_budget)

                    if not cap_logged:
                        cap_time_repr = (
                            str(cap_time_active) if cap_time_active is not None else "inf"
                        )
                        logger.info(
                            "[train:eval] cap=%d/%d (driver=%s)  cap_time=%s  cap_pos=%d",
                            cap_active,
                            total_windows,
                            cap_driver_active,
                            cap_time_repr,
                            cap_pos,
                        )
                        cap_logged = True

                have_all = all(k in batch for k in want)
                use_dummy = bool(cfg["training"].get("debug_dummy_labels", False))
                raw_tgt = {k: batch[k] for k in want} if have_all and not use_dummy else None

                out = model(x)

                onset_logits_full = out.get("onset_logits")
                offset_logits_full = out.get("offset_logits")
                valid_mask, invalid_reasons, onset_probs_full, offset_probs_full = _validate_onoff_outputs(
                    onset_logits_full,
                    offset_logits_full,
                    batch_size,
                )

                valid_count = int(valid_mask.sum().item())
                if valid_count < batch_size:
                    valid_mask_list = valid_mask.tolist()
                    for idx, is_valid in enumerate(valid_mask_list):
                        if is_valid:
                            continue
                        clip_label = _resolve_clip_label(batch, idx)
                        reasons = invalid_reasons[idx]
                        reason_label = _reason_to_label(reasons)
                        detail_suffix = f" (details: {', '.join(sorted(reasons))})" if reasons else ""
                        warn_msg = f"[eval WARNING] {reason_label} for clip {clip_label}, skipping from metrics{detail_suffix}"
                        print(warn_msg, flush=True)
                        logger.warning(warn_msg)
                if valid_count == 0:
                    continue

                valid_clip_counter += valid_count
                valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)
                valid_indices_list = valid_indices.tolist()
                index_cache: Dict[torch.device, torch.Tensor] = {}

                def _select_valid(value: Any):
                    if value is None:
                        return None
                    if torch.is_tensor(value):
                        if value.ndim == 0:
                            return value
                        if value.shape[0] != batch_size or len(valid_indices_list) == batch_size:
                            return value
                        device = value.device
                        if device not in index_cache:
                            index_cache[device] = valid_indices.to(device)
                        return value.index_select(0, index_cache[device])
                    if isinstance(value, list):
                        if len(value) != batch_size or len(valid_indices_list) == batch_size:
                            return value
                        return [value[i] for i in valid_indices_list]
                    if isinstance(value, tuple):
                        if len(value) != batch_size or len(valid_indices_list) == batch_size:
                            return value
                        return tuple(value[i] for i in valid_indices_list)
                    return value

                onset_probs_valid = _select_valid(onset_probs_full)
                offset_probs_valid = _select_valid(offset_probs_full)

                filtered_out: Dict[str, Any] = {}
                for key, value in out.items():
                    if torch.is_tensor(value):
                        filtered_out[key] = _select_valid(value)
                    else:
                        filtered_out[key] = value
                out = filtered_out

                if have_all and not use_dummy and raw_tgt is not None:
                    tgt = {}
                    for key in want:
                        val = _select_valid(raw_tgt[key])
                        if torch.is_tensor(val):
                            if key in ("pitch", "onset", "offset"):
                                val = val.float()
                            elif key in ("hand", "clef"):
                                val = val.long()
                        tgt[key] = val
                else:
                    tgt = fabricate_dummy_targets(valid_count)

                use_frame = (
                    getattr(model, "head_mode", "clip") == "frame"
                    and all(k in batch for k in ("pitch_roll", "onset_roll", "offset_roll", "hand_frame", "clef_frame"))
                )

                frame_batch = None
                if use_frame:
                    frame_batch = {
                        "pitch_roll": _select_valid(batch["pitch_roll"]).float(),
                        "onset_roll": _select_valid(batch["onset_roll"]).float(),
                        "offset_roll": _select_valid(batch["offset_roll"]).float(),
                        "hand_frame": _select_valid(batch["hand_frame"]).long(),
                        "clef_frame": _select_valid(batch["clef_frame"]).long(),
                    }
                    loss, parts = compute_loss_frame(out, frame_batch, weights=w)
                else:
                    if out["pitch_logits"].dim() == 3:
                        out = _time_pool_out_to_clip(out)
                        onset_probs_valid = None
                        offset_probs_valid = None
                    loss, parts = compute_loss(out, tgt, crit, w)

                for k in sums:
                    sums[k] += parts[k]
                n_batches += 1

                lag_ms_filtered = _select_valid(batch.get("lag_ms"))
                if lag_ms_filtered is not None:
                    if torch.is_tensor(lag_ms_filtered):
                        lag_iter = lag_ms_filtered.detach().cpu().reshape(-1).tolist()
                    else:
                        lag_iter = list(lag_ms_filtered)
                    for lag_val in lag_iter:
                        if lag_val is None:
                            continue
                        try:
                            lag_ms_values.append(float(lag_val))
                        except (TypeError, ValueError):
                            continue

                lag_flags_raw = batch.get("lag_flags")
                lag_flags_filtered = _select_valid(lag_flags_raw) if lag_flags_raw is not None else None
                for flags in (lag_flags_filtered or []):
                    if not flags:
                        continue
                    if any(flag == "low_corr_zero" for flag in flags):
                        lag_low_corr += 1
                    if any(flag == "lag_timeout" for flag in flags):
                        lag_timeouts += 1

                if DEBUG_EVAL_METRICS and valid_count > 0:
                    onset_logits = out.get("onset_logits")
                    offset_logits = out.get("offset_logits")
                    if onset_logits is not None and onset_logit_stats is not None and onset_prob_stats is not None:
                        _update_stats(onset_logit_stats, onset_logits)
                        onset_probs_stats = onset_probs_valid
                        if onset_probs_stats is None and onset_logits is not None:
                            onset_probs_stats = _apply_sigmoid_calibration(
                                onset_logits,
                                temperature=onset_cal["temperature"],
                                bias=onset_cal["bias"],
                            )
                        _update_stats(onset_prob_stats, onset_probs_stats)
                        if onset_k_effective is None:
                            onset_dim = onset_logits.shape[-1] if onset_logits.ndim > 0 else 1
                            onset_k_effective = max(1, min(int(agg_cfg["k_onset"]), onset_dim))
                    if offset_logits is not None and offset_logit_stats is not None and offset_prob_stats is not None:
                        _update_stats(offset_logit_stats, offset_logits)
                        offset_probs_stats = offset_probs_valid
                        if offset_probs_stats is None and offset_logits is not None:
                            offset_probs_stats = _apply_sigmoid_calibration(
                                offset_logits,
                                temperature=offset_cal["temperature"],
                                bias=offset_cal["bias"],
                            )
                        _update_stats(offset_prob_stats, offset_probs_stats)
                        if offset_k_effective is None:
                            offset_dim = offset_logits.shape[-1] if offset_logits.ndim > 0 else 1
                            offset_k_effective = max(1, min(int(agg_cfg["k_offset"]), offset_dim))

                # --- metrics ---
                if use_frame:
                    # --- align frame targets to logits time (T -> T_logits) ---
                    B, T_logits, P = out["onset_logits"].shape

                    onset_roll = frame_batch["onset_roll"]
                    offset_roll = frame_batch["offset_roll"]
                    hand_frame = frame_batch["hand_frame"]
                    clef_frame = frame_batch["clef_frame"]

                    if onset_roll.shape[1] != T_logits:
                        def _pool_bool_BT(x_btP, Tprime):
                            # (B,T,P) -> (B,T',P), preserving "any positive in window"
                            x = x_btP.permute(0, 2, 1)
                            x = F.adaptive_max_pool1d(x, Tprime)
                            return x.permute(0, 2, 1).contiguous()

                        def _interp_labels_BT(x_bt, Tprime):
                            x = x_bt.float().unsqueeze(1)
                            x = F.interpolate(x, size=Tprime, mode="nearest")
                            return x.squeeze(1).long()

                        onset_roll = _pool_bool_BT(onset_roll, T_logits)
                        offset_roll = _pool_bool_BT(offset_roll, T_logits)
                        hand_frame = _interp_labels_BT(hand_frame, T_logits)
                        clef_frame = _interp_labels_BT(clef_frame, T_logits)

                    onset_any = (onset_roll > 0).any(dim=-1).float()
                    offset_any = (offset_roll > 0).any(dim=-1).float()

                    loose_onset_pred, loose_onset_counts = _aggregate_onoff_predictions(
                        out["onset_logits"],
                        thr_on,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_onset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                    )
                    loose_offset_pred, loose_offset_counts = _aggregate_onoff_predictions(
                        out["offset_logits"],
                        thr_off,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_offset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=offset_cal["temperature"],
                        bias=offset_cal["bias"],
                    )
                    strict_onset_pred, strict_onset_counts, strict_onset_mask = _aggregate_onoff_predictions(
                        out["onset_logits"],
                        thr_on,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_onset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                        cap_count=agg_cfg["k_onset"],
                        return_mask=True,
                    )
                    strict_offset_pred, strict_offset_counts, strict_offset_mask = _aggregate_onoff_predictions(
                        out["offset_logits"],
                        thr_off,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_offset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=offset_cal["temperature"],
                        bias=offset_cal["bias"],
                        cap_count=agg_cfg["k_offset"],
                        return_mask=True,
                    )
                    loose_onset_pred = loose_onset_pred.float()
                    loose_offset_pred = loose_offset_pred.float()
                    strict_onset_pred = strict_onset_pred.float()
                    strict_offset_pred = strict_offset_pred.float()
                    strict_onset_mask_float = strict_onset_mask.float()
                    strict_offset_mask_float = strict_offset_mask.float()

                    onset_pred_legacy, _ = _aggregate_onoff_predictions(
                        out["onset_logits"],
                        thr_on,
                        mode="any",
                        k=1,
                        top_k=0,
                        tau_sum=0.0,
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                    )
                    offset_pred_legacy, _ = _aggregate_onoff_predictions(
                        out["offset_logits"],
                        thr_off,
                        mode="any",
                        k=1,
                        top_k=0,
                        tau_sum=0.0,
                        temperature=offset_cal["temperature"],
                        bias=offset_cal["bias"],
                    )
                    onset_pred_legacy = onset_pred_legacy.float()
                    offset_pred_legacy = offset_pred_legacy.float()

                    loose_counts["onset_key_sum"] += loose_onset_counts.sum().item()
                    loose_counts["offset_key_sum"] += loose_offset_counts.sum().item()
                    strict_counts["onset_key_sum"] += strict_onset_counts.sum().item()
                    strict_counts["offset_key_sum"] += strict_offset_counts.sum().item()
                    loose_counts["onset_frame_count"] += loose_onset_counts.numel()
                    loose_counts["offset_frame_count"] += loose_offset_counts.numel()
                    strict_counts["onset_frame_count"] += strict_onset_counts.numel()
                    strict_counts["offset_frame_count"] += strict_offset_counts.numel()
                    _accumulate_pred_key_histogram(loose_counts["onset_hist"], loose_onset_counts)
                    _accumulate_pred_key_histogram(loose_counts["offset_hist"], loose_offset_counts)
                    _accumulate_pred_key_histogram(strict_counts["onset_hist"], strict_onset_counts)
                    _accumulate_pred_key_histogram(strict_counts["offset_hist"], strict_offset_counts)

                    onset_gt_mask = (onset_roll > 0).float()
                    offset_gt_mask = (offset_roll > 0).float()
                    onset_gt_mask = _align_key_mask_to(strict_onset_mask_float, onset_gt_mask)
                    offset_gt_mask = _align_key_mask_to(strict_offset_mask_float, offset_gt_mask)

                    f1_on_loose = _binary_f1(loose_onset_pred.reshape(-1), onset_any.reshape(-1))
                    f1_off_loose = _binary_f1(loose_offset_pred.reshape(-1), offset_any.reshape(-1))
                    f1_on_strict = _binary_f1(strict_onset_mask_float.reshape(-1), onset_gt_mask.reshape(-1))
                    f1_off_strict = _binary_f1(strict_offset_mask_float.reshape(-1), offset_gt_mask.reshape(-1))
                    f1_on_legacy = _binary_f1(onset_pred_legacy.reshape(-1), onset_any.reshape(-1))
                    f1_off_legacy = _binary_f1(offset_pred_legacy.reshape(-1), offset_any.reshape(-1))
                    if f1_on_loose is not None:
                        loose_counts["onset_f1"] += f1_on_loose
                        loose_counts["n_on"] += 1
                    if f1_off_loose is not None:
                        loose_counts["offset_f1"] += f1_off_loose
                        loose_counts["n_off"] += 1
                    if f1_on_strict is not None:
                        strict_counts["onset_f1"] += f1_on_strict
                        strict_counts["n_on"] += 1
                    if f1_off_strict is not None:
                        strict_counts["offset_f1"] += f1_off_strict
                        strict_counts["n_off"] += 1
                    if f1_on_legacy is not None:
                        legacy_counts["onset_f1"] += f1_on_legacy
                        legacy_counts["n_on"] += 1
                    if f1_off_legacy is not None:
                        legacy_counts["offset_f1"] += f1_off_legacy
                        legacy_counts["n_off"] += 1

                    onset_pos_rate = onset_any.mean().item()
                    offset_pos_rate = offset_any.mean().item()
                    loose_counts["onset_pos_rate"] += onset_pos_rate
                    loose_counts["offset_pos_rate"] += offset_pos_rate
                    strict_counts["onset_pos_rate"] += onset_gt_mask.mean().item()
                    strict_counts["offset_pos_rate"] += offset_gt_mask.mean().item()
                    loose_counts["onset_pred_rate"] += loose_onset_pred.mean().item()
                    loose_counts["offset_pred_rate"] += loose_offset_pred.mean().item()
                    strict_counts["onset_pred_rate"] += strict_onset_mask_float.mean().item()
                    strict_counts["offset_pred_rate"] += strict_offset_mask_float.mean().item()
                    legacy_counts["onset_pred_rate"] += onset_pred_legacy.mean().item()
                    legacy_counts["offset_pred_rate"] += offset_pred_legacy.mean().item()
                    legacy_counts["metric_n"] += 1

                    hand_prob = F.softmax(out["hand_logits"], dim=-1)
                    clef_prob = F.softmax(out["clef_logits"], dim=-1)
                    hand_pred = hand_prob.argmax(dim=-1)
                    clef_pred = clef_prob.argmax(dim=-1)
                    Bx, Tx = hand_pred.shape
                    hand_acc_val = (hand_pred.reshape(Bx * Tx) == hand_frame.reshape(Bx * Tx)).float().mean().item()
                    clef_acc_val = (clef_pred.reshape(Bx * Tx) == clef_frame.reshape(Bx * Tx)).float().mean().item()
                    loose_counts["hand_acc"] += hand_acc_val
                    loose_counts["clef_acc"] += clef_acc_val
                    strict_counts["hand_acc"] += hand_acc_val
                    strict_counts["clef_acc"] += clef_acc_val

                    metric_n += 1


                else:
                    pitch_pred = (torch.sigmoid(out["pitch_logits"]) >= thr_pitch).float()
                    pitch_gt = (tgt["pitch"] >= 0.5).float()
                    f1_pitch = _binary_f1(pitch_pred.reshape(-1), pitch_gt.reshape(-1))
                    if f1_pitch is not None:
                        loose_counts["pitch_acc"] += f1_pitch
                        strict_counts["pitch_acc"] += f1_pitch

                    hand_pred = F.softmax(out["hand_logits"], dim=-1).argmax(dim=-1)
                    clef_pred = F.softmax(out["clef_logits"], dim=-1).argmax(dim=-1)
                    hand_acc_val = (hand_pred == tgt["hand"]).float().mean().item()
                    clef_acc_val = (clef_pred == tgt["clef"]).float().mean().item()
                    loose_counts["hand_acc"] += hand_acc_val
                    loose_counts["clef_acc"] += clef_acc_val
                    strict_counts["hand_acc"] += hand_acc_val
                    strict_counts["clef_acc"] += clef_acc_val

                    loose_onset_pred, _ = _aggregate_onoff_predictions(
                        out["onset_logits"],
                        thr_on,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_onset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                    )
                    loose_offset_pred, _ = _aggregate_onoff_predictions(
                        out["offset_logits"],
                        thr_off,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_offset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=offset_cal["temperature"],
                        bias=offset_cal["bias"],
                    )
                    strict_onset_pred, _, strict_onset_mask = _aggregate_onoff_predictions(
                        out["onset_logits"],
                        thr_on,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_onset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                        cap_count=agg_cfg["k_onset"],
                        return_mask=True,
                    )
                    strict_offset_pred, _, strict_offset_mask = _aggregate_onoff_predictions(
                        out["offset_logits"],
                        thr_off,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_offset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=offset_cal["temperature"],
                        bias=offset_cal["bias"],
                        cap_count=agg_cfg["k_offset"],
                        return_mask=True,
                    )
                    loose_onset_pred = loose_onset_pred.float()
                    loose_offset_pred = loose_offset_pred.float()
                    strict_onset_pred = strict_onset_pred.float()
                    strict_offset_pred = strict_offset_pred.float()
                    strict_onset_mask_float = strict_onset_mask.float()
                    strict_offset_mask_float = strict_offset_mask.float()

                    onset_pred_legacy, _ = _aggregate_onoff_predictions(
                        out["onset_logits"],
                        thr_on,
                        mode="any",
                        k=1,
                        top_k=0,
                        tau_sum=0.0,
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                    )
                    offset_pred_legacy, _ = _aggregate_onoff_predictions(
                        out["offset_logits"],
                        thr_off,
                        mode="any",
                        k=1,
                        top_k=0,
                        tau_sum=0.0,
                        temperature=offset_cal["temperature"],
                        bias=offset_cal["bias"],
                    )
                    onset_pred_legacy = onset_pred_legacy.float()
                    offset_pred_legacy = offset_pred_legacy.float()

                    onset_gt = (tgt["onset"] >= 0.5).float()
                    offset_gt = (tgt["offset"] >= 0.5).float()
                    onset_gt_bin = onset_gt.any(dim=-1)
                    offset_gt_bin = offset_gt.any(dim=-1)

                    onset_f1_loose = _binary_f1(loose_onset_pred, onset_gt_bin)
                    offset_f1_loose = _binary_f1(loose_offset_pred, offset_gt_bin)
                    onset_f1_strict = _binary_f1(strict_onset_mask_float.reshape(-1), onset_gt.reshape(-1))
                    offset_f1_strict = _binary_f1(strict_offset_mask_float.reshape(-1), offset_gt.reshape(-1))
                    onset_f1_legacy = _binary_f1(onset_pred_legacy, onset_gt_bin)
                    offset_f1_legacy = _binary_f1(offset_pred_legacy, offset_gt_bin)
                    if onset_f1_loose is not None:
                        loose_counts["onset_f1"] += onset_f1_loose
                        loose_counts["n_on"] += 1
                    if offset_f1_loose is not None:
                        loose_counts["offset_f1"] += offset_f1_loose
                        loose_counts["n_off"] += 1
                    if onset_f1_strict is not None:
                        strict_counts["onset_f1"] += onset_f1_strict
                        strict_counts["n_on"] += 1
                    if offset_f1_strict is not None:
                        strict_counts["offset_f1"] += offset_f1_strict
                        strict_counts["n_off"] += 1
                    if onset_f1_legacy is not None:
                        legacy_counts["onset_f1"] += onset_f1_legacy
                        legacy_counts["n_on"] += 1
                    if offset_f1_legacy is not None:
                        legacy_counts["offset_f1"] += offset_f1_legacy
                        legacy_counts["n_off"] += 1

                    onset_pos_rate = onset_gt_bin.mean().item()
                    offset_pos_rate = offset_gt_bin.mean().item()
                    loose_counts["onset_pos_rate"] += onset_pos_rate
                    loose_counts["offset_pos_rate"] += offset_pos_rate
                    strict_counts["onset_pos_rate"] += onset_gt.mean().item()
                    strict_counts["offset_pos_rate"] += offset_gt.mean().item()
                    loose_counts["onset_pred_rate"] += loose_onset_pred.mean().item()
                    loose_counts["offset_pred_rate"] += loose_offset_pred.mean().item()
                    strict_counts["onset_pred_rate"] += strict_onset_mask_float.mean().item()
                    strict_counts["offset_pred_rate"] += strict_offset_mask_float.mean().item()
                    metric_n += 1
        if warmup_elapsed is not None and not warmup_logged:
            throughput_live_fallback = warmup_processed / max(warmup_elapsed, 1e-6)
            if verbosity != "quiet":
                logger.info(
                    "[train:eval] warmup throughput=%.2f win/s (%d in %.1fs)",
                    throughput_live_fallback,
                    warmup_processed,
                    warmup_elapsed,
                )
            warmup_logged = True
            throughput_used = throughput_live_fallback
            cap_payload = _compute_eval_cap(
                total_windows,
                pos_per_window,
                throughput_live_fallback if throughput_live_fallback > 0 else None,
            )
            cap_driver_active = cap_payload["driver"]
            cap_time_active = cap_payload["cap_time"]
            cap_active = min(cap_active, cap_payload["cap"])
            total_clips = cap_active
            progress["total"] = cap_active
            progress["count"] = min(progress["count"], cap_active)
        if not cap_logged:
            cap_time_repr = str(cap_time_active) if cap_time_active is not None else "inf"
            logger.info(
                "[train:eval] cap=%d/%d (driver=%s)  cap_time=%s  cap_pos=%d",
                cap_active,
                total_windows,
                cap_driver_active,
                cap_time_repr,
                cap_pos,
            )
            cap_logged = True
    except _StopEvalLoop:
        pass
    finally:
        stop_event.set()
        _join_thread(heartbeat_thread)
        _join_thread(watchdog_thread)
        torch.set_num_threads(num_threads_prev)
        if interop_prev is not None and hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(int(interop_prev))
            except RuntimeError:
                pass
        if moved_to_cpu:
            model.to(original_device)
            _move_optimizer_state(optimizer, original_device)
        if prev_training:
            model.train()
        else:
            model.eval()

    elapsed_total = perf_counter() - eval_start
    print(f"[train:eval] done dt={elapsed_total:.1f}s", flush=True)

    sorted_lag = sorted(lag_ms_values)
    lag_mean = (sum(sorted_lag) / len(sorted_lag)) if sorted_lag else 0.0
    lag_median = _compute_percentile(sorted_lag, 50.0)
    lag_p95 = _compute_percentile(sorted_lag, 95.0)
    print(
        f"[train:eval] A/V lag ms: mean={lag_mean:.1f}, median={lag_median:.1f}, p95={lag_p95:.1f}, low_corr={lag_low_corr}, timeouts={lag_timeouts}",
        flush=True,
    )
    logger.info(
        "[train:eval] A/V lag ms: mean=%.1f median=%.1f p95=%.1f low_corr=%d timeouts=%d",
        lag_mean,
        lag_median,
        lag_p95,
        lag_low_corr,
        lag_timeouts,
    )

    if timed_out:
        timeout_label = int(round(timeout_seconds / 60.0)) if timeout_seconds > 0 else 0
        print(f"[train:eval] timeout after {timeout_label}m; skipping metrics", flush=True)
        logger.warning("[train:eval] timeout after %sm; skipping metrics", timeout_label)
        return None
    have_valid_metrics = valid_clip_counter > 0

    if DEBUG_EVAL_METRICS and onset_logit_stats is not None and offset_logit_stats is not None:
        onset_logit_min, onset_logit_mean, onset_logit_max = _finalize_stats(onset_logit_stats)
        offset_logit_min, offset_logit_mean, offset_logit_max = _finalize_stats(offset_logit_stats)
        onset_prob_min, onset_prob_mean, onset_prob_max = _finalize_stats(onset_prob_stats or _init_stats())
        offset_prob_min, offset_prob_mean, offset_prob_max = _finalize_stats(offset_prob_stats or _init_stats())
        onset_k_report = onset_k_effective if onset_k_effective is not None else int(agg_cfg["k_onset"])
        offset_k_report = offset_k_effective if offset_k_effective is not None else int(agg_cfg["k_offset"])

        if not math.isfinite(onset_logit_min):
            onset_logit_min = 0.0
        if not math.isfinite(onset_logit_mean):
            onset_logit_mean = 0.0
        if not math.isfinite(onset_logit_max):
            onset_logit_max = 0.0
        if not math.isfinite(offset_logit_min):
            offset_logit_min = 0.0
        if not math.isfinite(offset_logit_mean):
            offset_logit_mean = 0.0
        if not math.isfinite(offset_logit_max):
            offset_logit_max = 0.0
        if not math.isfinite(onset_prob_min):
            onset_prob_min = 0.0
        if not math.isfinite(onset_prob_mean):
            onset_prob_mean = 0.0
        if not math.isfinite(onset_prob_max):
            onset_prob_max = 0.0
        if not math.isfinite(offset_prob_min):
            offset_prob_min = 0.0
        if not math.isfinite(offset_prob_mean):
            offset_prob_mean = 0.0
        if not math.isfinite(offset_prob_max):
            offset_prob_max = 0.0

        print(
            f"[train:eval-debug] onset logits min={onset_logit_min:.4f} mean={onset_logit_mean:.4f} max={onset_logit_max:.4f} | "
            f"offset logits min={offset_logit_min:.4f} mean={offset_logit_mean:.4f} max={offset_logit_max:.4f}",
            flush=True,
        )
        print(
            f"[train:eval-debug] onset probs  min={onset_prob_min:.4f} mean={onset_prob_mean:.4f} max={onset_prob_max:.4f} | "
            f"offset probs  min={offset_prob_min:.4f} mean={offset_prob_mean:.4f} max={offset_prob_max:.4f}",
            flush=True,
        )
        print(
            f"[train:eval-debug] thresholds onset={thr_on:.3f} (k={onset_k_report}) offset={thr_off:.3f} (k={offset_k_report})",
            flush=True,
        )

    # averages
    losses = {k: sums[k] / max(1, n_batches) for k in sums}
    def _finalize_branch_counts(counts: Dict[str, Any]) -> Dict[str, float]:
        branch: Dict[str, float] = {}
        denom = max(1, metric_n)
        branch["pitch_acc"] = counts["pitch_acc"] / denom
        branch["hand_acc"] = counts["hand_acc"] / denom
        branch["clef_acc"] = counts["clef_acc"] / denom
        branch["onset_f1"] = counts["onset_f1"] / max(1, counts.get("n_on", 0))
        branch["offset_f1"] = counts["offset_f1"] / max(1, counts.get("n_off", 0))
        branch["onset_pos_rate"] = counts["onset_pos_rate"] / denom
        branch["offset_pos_rate"] = counts["offset_pos_rate"] / denom
        branch["onset_pred_rate"] = counts["onset_pred_rate"] / denom
        branch["offset_pred_rate"] = counts["offset_pred_rate"] / denom

        onset_frames = counts["onset_frame_count"]
        offset_frames = counts["offset_frame_count"]
        if onset_frames > 0:
            onset_hist = counts["onset_hist"]
            branch["mean_pred_keys_per_frame_onset"] = counts["onset_key_sum"] / onset_frames
            branch["pred_keys_hist_onset_0"] = onset_hist[0] / onset_frames
            branch["pred_keys_hist_onset_1"] = onset_hist[1] / onset_frames
            branch["pred_keys_hist_onset_2"] = onset_hist[2] / onset_frames
            branch["pred_keys_hist_onset_3"] = onset_hist[3] / onset_frames
            branch["pred_keys_hist_onset_ge4"] = onset_hist[4] / onset_frames
        else:
            branch["mean_pred_keys_per_frame_onset"] = 0.0
            branch["pred_keys_hist_onset_0"] = 0.0
            branch["pred_keys_hist_onset_1"] = 0.0
            branch["pred_keys_hist_onset_2"] = 0.0
            branch["pred_keys_hist_onset_3"] = 0.0
            branch["pred_keys_hist_onset_ge4"] = 0.0

        if offset_frames > 0:
            offset_hist = counts["offset_hist"]
            branch["mean_pred_keys_per_frame_offset"] = counts["offset_key_sum"] / offset_frames
            branch["pred_keys_hist_offset_0"] = offset_hist[0] / offset_frames
            branch["pred_keys_hist_offset_1"] = offset_hist[1] / offset_frames
            branch["pred_keys_hist_offset_2"] = offset_hist[2] / offset_frames
            branch["pred_keys_hist_offset_3"] = offset_hist[3] / offset_frames
            branch["pred_keys_hist_offset_ge4"] = offset_hist[4] / offset_frames
        else:
            branch["mean_pred_keys_per_frame_offset"] = 0.0
            branch["pred_keys_hist_offset_0"] = 0.0
            branch["pred_keys_hist_offset_1"] = 0.0
            branch["pred_keys_hist_offset_2"] = 0.0
            branch["pred_keys_hist_offset_3"] = 0.0
            branch["pred_keys_hist_offset_ge4"] = 0.0

        return branch

    loose_branch = _finalize_branch_counts(loose_counts)
    strict_branch = _finalize_branch_counts(strict_counts)

    metrics_any = dict(loose_branch)
    metrics_any["onset_f1"] = legacy_counts["onset_f1"] / max(1, legacy_counts.get("n_on", 0))
    metrics_any["offset_f1"] = legacy_counts["offset_f1"] / max(1, legacy_counts.get("n_off", 0))
    metrics_any["onset_pred_rate"] = legacy_counts["onset_pred_rate"] / max(1, legacy_counts.get("metric_n", 0))
    metrics_any["offset_pred_rate"] = legacy_counts["offset_pred_rate"] / max(1, legacy_counts.get("metric_n", 0))

    if not have_valid_metrics:
        warn_msg = "[eval WARNING] No valid clips to score after filtering. Metrics set to 0."
        print(warn_msg, flush=True)
        logger.warning(warn_msg)
        loose_branch = {k: 0.0 for k in loose_branch.keys()}
        strict_branch = {k: 0.0 for k in strict_branch.keys()}
        metrics_any = {k: 0.0 for k in metrics_any.keys()}

    loose_prefixed = {f"loose_{k}": v for k, v in loose_branch.items()}
    strict_prefixed = {f"strict_{k}": v for k, v in strict_branch.items()}

    val_metrics = {**losses, **loose_prefixed, **strict_prefixed}
    legacy_metrics = {**losses, **metrics_any}

    def _fmt_metrics_line(items: Mapping[str, float]):
        return " ".join(f"{k}={v:.3f}\t" for k, v in items.items())

    loose_line = _fmt_metrics_line({**losses, **loose_branch})
    strict_line = _fmt_metrics_line({**losses, **strict_branch})
    legacy_line = _fmt_metrics_line(legacy_metrics)
    print(f"[loose] {loose_line}\n")
    print(f"[strict] {strict_line}\n")
    print(f"[any] {legacy_line}\n")
    quiet_extra = {QUIET_INFO_FLAG: True}
    logger.info("[loose] %s", loose_line, extra=quiet_extra)
    logger.info("[strict] %s", strict_line, extra=quiet_extra)
    logger.info("[any] %s", legacy_line, extra=quiet_extra)

    processed_total = progress.get("count", 0)
    throughput_for_cache = throughput_used if throughput_used and throughput_used > 0 else processed_total / max(elapsed_total, 1e-6)
    if not math.isfinite(throughput_for_cache):
        throughput_for_cache = 0.0
    cache_payload = {
        "throughput_win_per_s": float(throughput_for_cache),
        "pos_per_window": float(pos_per_window),
        "updated": datetime.utcnow().isoformat(timespec="seconds"),
    }
    _store_inner_eval_cache(cache_payload)

    return val_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--train-split", choices=["train", "val", "test"], help="Dataset split for training")
    ap.add_argument("--val-split", choices=["train", "val", "test"], help="Validation split")
    ap.add_argument("--max-clips", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seed", type=int, help="Seed for RNGs and dataloader shuffling")
    ap.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Toggle deterministic torch backends (default: config or enabled)",
    )
    ap.add_argument(
        "--verbose",
        choices=["quiet", "info", "debug"],
        help="Logging verbosity (default: quiet or $TIVIT_VERBOSE)",
    )
    ap.add_argument("--smoke", action="store_true", help="Run a quick synthetic pass")
    args = ap.parse_args()

    args.verbose = configure_verbosity(args.verbose)

    cfg = dict(load_config(args.config))
    seed = resolve_seed(args.seed, cfg)
    deterministic = resolve_deterministic_flag(args.deterministic, cfg)
    exp_cfg = cfg.setdefault("experiment", {})
    exp_cfg["seed"] = seed
    exp_cfg["deterministic"] = deterministic

    configure_determinism(seed, deterministic)
    logger.info(
        "[determinism] seed=%d deterministic=%s",
        seed,
        "on" if deterministic else "off",
        extra={QUIET_INFO_FLAG: True},
    )

    faulthandler.enable()
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except (AttributeError, RuntimeError, ValueError, OSError):
        pass
    if args.max_clips is not None:
        cfg["dataset"]["max_clips"] = args.max_clips
    if args.frames is not None:
        cfg["dataset"]["frames"] = args.frames
        
    if args.smoke:
        logger.info("Running smoke test")
        model = build_model(cfg)
        x = torch.randn(1, 4, 3, 224, 224)
        out = model(x)
        logger.info("Smoke forward keys: %s", list(out.keys()))
        return
        
    freeze_epochs = int(cfg.get("train", {}).get("freeze_backbone_epochs", 0))
    ckpt_dir, log_root = ensure_dirs(cfg)

    # Build experiment-specific log dir
    exp_name = cfg.get("experiment", {}).get("name", "default")
    log_dir = log_root / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    log_cfg = cfg.get("logging", {})
    use_tb = bool(log_cfg.get("tensorboard", False))
    writer = SummaryWriter(log_dir) if use_tb else None
    
    # Data
    train_split = args.train_split or cfg["dataset"].get("split_train") or cfg["dataset"].get("split") or "train"
    val_split = args.val_split or cfg["dataset"].get("split_val") or cfg["dataset"].get("split") or "val"
    train_loader = make_dataloader(cfg, split=train_split, seed=seed)
    train_base_dataset = _unwrap_dataset(getattr(train_loader, "dataset", train_loader))

    # If you have a dedicated val split, use it; otherwise reuse "test" as a stand-in.
    val_loader = None
    if cfg["training"].get("eval_freq", 0):
        # try to build test or val loader
        try:
            val_loader = make_dataloader(cfg, split=val_split, seed=seed)
        except Exception:
            try:
                test_split = cfg["dataset"].get("split_test") or cfg["dataset"].get("split") or "test"
                val_loader = make_dataloader(cfg, split=test_split, seed=seed)
            except Exception:
                val_loader = None

    if val_loader is not None:
        val_loader = _prepare_inner_eval_loader(cfg, val_loader)
    
    # Model & optimizer
    model = build_model(cfg)
    
    # --- diag: baseline random prediction stats right after init (removable) ---
    _baseline_batch = None
    try:
        _baseline_batch = next(iter(train_loader))
    except StopIteration:
        logger.warning("No batches available for baseline prediction check.")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Could not grab batch for baseline prediction check: %s", exc)
    if _baseline_batch is not None and isinstance(_baseline_batch, dict) and "video" in _baseline_batch:
        video = _baseline_batch["video"]
        model_device = next(model.parameters()).device
        dtype = video.dtype if torch.is_floating_point(video) else torch.float32
        rand_input = torch.rand(video.shape, device=model_device, dtype=dtype)
        model_mode = model.training
        model.eval()
        with torch.no_grad():
            init_out = model(rand_input)
        init_stats = {
            "onset": _prediction_stats_from_logits(init_out.get("onset_logits")),
            "offset": _prediction_stats_from_logits(init_out.get("offset_logits")),
        }
        _log_prediction_stats("init_random", init_stats, writer=writer, step=0)
        if model_mode:
            model.train()
        del rand_input, init_out
    _baseline_batch = None  
    # --- End of diag: baseline random prediction stats right after init (removable) ---
    
    def build_optimizer(model, optim_cfg):
        base_lr = float(optim_cfg["learning_rate"])
        wd = float(optim_cfg.get("weight_decay", 0.0))
        head_lr_mult = float(optim_cfg.get("head_lr_multiplier", 5.0))
        onset_lr_mult = float(optim_cfg.get("onset_lr_multiplier", head_lr_mult))
        offset_lr_mult = float(optim_cfg.get("offset_lr_multiplier", head_lr_mult))
        onset_wd = float(optim_cfg.get("onset_weight_decay", wd * float(optim_cfg.get("onset_weight_decay_multiplier", 1.0))))
        offset_wd = float(optim_cfg.get("offset_weight_decay", wd * float(optim_cfg.get("offset_weight_decay_multiplier", 1.0))))

        base_params: list[torch.nn.Parameter] = []
        onset_params: list[torch.nn.Parameter] = []
        offset_params: list[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lname = name.lower()
            if "offset" in lname:
                offset_params.append(p)
            elif "onset" in lname:
                onset_params.append(p)
            else:
                base_params.append(p)

        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": base_lr, "weight_decay": wd})
        if onset_params:
            param_groups.append({
                "params": onset_params,
                "lr": base_lr * onset_lr_mult,
                "weight_decay": onset_wd,
            })
        if offset_params:
            param_groups.append({
                "params": offset_params,
                "lr": base_lr * offset_lr_mult,
                "weight_decay": offset_wd,
            })

        if not param_groups:
            raise RuntimeError("No trainable parameters found for optimizer setup")

        opt = torch.optim.AdamW(param_groups)

        num_base = sum(p.numel() for p in base_params)
        num_onset = sum(p.numel() for p in onset_params)
        num_offset = sum(p.numel() for p in offset_params)
        logger.info(
            "[OPT] base params: %s @ lr=%.2e wd=%.2e | onset: %s @ lr=%.2e wd=%.2e | offset: %s @ lr=%.2e wd=%.2e",
            f"{num_base:,}",
            base_lr,
            wd,
            f"{num_onset:,}",
            base_lr * onset_lr_mult,
            onset_wd,
            f"{num_offset:,}",
            base_lr * offset_lr_mult,
            offset_wd,
        )
        return opt

    optimizer = build_optimizer(model, cfg["training"])

    accum_steps = int(cfg.get("train", {}).get("accumulate_steps", 1))
    grad_clip = float(cfg["optim"].get("grad_clip", 1.0))
    freeze_backbone_epochs = int(cfg.get("train", {}).get("freeze_backbone_epochs", 0))
    
    # Train
    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"].get("save_every", 1))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    best_path = ckpt_dir / "tivit_best.pt"
    last_path = ckpt_dir / "tivit_last.pt"
    eval_freq = int(cfg["training"].get("eval_freq", 0))

    # Try to resume from checkpoint
    resume_path = ckpt_dir / "tivit_best.pt"
    want_resume = bool(cfg.get("training", {}).get("resume", False))
    if resume_path.exists() and want_resume:
        print(f"[resume] Loading from {resume_path}")
        logger.info("[resume] Loading from %s", resume_path)
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        best_val = ckpt.get("best_val", math.inf)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[resume] optimizer groups mismatch; skipping optimizer state. ({e})")
            logger.warning("[resume] optimizer groups mismatch; skipping optimizer state. (%s)", e)
            # scheduler will be re-created fresh below
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    else:
        if resume_path.exists() and not want_resume:
            print(f"[resume] Found checkpoint at {resume_path} but resume disabled; starting fresh")
            logger.info("[resume] Found checkpoint at %s but resume disabled; starting fresh", resume_path)
        if not resume_path.exists() and want_resume:
            print(f"[resume] Requested resume but checkpoint missing at {resume_path}; starting fresh")
            logger.info("[resume] Requested resume but checkpoint missing at %s; starting fresh", resume_path)
        # Fresh init → apply onset/offset bias if requested
        if cfg.get("training", {}).get("reset_head_bias", True):
            training_cfg = cfg.get("training", {})
            bias_cfg = training_cfg.get("bias_seed", {})
            prior_cfg = float(bias_cfg.get("onoff_prior_mean", 0.02))
            _set_onoff_head_bias(model, prior=prior_cfg)
        best_val = math.inf
        start_epoch = 1
        
    def _freeze_backbone_keep_heads(model):
        for _, p in model.named_parameters():
            p.requires_grad = False
        # Unfreeze any modules that look like heads. This covers
        # architectures that expose multiple head_* attributes (e.g.
        # head_pitch, head_onset, head_offset, ...), as well as single
        # "head" or "heads" containers.  We look through immediate
        # children and re-enable gradients for those whose name
        # contains "head".
        for name, module in model.named_children():
            if "head" in name.lower():
                for p in module.parameters():
                    p.requires_grad = True

    def _unfreeze_backbone(model):
        for _, p in model.named_parameters():
            p.requires_grad = True

    if freeze_epochs > 0 and start_epoch <= freeze_epochs:
        _freeze_backbone_keep_heads(model)
        optimizer = build_optimizer(model, cfg["training"])
        n_trainable = sum(p.requires_grad for p in model.parameters())
        print(f"[warmup] trainable params: {n_trainable}")
        logger.info("[warmup] trainable params: %s", n_trainable)

    for epoch in range(start_epoch, epochs + 1):
        if freeze_epochs and epoch == freeze_epochs:
            _unfreeze_backbone(model)
            optimizer = build_optimizer(model, cfg["training"])
            n_trainable = sum(p.requires_grad for p in model.parameters())
            print(f"[warmup] trainable params: {n_trainable}")    
        t0 = perf_counter()
        epoch_prepare = getattr(train_base_dataset, "prepare_epoch_snapshot", None)
        if callable(epoch_prepare):
            try:
                epoch_prepare(shuffle=True, audit=True)
            except Exception as exc:
                logger.warning("[train] failed to prepare epoch snapshot: %s", exc)
        # --- train one epoch ---
        model.train()
        crit = make_criterions()
        w = cfg["training"]["loss_weights"]
        #w = get_loss_weights(cfg) if "get_loss_weights" in globals() else cfg["training"]["loss_weights"] #get_loss_weights not defined in this file
        log_interval = int(cfg["training"].get("log_interval", 20))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", ncols=100)
        running = {"total": 0.0, "pitch": 0.0, "onset": 0.0, "offset": 0.0, "hand": 0.0, "clef": 0.0}
        count = 0
        avg_events_recent = None  # for TB data stat

        # --- diag accumulators for onset/offset prediction variance ---
        pred_stat_accum = {
            "onset": {"mean_sum": 0.0, "std_sum": 0.0, "min_sum": 0.0, "max_sum": 0.0, "min_all": math.inf, "max_all": -math.inf, "count": 0},
            "offset": {"mean_sum": 0.0, "std_sum": 0.0, "min_sum": 0.0, "max_sum": 0.0, "min_all": math.inf, "max_all": -math.inf, "count": 0},
        }
        low_std_streak = {"onset": 0, "offset": 0}
        low_std_patience = int(cfg.get("training", {}).get("pred_std_warn_patience", 25))
        # ---End of diag accumulators for onset/offset prediction variance ---

        #CAL
        _first_batch = next(iter(train_loader))
        _have_frame_keys = all(k in _first_batch for k in ("pitch_roll","onset_roll","offset_roll","hand_frame","clef_frame"))
        print(f"[DEBUG] frame-mode keys present: {_have_frame_keys}")
        logger.debug("frame-mode keys present: %s", _have_frame_keys)
        del _first_batch
        #END CAL
        
        optimizer.zero_grad(set_to_none=True)
        
        for it, batch in pbar:
            x = batch["video"]
            B = x.shape[0]

            # Prefer real labels; fallback to dummy
            want = ("pitch", "onset", "offset", "hand", "clef")
            have_all = all(k in batch for k in want)
            use_dummy_flag = bool(cfg["training"].get("debug_dummy_labels", False))
            if have_all and not use_dummy_flag:
                tgt = {k: batch[k] for k in want}
                tgt["pitch"]  = tgt["pitch"].float()
                tgt["hand"]   = tgt["hand"].long()
                tgt["clef"]   = tgt["clef"].long()
                tgt["onset"]  = tgt["onset"].float()
                tgt["offset"] = tgt["offset"].float()
            else:
                tgt = fabricate_dummy_targets(B)

            out = model(x)

            # --- diag: prediction distribution monitoring (remove post-debug) ---
            batch_pred_stats = {
                "onset": _prediction_stats_from_logits(out.get("onset_logits")),
                "offset": _prediction_stats_from_logits(out.get("offset_logits")),
            }
            if any(batch_pred_stats.values()):
                for head, stats in batch_pred_stats.items():
                    if not stats:
                        continue
                    acc = pred_stat_accum[head]
                    acc["mean_sum"] += stats["mean"]
                    acc["std_sum"] += stats["std"]
                    acc["min_sum"] += stats["min"]
                    acc["max_sum"] += stats["max"]
                    acc["min_all"] = min(acc["min_all"], stats["min"])
                    acc["max_all"] = max(acc["max_all"], stats["max"])
                    acc["count"] += 1
                    if stats["std"] < 0.1:
                        low_std_streak[head] += 1
                    else:
                        low_std_streak[head] = 0
                    #MK - Remove printing of stats per batch 
                    if low_std_patience > 0 and low_std_streak[head] >= low_std_patience:
                    #    warn_msg = (
                    #        f"[{head}] prediction std {stats['std']:.3f} < 0.1 for {low_std_patience} consecutive batches"
                    #    )
                    #    logger.warning(warn_msg)
                    #    pbar.write(warn_msg)
                        low_std_streak[head] = 0
                    #End MK

                if (it + 1) % log_interval == 0:
                    diag_line = []
                    for head, stats in batch_pred_stats.items():
                        if not stats:
                            continue
                        diag_line.append(
                            f"{head}: mean={stats['mean']:.3f} std={stats['std']:.3f} min={stats['min']:.3f} max={stats['max']:.3f}"
                        )
                        if writer is not None:
                            global_step = (epoch - 1) * len(train_loader) + (it + 1)
                            for key, val in stats.items():
                                writer.add_scalar(f"train_batch_pred/{head}_{key}", val, global_step)
                    if diag_line:
                        msg = f"[pred-stats] batch {it + 1}: " + " | ".join(diag_line)
                        pbar.write(msg)
                        logger.debug(msg)
            # --- End of diag: prediction distribution monitoring ---

            # --- Route to frame loss iff model is in frame mode AND batch has frame targets ---
            use_frame = (
                getattr(model, "head_mode", "clip") == "frame"
                and all(k in batch for k in ("pitch_roll", "onset_roll", "offset_roll", "hand_frame", "clef_frame"))
            )

            if use_frame:
                loss, parts = compute_loss_frame(out, batch, weights=w)
            else:
                # Guard: if model outputs (B,T,...) but we're using clip loss, pool over time
                if out["pitch_logits"].dim() == 3:
                    out = _time_pool_out_to_clip(out)
                loss, parts = compute_loss(out, tgt, crit, w)

            (loss / accum_steps).backward()
            if (it + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                _print_head_grad_norms(model, step_tag=f"e{epoch}_it{it}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for k in running.keys():
                running[k] += parts[k]
            count += 1

            if (it + 1) % log_interval == 0:
                avg = {k: running[k] / count for k in running}
                pbar.set_postfix({k: f"{v:.3f}" for k, v in avg.items()})

            # track avg events per clip from current batch (if present)
            if "labels" in batch:
                if isinstance(batch["labels"], list):
                    lens = [lbl.shape[0] for lbl in batch["labels"]]
                    if len(lens):
                        avg_events_recent = sum(lens) / len(lens)
                elif torch.is_tensor(batch["labels"]) and "labels_mask" in batch:
                    avg_events_recent = batch["labels_mask"].sum(dim=1).float().mean().item()

        if (count % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # epoch metrics
        train_metrics = {k: running[k] / max(1, count) for k in running}

        # diag: epoch-level prediction stats
        pred_epoch_summary = {}
        for head in ("onset", "offset"):
            acc = pred_stat_accum[head]
            if not acc["count"]:
                continue
            pred_epoch_summary[f"{head}_mean"] = acc["mean_sum"] / acc["count"]
            pred_epoch_summary[f"{head}_std"] = acc["std_sum"] / acc["count"]
            pred_epoch_summary[f"{head}_min_avg"] = acc["min_sum"] / acc["count"]
            pred_epoch_summary[f"{head}_max_avg"] = acc["max_sum"] / acc["count"]
            pred_epoch_summary[f"{head}_min_global"] = acc["min_all"]
            pred_epoch_summary[f"{head}_max_global"] = acc["max_all"]
        # End of diag: epoch-level prediction stats

        dt = perf_counter() - t0
        print(f"Epoch {epoch} | time {dt:.1f}s | " + " ".join([f"{k}={v:.3f}" for k, v in train_metrics.items()]))

        # diag: epoch-level prediction stats
        if pred_epoch_summary:
            summary_bits = ", ".join(f"{k}={v:.3f}" for k, v in pred_epoch_summary.items())
            print(f"    ↳ pred-stats: {summary_bits}")
            logger.info("[pred-epoch] %s", summary_bits)
        # End of diag: epoch-level prediction stats

        # TB logging (train scalars + data stat)
        if writer is not None:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            if avg_events_recent is not None:
                writer.add_scalar("data/avg_events_per_clip", avg_events_recent, epoch)
            # diag: epoch-level prediction stats
            for k, v in pred_epoch_summary.items():
                writer.add_scalar(f"train_pred/{k}", v, epoch)
            # End of diag: epoch-level prediction stats

        # --- evaluation & best checkpoint ---
        val_total = None
        if eval_freq and val_loader is not None and (epoch % eval_freq == 0):
            val_metrics = evaluate_one_epoch(model, val_loader, cfg, optimizer=optimizer)
            if val_metrics is None:
                logger.warning("[train:eval] metrics skipped (timeout) for epoch %d", epoch)
            else:
                print("Val:", " ".join([f"{k}={v:.3f}\t" for k, v in val_metrics.items()]))
                logger.info("Val: %s", " ".join(f"{k}={v:.3f}" for k, v in val_metrics.items()))
                if writer is not None:
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, epoch)

                metrics_payload: Dict[str, Any] = {
                    "epoch": float(epoch),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                metrics_payload.update({k: float(v) for k, v in val_metrics.items()})
                metrics_path = log_dir / "inner_eval_metrics.yaml"
                if _atomic_write_metrics(metrics_payload, metrics_path):
                    print(f"[train:eval] metrics saved (atomic) path={metrics_path}", flush=True)
                    logger.info("[train:eval] metrics saved (atomic) path=%s", metrics_path)
                else:
                    print("[train:eval] skip write (lock timeout)", flush=True)
                    logger.warning("[train:eval] skip write (lock timeout)")

                val_total = val_metrics.get("total")
                if val_total is not None and val_total < best_val:
                    best_val = val_total
                    save_checkpoint(best_path, model, optimizer, epoch, cfg, best_val)
                    print(f"Saved BEST to: {best_path} (val_total={val_total:.3f})")
                    logger.info("Saved BEST to: %s (val_total=%.3f)", best_path, val_total)

        epoch_finish = getattr(train_base_dataset, "finish_epoch_snapshot", None)
        if callable(epoch_finish):
            try:
                epoch_finish()
            except Exception as exc:
                logger.warning("[train] failed to finalize epoch snapshot: %s", exc)

        # --- always save LAST ---
        save_checkpoint(last_path, model, optimizer, epoch, cfg, best_val)
        if (epoch % save_every) == 0:
            # optional per-epoch named snapshot
            save_checkpoint(ckpt_dir / f"tivit_epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg, best_val)

    # close writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
