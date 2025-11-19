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
    - decoder.decode helpers: Shared decoder normalization/pooling utilities
      imported here so training matches eval/calibration behavior.

CLI:
    Run ``python scripts/train.py --config configs/config.yaml`` with optional
    overrides such as ``--epochs`` or dataset adjustments defined within the
    configuration file. Use ``--seed``/``--deterministic`` to control global
    reproducibility settings.
"""

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, cast, overload, Literal


import argparse
import copy
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
    read_selection_metadata,
)
from decoder.decode import (
    DECODER_DEFAULTS,
    pool_roll_BT,
    resolve_decoder_from_config,
    resolve_decoder_gates,
)
from theory.key_prior_runtime import (
    resolve_key_prior_settings,
    apply_key_prior_to_logits,
)

logger = get_logger(__name__)

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path("configs/config.yaml")

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

class PosRateEMA:
    """Track an exponential moving average for positive class rates."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.value: Optional[torch.Tensor] = None

    def process(self, observation: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        """
        Incorporate a new observation and return the effective EMA value.

        Args:
            observation: Tensor containing the latest positive rate estimate.
            update: Whether to update the internal EMA state.

        Returns:
            Tensor with the EMA-smoothed positive rate (clone, on CPU).
        """
        obs = observation.detach().float().cpu()
        if self.value is None:
            if update:
                self.value = obs.clone()
            return obs.clone()
        if update:
            self.value.mul_(1.0 - self.alpha)
            self.value.add_(obs * self.alpha)
        return self.value.clone()

    def peek(self) -> Optional[torch.Tensor]:
        return None if self.value is None else self.value.clone()

    def state_dict(self) -> Dict[str, Any]:
        if self.value is None:
            return {"value": None}
        return {"value": self.value.tolist()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        value = state.get("value") if isinstance(state, Mapping) else None
        if value is None:
            self.value = None
        else:
            self.value = torch.tensor(value, dtype=torch.float32)


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_positive_int(value: Any) -> Optional[int]:
    try:
        float_val = float(value)
    except (TypeError, ValueError):
        return None
    if float_val <= 0.0:
        return None
    rounded = int(round(float_val))
    if abs(float_val - rounded) > 1e-6:
        return None
    return rounded if rounded > 0 else None


def _load_selection_metric(path: Path, field: str = "mean_event_f1") -> Optional[float]:
    metadata = read_selection_metadata(path)
    if not metadata:
        return None
    metrics = metadata.get("metrics", {})
    value = metrics.get(field)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _update_selection_in_config(
    result: SelectionResult,
    context: SelectionContext,
    cfg: MutableMapping[str, Any],
    decoder_snapshot: Mapping[str, Any],
) -> None:
    training_cfg = cast(MutableMapping[str, Any], cfg.setdefault("training", {}))
    metrics_cfg = cast(MutableMapping[str, Any], training_cfg.setdefault("metrics", {}))
    decoder_before = copy.deepcopy(metrics_cfg.get("decoder"))

    metrics_cfg["prob_threshold_onset"] = float(result.onset_threshold)
    metrics_cfg["prob_threshold"] = float(result.onset_threshold)
    metrics_cfg["prob_threshold_offset"] = float(result.offset_threshold)

    agg_cfg = cast(MutableMapping[str, Any], metrics_cfg.setdefault("aggregation", {}))
    k_cfg = cast(MutableMapping[str, Any], agg_cfg.setdefault("k", {}))
    k_cfg["onset"] = int(result.k_onset)
    if _coerce_positive_int(k_cfg.get("offset")) is None:
        k_cfg["offset"] = 1

    temp_on = context.temperature_onset if context.temperature_onset is not None else context.temperature
    temp_off = context.temperature_offset if context.temperature_offset is not None else context.temperature
    if temp_on is not None:
        metrics_cfg["prob_temperature_onset"] = float(temp_on)
    if temp_off is not None:
        metrics_cfg["prob_temperature_offset"] = float(temp_off)
    temp_scalar = (
        float(temp_on)
        if temp_on is not None
        else (float(temp_off) if temp_off is not None else None)
    )
    if temp_scalar is not None:
        metrics_cfg["prob_temperature"] = temp_scalar

    bias_on = context.bias_onset if context.bias_onset is not None else context.bias
    bias_off = context.bias_offset if context.bias_offset is not None else context.bias
    if bias_on is not None:
        metrics_cfg["prob_logit_bias_onset"] = float(bias_on)
    if bias_off is not None:
        metrics_cfg["prob_logit_bias_offset"] = float(bias_off)
    bias_scalar = (
        float(bias_on)
        if bias_on is not None
        else (float(bias_off) if bias_off is not None else None)
    )
    if bias_scalar is not None:
        metrics_cfg["prob_logit_bias"] = bias_scalar

    config_path = CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    reloaded_cfg = load_config(config_path)
    reloaded_metrics = reloaded_cfg.get("training", {}).get("metrics", {}) if isinstance(reloaded_cfg, Mapping) else {}
    decoder_after = reloaded_metrics.get("decoder") if isinstance(reloaded_metrics, Mapping) else {}
    snapshot_normalized = copy.deepcopy(decoder_snapshot)
    if decoder_after != decoder_before or decoder_after != snapshot_normalized:
        logger.error("[train] decoder subtree changed during selection write-back; aborting to protect config")
        raise RuntimeError("Decoder subtree changed during selection write-back")


def _tensor_from_scalar(value: float) -> torch.Tensor:
    return torch.tensor([float(value)], dtype=torch.float32)


def _pos_weight_from_rate(
    rate: torch.Tensor,
    *,
    eps: float = 1e-6,
    clamp_min: float = 1.0,
    clamp_max: float = 100.0,
) -> torch.Tensor:
    rate_clamped = rate.clone().clamp(min=eps, max=1.0 - eps)
    pos_weight = (1.0 - rate_clamped) / rate_clamped
    return pos_weight.clamp_(min=clamp_min, max=clamp_max)


class OnOffPosWeightEMA:
    """Maintain EMA statistics for onset/offset heads (clip & frame modes)."""

    def __init__(self, alpha: float):
        alpha = float(alpha)
        self.alpha = alpha
        self.clip = {"onset": PosRateEMA(alpha), "offset": PosRateEMA(alpha)}
        self.frame = {"onset": PosRateEMA(alpha), "offset": PosRateEMA(alpha)}

    @staticmethod
    def _reduce_clip_targets(targets: torch.Tensor) -> torch.Tensor:
        return _tensor_from_scalar(float(targets.float().mean().item()))

    @staticmethod
    def _reduce_frame_roll(roll: torch.Tensor) -> torch.Tensor:
        P = roll.shape[-1]
        return roll.reshape(-1, P).float().mean(dim=0).cpu()

    def clip_pos_weight(self, head: str, targets: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        tracker = self.clip[head]
        rate = tracker.process(self._reduce_clip_targets(targets), update=update)
        return _pos_weight_from_rate(rate)

    def frame_pos_weight(self, head: str, roll: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        tracker = self.frame[head]
        rate = tracker.process(self._reduce_frame_roll(roll), update=update)
        return _pos_weight_from_rate(rate)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "clip": {k: v.state_dict() for k, v in self.clip.items()},
            "frame": {k: v.state_dict() for k, v in self.frame.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            return
        for scope_name, registry in (("clip", self.clip), ("frame", self.frame)):
            saved_scope = state.get(scope_name, {}) if isinstance(state.get(scope_name), Mapping) else {}
            for head, tracker in registry.items():
                tracker_state = saved_scope.get(head)
                if tracker_state is not None:
                    tracker.load_state_dict(tracker_state)

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

def _dynamic_pos_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    base_crit: nn.BCEWithLogitsLoss,
    *,
    pos_rate_override: Optional[torch.Tensor] = None,
    pos_weight_override: Optional[torch.Tensor] = None,
):
    """Compute BCEWithLogits with adaptive or overridden positive class weights."""

    eps = 1e-6
    target_float = targets.float()
    if pos_weight_override is not None:
        pos_weight = pos_weight_override.detach().to(device=logits.device, dtype=logits.dtype)
    else:
        if pos_rate_override is not None:
            positive_rate = pos_rate_override.detach().to(device=logits.device, dtype=logits.dtype)
        else:
            positive_rate = target_float.mean()
        positive_rate = positive_rate.clamp(min=eps, max=1.0 - eps)
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


def _resolve_onoff_loss_config(weights: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build per-head onset/offset loss settings with backward-compatible defaults.

    Legacy flat keys (onoff_loss, focal_gamma, ...) remain supported, while
    per-head overrides can be provided via ``onoff_heads.<head>.<key>`` or
    simple aliases such as ``offset_focal_gamma``.
    """
    if not isinstance(weights, Mapping):
        weights = {}

    default_cfg = {
        "loss": str(weights.get("onoff_loss", "bce_pos")).lower(),
        "pos_weight_mode": str(weights.get("onoff_pos_weight_mode", "adaptive")).lower(),
        "pos_weight": weights.get("onoff_pos_weight"),
        "focal_gamma": float(weights.get("focal_gamma", 2.0)),
        "focal_alpha": float(weights.get("focal_alpha", 0.25)),
        "prior_mean": float(weights.get("onoff_prior_mean", 0.12)),
        "prior_weight": float(weights.get("onoff_prior_weight", 0.0)),
    }

    per_head_overrides = weights.get("onoff_heads", {})
    if not isinstance(per_head_overrides, Mapping):
        per_head_overrides = {}

    alias_patterns = {
        "loss": ("{head}_loss", "{head}_onoff_loss"),
        "pos_weight_mode": ("{head}_pos_weight_mode", "{head}_onoff_pos_weight_mode"),
        "pos_weight": ("{head}_pos_weight", "{head}_onoff_pos_weight"),
        "focal_gamma": ("{head}_focal_gamma",),
        "focal_alpha": ("{head}_focal_alpha",),
        "prior_mean": ("{head}_prior_mean", "{head}_onoff_prior_mean"),
        "prior_weight": ("{head}_prior_weight", "{head}_onoff_prior_weight"),
    }

    resolved: Dict[str, Dict[str, Any]] = {}
    for head in ("onset", "offset"):
        cfg = dict(default_cfg)

        nested = per_head_overrides.get(head)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                if key in cfg:
                    cfg[key] = value

        for key, patterns in alias_patterns.items():
            for pattern in patterns:
                alias = pattern.format(head=head)
                if alias in weights:
                    cfg[key] = weights[alias]
                    break

        cfg["loss"] = str(cfg["loss"]).lower()
        cfg["pos_weight_mode"] = str(cfg["pos_weight_mode"]).lower()
        cfg["focal_gamma"] = float(cfg["focal_gamma"])
        cfg["focal_alpha"] = float(cfg["focal_alpha"])
        cfg["prior_mean"] = float(cfg["prior_mean"])
        cfg["prior_weight"] = float(cfg["prior_weight"])

        pos_weight_raw = cfg.get("pos_weight")
        if pos_weight_raw is None:
            pos_weight_val: Optional[float] = None
        else:
            try:
                pos_weight_val = float(pos_weight_raw)
            except (TypeError, ValueError):
                pos_weight_val = None
        if pos_weight_val is None or pos_weight_val <= 0.0:
            cfg["pos_weight"] = None
        else:
            cfg["pos_weight"] = pos_weight_val

        resolved[head] = cfg

    return resolved

def compute_loss(
    out: dict,
    tgt: dict,
    crit: dict,
    weights: dict,
    pos_rate_state: Optional[OnOffPosWeightEMA] = None,
    *,
    update_stats: bool = True,
):
    # Guard: if logits are time-distributed but we're in clip-loss, pool over time
    if out["pitch_logits"].dim() == 3:  # (B,T,P)
        out = _time_pool_out_to_clip(out)

    loss_pitch = _dynamic_pos_weighted_bce(out["pitch_logits"], tgt["pitch"], crit["pitch"]) * float(weights.get("pitch", 1.0))

    head_cfgs = _resolve_onoff_loss_config(weights)
    onset_cfg = head_cfgs["onset"]
    offset_cfg = head_cfgs["offset"]

    def _clip_pos_weight(
        head: str,
        cfg: Mapping[str, Any],
        targets: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        mode = str(cfg.get("pos_weight_mode", "adaptive")).lower()
        loss_mode = str(cfg.get("loss", "bce_pos")).lower()
        if loss_mode not in {"bce_pos", "bce", "bce_with_logits"}:
            return None
        if mode == "fixed" and cfg.get("pos_weight") is not None:
            return torch.full((targets.shape[-1],), float(cfg["pos_weight"]), dtype=torch.float32)
        if mode == "ema" and pos_rate_state is not None:
            return pos_rate_state.clip_pos_weight(head, targets, update=update_stats)
        if mode in {"none", "off"}:
            return None
        # adaptive / fallback handled within _dynamic_pos_weighted_bce
        return None

    def _compute_clip_head_loss(
        head: str,
        cfg: Mapping[str, Any],
        logits: torch.Tensor,
        targets: torch.Tensor,
        base_crit: nn.BCEWithLogitsLoss,
    ) -> torch.Tensor:
        mode = str(cfg.get("loss", "bce_pos")).lower()
        if mode in {"focal", "focal_bce"}:
            gamma = float(cfg.get("focal_gamma", 2.0))
            alpha = float(cfg.get("focal_alpha", 0.25))
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            weight = (alpha * (1.0 - probs).pow(gamma)).detach()
            return (weight * bce).mean()
        pos_weight_override = _clip_pos_weight(head, cfg, targets)
        return _dynamic_pos_weighted_bce(
            logits,
            targets,
            base_crit,
            pos_weight_override=pos_weight_override,
        )

    loss_onset = _compute_clip_head_loss("onset", onset_cfg, out["onset_logits"], tgt["onset"], crit["onset"]) * float(weights.get("onset", 1.0))
    loss_offset = _compute_clip_head_loss("offset", offset_cfg, out["offset_logits"], tgt["offset"], crit["offset"]) * float(weights.get("offset", 1.0))
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
    
def compute_loss_frame(
    out: dict,
    batch: dict,
    weights: dict,
    pos_rate_state: Optional[OnOffPosWeightEMA] = None,
    *,
    update_stats: bool = True,
):
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
        pitch_roll  = pool_roll_BT(pitch_roll,  T_logits)
        onset_roll  = pool_roll_BT(onset_roll,  T_logits)
        offset_roll = pool_roll_BT(offset_roll, T_logits)
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

    head_cfgs = _resolve_onoff_loss_config(weights)
    onset_cfg = head_cfgs["onset"]
    offset_cfg = head_cfgs["offset"]

    def _frame_pos_weight(head: str, cfg: Mapping[str, Any], roll: torch.Tensor) -> Optional[torch.Tensor]:
        mode = str(cfg.get("pos_weight_mode", "adaptive")).lower()
        loss_mode = str(cfg.get("loss", "bce_pos")).lower()
        if loss_mode not in {"bce_pos", "bce", "bce_with_logits"}:
            return None
        if mode == "fixed" and cfg.get("pos_weight") is not None:
            return torch.full((P,), float(cfg["pos_weight"]), dtype=torch.float32)
        if mode == "ema" and pos_rate_state is not None:
            return pos_rate_state.frame_pos_weight(head, roll, update=update_stats)
        if mode in {"none", "off"}:
            return None
        if mode in {"adaptive", "auto"}:
            return _adaptive_pos_weight(roll, P, eps)
        # fallback if unknown
        return _adaptive_pos_weight(roll, P, eps)

    def _compute_frame_head_loss(
        head: str,
        cfg: Mapping[str, Any],
        logits: torch.Tensor,
        roll: torch.Tensor,
        pos_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mode = str(cfg.get("loss", "bce_pos")).lower()
        if mode in {"focal", "focal_bce"}:
            gamma = float(cfg.get("focal_gamma", 2.0))
            alpha = float(cfg.get("focal_alpha", 0.25))
            bce = F.binary_cross_entropy_with_logits(logits, roll, reduction="none")
            probs = torch.sigmoid(logits)
            weight = (alpha * (1.0 - probs).pow(gamma)).detach()
            return (weight * bce).mean()

        # default: BCE with optional pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
            crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            crit = nn.BCEWithLogitsLoss()
        return crit(logits, roll)

    pos_w_on = _frame_pos_weight("onset", onset_cfg, onset_roll)
    pos_w_off = _frame_pos_weight("offset", offset_cfg, offset_roll)

    loss_onset = _compute_frame_head_loss("onset", onset_cfg, onset_logit, onset_roll, pos_w_on)
    loss_offset = _compute_frame_head_loss("offset", offset_cfg, offset_logit, offset_roll, pos_w_off)

    loss_onset = loss_onset * float(weights.get("onset", 1.0))
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

    reg_terms: Dict[str, torch.Tensor] = {}
    for head, cfg, logits in (
        ("onset", onset_cfg, onset_logit),
        ("offset", offset_cfg, offset_logit),
    ):
        prior_w = float(cfg.get("prior_weight", 0.0))
        if prior_w <= 0.0:
            continue
        prior_mean = float(cfg.get("prior_mean", 0.12))
        act_mean = torch.sigmoid(logits).mean()
        reg = prior_w * (act_mean - prior_mean).abs()
        total = total + reg
        reg_terms[head] = reg

    if reg_terms:
        reg_sum = sum(float(reg.detach().cpu()) for reg in reg_terms.values())
        parts["reg_onoff"] = reg_sum
        for head, reg in reg_terms.items():
            parts[f"reg_{head}"] = float(reg.detach().cpu())

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


def _summarize_pitch_predictions(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> Optional[Dict[str, float]]:
    """Return aggregate TP/FP/FN counts plus frame-exact matches for pitch predictions."""

    if not torch.is_tensor(pred_mask) or not torch.is_tensor(target_mask):
        return None
    if pred_mask.shape != target_mask.shape:
        raise ValueError(f"Pitch prediction/target shape mismatch: pred={pred_mask.shape} target={target_mask.shape}")
    if pred_mask.numel() == 0:
        return None

    pred_bool = pred_mask.bool()
    target_bool = target_mask.bool()
    tp = float((pred_bool & target_bool).sum().item())
    fp = float((pred_bool & (~target_bool)).sum().item())
    fn = float(((~pred_bool) & target_bool).sum().item())
    pred_pos = float(pred_bool.sum().item())
    target_pos = float(target_bool.sum().item())

    last_dim = pred_bool.shape[-1]
    pred_flat = pred_bool.reshape(-1, last_dim)
    target_flat = target_bool.reshape(-1, last_dim)
    matches = (pred_flat == target_flat).all(dim=-1)
    frame_match = float(matches.sum().item())
    frame_total = float(matches.numel())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_pos": pred_pos,
        "target_pos": target_pos,
        "frame_match": frame_match,
        "frame_total": frame_total,
    }


def _accumulate_pitch_counts(counts: Dict[str, Any], stats: Mapping[str, float]) -> None:
    counts["pitch_pos_tp"] += stats["tp"]
    counts["pitch_pos_fp"] += stats["fp"]
    counts["pitch_pos_fn"] += stats["fn"]
    counts["pitch_preds_pos"] += stats["pred_pos"]
    counts["pitch_targets_pos"] += stats["target_pos"]
    counts["pitch_frame_match"] += stats["frame_match"]
    counts["pitch_frame_total"] += stats["frame_total"]

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


@overload
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
    return_mask: Literal[True],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...


@overload
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
    return_mask: Literal[False] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


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
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        "pitch_pos_tp": 0.0,
        "pitch_pos_fp": 0.0,
        "pitch_pos_fn": 0.0,
        "pitch_preds_pos": 0.0,
        "pitch_targets_pos": 0.0,
        "pitch_frame_match": 0.0,
        "pitch_frame_total": 0.0,
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
    if Bp != Bg:
        raise ValueError(
            f"Cannot align masks with different batch sizes pred={pred_mask.shape} gt={gt_mask.shape}"
        )

    aligned = gt_mask
    if Tg != Tp:
        aligned = aligned.permute(0, 2, 1).float()
        aligned = F.adaptive_max_pool1d(aligned, Tp)
        aligned = aligned.permute(0, 2, 1).contiguous()

    if Pg != Pp:
        aligned = aligned.reshape(Bg * Tp, 1, Pg).float()
        aligned = F.adaptive_max_pool1d(aligned, Pp)
        aligned = aligned.reshape(Bg, Tp, Pp)

    return aligned.to(gt_mask.dtype)


def _median_filter_time_proxy(clip_probs: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply a 1D median filter over time for a single (T,P) pianoroll."""

    if kernel_size <= 1:
        return clip_probs
    if clip_probs.ndim != 2:
        raise ValueError(f"Median filter expects (T,P) tensor, got {clip_probs.ndim} dims")
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    probs_pt = clip_probs.transpose(0, 1).unsqueeze(1)  # (P,1,T)
    padded = F.pad(probs_pt, (pad, pad), mode="replicate")
    windows = padded.unfold(-1, kernel_size, 1)  # (P,1,T,k)
    filtered = windows.median(dim=-1).values  # (P,1,T)
    return filtered.squeeze(1).transpose(0, 1).contiguous()


def _proxy_decode_mask(
    probs: torch.Tensor,
    *,
    open_thr: float,
    hold_thr: float,
    min_on: int,
    min_off: int,
    merge_gap: int,
    median: int,
) -> torch.Tensor:
    """Decode per-key probabilities with lightweight hysteresis-style smoothing."""

    if probs.ndim not in (2, 3):
        raise ValueError(f"Expected probs with 2 or 3 dims, got {probs.ndim}")
    if probs.numel() == 0:
        return torch.zeros_like(probs, dtype=torch.bool)

    high_thr = float(open_thr)
    low_thr = float(hold_thr)
    if not math.isfinite(high_thr):
        high_thr = 0.5
    if not math.isfinite(low_thr):
        low_thr = high_thr
    if low_thr > high_thr:
        low_thr = high_thr
    if high_thr < 0.0:
        high_thr = 0.0
    if low_thr < 0.0:
        low_thr = 0.0
    min_on = max(0, int(min_on))
    min_off = max(0, int(min_off))
    merge_gap = max(0, int(merge_gap))
    median = int(median)

    def _decode_single(single: torch.Tensor) -> torch.Tensor:
        work = single.detach().float().cpu()
        if median > 1:
            work = _median_filter_time_proxy(work, median)
        T, P = work.shape
        mask_cpu = torch.zeros((T, P), dtype=torch.bool)
        for pitch in range(P):
            seq = work[:, pitch].tolist()
            segments: list[Tuple[int, int]] = []
            active = False
            start_idx = 0
            for t, raw_val in enumerate(seq):
                val = float(raw_val) if math.isfinite(raw_val) else 0.0
                if not active:
                    if val >= high_thr:
                        active = True
                        start_idx = t
                else:
                    if val < low_thr:
                        segments.append((start_idx, t))
                        active = False
            if active:
                segments.append((start_idx, T))
            if not segments:
                continue
            merged: list[list[int]] = []
            for seg_start, seg_end in segments:
                if not merged:
                    merged.append([seg_start, seg_end])
                    continue
                prev_start, prev_end = merged[-1]
                gap = seg_start - prev_end
                should_merge = False
                if merge_gap >= 0 and gap <= merge_gap:
                    should_merge = True
                if min_off > 0 and gap < min_off:
                    should_merge = True
                if should_merge:
                    merged[-1][1] = seg_end
                else:
                    merged.append([seg_start, seg_end])
            for seg_start, seg_end in merged:
                if seg_end - seg_start < min_on:
                    continue
                mask_cpu[seg_start:seg_end, pitch] = True
        return mask_cpu.to(single.device)

    if probs.ndim == 2:
        return _decode_single(probs)

    decoded = [_decode_single(clip) for clip in probs]
    if not decoded:
        return torch.zeros_like(probs, dtype=torch.bool)
    return torch.stack(decoded, dim=0)


def _event_f1_clip(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    hop_seconds: float,
    tol_sec: float,
    eps: float = 1e-8,
) -> Optional[float]:
    """Compute event-level F1 for a single clip (T,P) mask pair."""

    if pred_mask.shape != target_mask.shape:
        raise ValueError("Prediction and target masks must share shape for event F1")

    pred_idx = pred_mask.nonzero(as_tuple=False)
    target_idx = target_mask.nonzero(as_tuple=False)
    if pred_idx.numel() == 0 and target_idx.numel() == 0:
        return None

    pred_idx = pred_idx.cpu()
    target_idx = target_idx.cpu()
    pred_times = pred_idx[:, 0].to(torch.float32) * float(hop_seconds)
    target_times = target_idx[:, 0].to(torch.float32) * float(hop_seconds)
    pred_pitch = pred_idx[:, 1]
    target_pitch = target_idx[:, 1]

    used = torch.zeros(target_idx.shape[0], dtype=torch.bool)
    tp = 0
    for i in range(pred_idx.shape[0]):
        pitch = pred_pitch[i]
        time_val = pred_times[i]
        mask = (target_pitch == pitch) & (~used)
        if mask.any():
            cand_idx = torch.where(mask)[0]
            diffs = torch.abs(target_times[cand_idx] - time_val)
            min_diff, rel = torch.min(diffs, dim=0)
            if min_diff.item() <= float(tol_sec):
                tp += 1
                used[cand_idx[rel]] = True

    fp = pred_idx.shape[0] - tp
    fn = target_idx.shape[0] - tp
    precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall + eps)


def _event_f1_batch(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    hop_seconds: float,
    tol_sec: float,
) -> Tuple[float, int]:
    """Aggregate event-level F1 over a batch."""

    if pred_mask.shape != target_mask.shape:
        raise ValueError("Predictions and targets must align for batch event F1")

    if pred_mask.ndim == 2:
        score = _event_f1_clip(pred_mask, target_mask, hop_seconds, tol_sec)
        if score is None:
            return 0.0, 0
        return float(score), 1

    if pred_mask.ndim != 3:
        raise ValueError("Expected 2D or 3D pianoroll masks for event F1 computation")

    total = 0.0
    count = 0
    for clip_pred, clip_target in zip(pred_mask, target_mask):
        score = _event_f1_clip(clip_pred, clip_target, hop_seconds, tol_sec)
        if score is None:
            continue
        total += float(score)
        count += 1
    return total, count


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

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    cfg,
    writer=None,
    epoch=1,
    pos_rate_state: Optional[OnOffPosWeightEMA] = None,
):
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
                loss, parts = compute_loss_frame(out, batch, weights=w, pos_rate_state=pos_rate_state)
            else:
                # Guard: if model outputs (B,T,...) but we're using clip loss, pool over time
                if out["pitch_logits"].dim() == 3:
                    out = _time_pool_out_to_clip(out)
                loss, parts = compute_loss(out, tgt, crit, w, pos_rate_state=pos_rate_state)

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


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    cfg: Mapping[str, Any],
    best_val: float | None = None,
    best_event_f1: float | None = None,
    trainer_state: Optional[Dict[str, Any]] = None,
):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "best_val": best_val,
        "best_event_f1": best_event_f1,
    }
    if trainer_state is not None:
        state["trainer_state"] = trainer_state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    
def evaluate_one_epoch(
    model,
    loader,
    cfg,
    *,
    optimizer=None,
    timeout_minutes: int = 15,
    pos_rate_state: Optional[OnOffPosWeightEMA] = None,
):
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
        trough_val = cache_data.get("throughput_win_per_s")
        if trough_val is None:
            cached_throughput = None
        else:
            try:
                cached_throughput = float(trough_val)
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
    dataset_cfg = cfg.get("dataset", {}) or {}
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
    decoder_cfg_runtime = cfg.get("decoder", {}) or {}
    key_prior_settings = resolve_key_prior_settings(decoder_cfg_runtime.get("key_prior"))
    if key_prior_settings.enabled:
        logger.info(
            "[decoder] key prior active (ref_head=%s, apply_to=%s)",
            key_prior_settings.ref_head,
            ",".join(key_prior_settings.apply_to),
        )
    midi_low_cfg = frame_targets_cfg.get("note_min")
    key_prior_midi_low = int(midi_low_cfg) if isinstance(midi_low_cfg, (int, float)) else 21
    temporal_decoder_cfg = resolve_decoder_from_config(
        metrics_cfg,
        fallback_open={"onset": thr_on, "offset": thr_off},
    )
    decoder_onset = temporal_decoder_cfg["onset"]
    decoder_offset = temporal_decoder_cfg["offset"]
    onset_open_thr, onset_hold_thr = resolve_decoder_gates(
        decoder_onset,
        fallback_open=thr_on,
        default_hold=DECODER_DEFAULTS["onset"]["hold"],
    )
    offset_open_thr, offset_hold_thr = resolve_decoder_gates(
        decoder_offset,
        fallback_open=thr_off,
        default_hold=DECODER_DEFAULTS["offset"]["hold"],
    )
    decoder_onset["open_effective"] = onset_open_thr
    decoder_onset["hold_effective"] = onset_hold_thr
    decoder_offset["open_effective"] = offset_open_thr
    decoder_offset["hold_effective"] = offset_hold_thr
    event_proxy_accum = {
        "onset_sum": 0.0,
        "offset_sum": 0.0,
        "onset_count": 0,
        "offset_count": 0,
    }

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

                metrics_out: Dict[str, Any] = out
                heads_for_prior = cast(dict[str, torch.Tensor], {})
                onset_logits_for_prior = out.get("onset_logits")
                offset_logits_for_prior = out.get("offset_logits")
                pitch_logits_for_prior = out.get("pitch_logits")
                if torch.is_tensor(onset_logits_for_prior):
                    heads_for_prior["onset"] = onset_logits_for_prior
                if torch.is_tensor(offset_logits_for_prior):
                    heads_for_prior["offset"] = offset_logits_for_prior
                if torch.is_tensor(pitch_logits_for_prior):
                    heads_for_prior["pitch"] = pitch_logits_for_prior
                prior_logits = cast(dict[str, torch.Tensor], {})
                if len(heads_for_prior) > 0:
                    prior_logits = apply_key_prior_to_logits(
                        heads_for_prior,
                        key_prior_settings,
                        fps=decode_fps,
                        midi_low=key_prior_midi_low,
                        midi_high=None,
                    )
                if prior_logits:
                    metrics_out = dict(out)
                    for head_name, tensor in prior_logits.items():
                        metrics_out[f"{head_name}_logits"] = tensor
                    if torch.is_tensor(onset_probs_valid) and "onset" in prior_logits:
                        onset_probs_valid = _apply_sigmoid_calibration(
                            metrics_out["onset_logits"],
                            temperature=onset_cal["temperature"],
                            bias=onset_cal["bias"],
                        )
                    if torch.is_tensor(offset_probs_valid) and "offset" in prior_logits:
                        offset_probs_valid = _apply_sigmoid_calibration(
                            metrics_out["offset_logits"],
                            temperature=offset_cal["temperature"],
                            bias=offset_cal["bias"],
                        )

                onset_logits_eval = metrics_out.get("onset_logits")
                offset_logits_eval = metrics_out.get("offset_logits")
                pitch_logits_eval = metrics_out.get("pitch_logits")

                use_frame = (
                    getattr(model, "head_mode", "clip") == "frame"
                    and all(k in batch for k in ("pitch_roll", "onset_roll", "offset_roll", "hand_frame", "clef_frame"))
                )

                frame_batch = None
                if use_frame:
                    pitch_roll_sel = _select_valid(batch["pitch_roll"])
                    onset_roll_sel = _select_valid(batch["onset_roll"])
                    offset_roll_sel = _select_valid(batch["offset_roll"])
                    hand_frame_sel = _select_valid(batch["hand_frame"])
                    clef_frame_sel = _select_valid(batch["clef_frame"])

                    required_tensors = (pitch_roll_sel, onset_roll_sel, offset_roll_sel, hand_frame_sel, clef_frame_sel)
                    if any(sel is None or not torch.is_tensor(sel) for sel in required_tensors):
                        logger.warning("[eval] Missing frame targets after validity filtering; skipping batch.")
                        continue

                    pitch_roll_tensor = cast(torch.Tensor, pitch_roll_sel)
                    onset_roll_tensor = cast(torch.Tensor, onset_roll_sel)
                    offset_roll_tensor = cast(torch.Tensor, offset_roll_sel)
                    hand_frame_tensor = cast(torch.Tensor, hand_frame_sel)
                    clef_frame_tensor = cast(torch.Tensor, clef_frame_sel)

                    frame_batch = {
                        "pitch_roll": pitch_roll_tensor.float(),
                        "onset_roll": onset_roll_tensor.float(),
                        "offset_roll": offset_roll_tensor.float(),
                        "hand_frame": hand_frame_tensor.long(),
                        "clef_frame": clef_frame_tensor.long(),
                    }
                    loss, parts = compute_loss_frame(
                        out,
                        frame_batch,
                        weights=w,
                        pos_rate_state=pos_rate_state,
                        update_stats=False,
                    )
                else:
                    if out["pitch_logits"].dim() == 3:
                        out = _time_pool_out_to_clip(out)
                        onset_probs_valid = None
                        offset_probs_valid = None
                    loss, parts = compute_loss(
                        out,
                        tgt,
                        crit,
                        w,
                        pos_rate_state=pos_rate_state,
                        update_stats=False,
                    )

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
                    if onset_logits_eval is not None and onset_logit_stats is not None and onset_prob_stats is not None:
                        _update_stats(onset_logit_stats, onset_logits_eval)
                        onset_probs_stats = onset_probs_valid if torch.is_tensor(onset_probs_valid) else None
                        if onset_probs_stats is None:
                            onset_probs_stats = _apply_sigmoid_calibration(
                                onset_logits_eval,
                                temperature=onset_cal["temperature"],
                                bias=onset_cal["bias"],
                            )
                            if not torch.is_tensor(onset_probs_stats):
                                onset_probs_stats = None
                        _update_stats(onset_prob_stats, onset_probs_stats)
                        if onset_k_effective is None:
                            onset_dim = onset_logits_eval.shape[-1] if onset_logits_eval.ndim > 0 else 1
                            onset_k_effective = max(1, min(int(agg_cfg["k_onset"]), onset_dim))
                    if offset_logits_eval is not None and offset_logit_stats is not None and offset_prob_stats is not None:
                        _update_stats(offset_logit_stats, offset_logits_eval)
                        offset_probs_stats = offset_probs_valid if torch.is_tensor(offset_probs_valid) else None
                        if offset_probs_stats is None:
                            offset_probs_stats = _apply_sigmoid_calibration(
                                offset_logits_eval,
                                temperature=offset_cal["temperature"],
                                bias=offset_cal["bias"],
                            )
                            if not torch.is_tensor(offset_probs_stats):
                                offset_probs_stats = None
                        _update_stats(offset_prob_stats, offset_probs_stats)
                        if offset_k_effective is None:
                            offset_dim = offset_logits_eval.shape[-1] if offset_logits_eval.ndim > 0 else 1
                            offset_k_effective = max(1, min(int(agg_cfg["k_offset"]), offset_dim))

                # --- metrics ---
                if use_frame:
                    # --- align frame targets to logits time (T -> T_logits) ---
                    if frame_batch is None:
                        logger.warning("[eval] Frame batch missing despite frame_mode; skipping batch for metrics.")
                        continue
                    if onset_logits_eval is None or offset_logits_eval is None:
                        logger.warning("[eval] Missing logits for frame-mode evaluation; skipping batch.")
                        continue
                    if not torch.is_tensor(pitch_logits_eval):
                        logger.warning("[eval] Missing pitch logits for frame-mode evaluation; skipping batch.")
                        continue
                    B, T_logits, P = onset_logits_eval.shape

                    pitch_roll = frame_batch["pitch_roll"].to(pitch_logits_eval.device)
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
                        onset_logits_eval,
                        thr_on,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_onset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                    )
                    loose_offset_pred, loose_offset_counts = _aggregate_onoff_predictions(
                        offset_logits_eval,
                        thr_off,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_offset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=offset_cal["temperature"],
                        bias=offset_cal["bias"],
                    )
                    strict_onset_pred, strict_onset_counts, strict_onset_mask = _aggregate_onoff_predictions(
                        onset_logits_eval,
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
                        offset_logits_eval,
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
                    onset_event_mask = strict_onset_mask.bool()
                    offset_event_mask = strict_offset_mask.bool()

                    pitch_roll_aligned = _align_key_mask_to(pitch_logits_eval, pitch_roll)
                    pitch_target_mask = (pitch_roll_aligned >= 0.5)
                    pitch_pred_probs = torch.sigmoid(pitch_logits_eval)
                    pitch_pred_mask = (pitch_pred_probs >= thr_pitch)
                    pitch_stats = _summarize_pitch_predictions(pitch_pred_mask, pitch_target_mask)
                    if pitch_stats is not None:
                        _accumulate_pitch_counts(loose_counts, pitch_stats)
                        _accumulate_pitch_counts(strict_counts, pitch_stats)

                    onset_probs_proxy = onset_probs_valid if torch.is_tensor(onset_probs_valid) else None
                    if onset_probs_proxy is None and "onset_logits" in metrics_out:
                        maybe_proxy = _apply_sigmoid_calibration(
                            metrics_out["onset_logits"],
                            temperature=onset_cal["temperature"],
                            bias=onset_cal["bias"],
                        )
                        if torch.is_tensor(maybe_proxy):
                            onset_probs_proxy = maybe_proxy
                    if onset_probs_proxy is not None and onset_probs_proxy.shape == strict_onset_mask_float.shape:
                        masked_onset_probs = (onset_probs_proxy.float() * strict_onset_mask_float).contiguous()
                        try:
                            onset_event_mask = _proxy_decode_mask(
                                masked_onset_probs,
                                open_thr=onset_open_thr,
                                hold_thr=onset_hold_thr,
                                min_on=decoder_onset["min_on"],
                                min_off=decoder_onset["min_off"],
                                merge_gap=decoder_onset["merge_gap"],
                                median=decoder_onset["median"],
                            )
                        except ValueError:
                            onset_event_mask = strict_onset_mask.bool()

                    offset_probs_proxy = offset_probs_valid if torch.is_tensor(offset_probs_valid) else None
                    if offset_probs_proxy is None and "offset_logits" in metrics_out:
                        maybe_proxy_off = _apply_sigmoid_calibration(
                            metrics_out["offset_logits"],
                            temperature=offset_cal["temperature"],
                            bias=offset_cal["bias"],
                        )
                        if torch.is_tensor(maybe_proxy_off):
                            offset_probs_proxy = maybe_proxy_off
                    if offset_probs_proxy is not None and offset_probs_proxy.shape == strict_offset_mask_float.shape:
                        masked_offset_probs = (offset_probs_proxy.float() * strict_offset_mask_float).contiguous()
                        try:
                            offset_event_mask = _proxy_decode_mask(
                                masked_offset_probs,
                                open_thr=offset_open_thr,
                                hold_thr=offset_hold_thr,
                                min_on=decoder_offset["min_on"],
                                min_off=decoder_offset["min_off"],
                                merge_gap=decoder_offset["merge_gap"],
                                median=decoder_offset["median"],
                            )
                        except ValueError:
                            offset_event_mask = strict_offset_mask.bool()

                    onset_pred_legacy, _ = _aggregate_onoff_predictions(
                        onset_logits_eval,
                        thr_on,
                        mode="any",
                        k=1,
                        top_k=0,
                        tau_sum=0.0,
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                    )
                    offset_pred_legacy, _ = _aggregate_onoff_predictions(
                        offset_logits_eval,
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
                    onset_gt_bool = onset_gt_mask > 0.5
                    offset_gt_bool = offset_gt_mask > 0.5

                    if onset_event_mask.ndim == 3:
                        sum_on, count_on = _event_f1_batch(
                            onset_event_mask,
                            onset_gt_bool,
                            hop_seconds,
                            event_tolerance,
                        )
                        event_proxy_accum["onset_sum"] += sum_on
                        event_proxy_accum["onset_count"] += count_on
                    if offset_event_mask.ndim == 3:
                        sum_off, count_off = _event_f1_batch(
                            offset_event_mask,
                            offset_gt_bool,
                            hop_seconds,
                            event_tolerance,
                        )
                        event_proxy_accum["offset_sum"] += sum_off
                        event_proxy_accum["offset_count"] += count_off

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
                    if onset_logits_eval is None or offset_logits_eval is None:
                        logger.warning("[eval] Missing logits for clip-mode evaluation; skipping batch.")
                        continue
                    pitch_head_eval = pitch_logits_eval
                    if torch.is_tensor(pitch_head_eval):
                        pitch_probs = torch.sigmoid(pitch_head_eval)
                        pitch_pred_mask = (pitch_probs >= thr_pitch)
                        pitch_gt_raw = tgt.get("pitch")
                        if torch.is_tensor(pitch_gt_raw):
                            pitch_gt_mask = (pitch_gt_raw >= 0.5)
                            if pitch_gt_mask.shape != pitch_pred_mask.shape:
                                if pitch_pred_mask.dim() == 3 and pitch_gt_mask.dim() == 3:
                                    pitch_gt_mask = _align_key_mask_to(pitch_pred_mask, pitch_gt_mask)
                                else:
                                    logger.warning(
                                        "[eval] Pitch target shape %s does not match predictions %s; skipping batch stats.",
                                        tuple(pitch_gt_mask.shape),
                                        tuple(pitch_pred_mask.shape),
                                    )
                                    pitch_gt_mask = None
                            if pitch_gt_mask is not None:
                                pitch_stats = _summarize_pitch_predictions(pitch_pred_mask, pitch_gt_mask)
                                if pitch_stats is not None:
                                    _accumulate_pitch_counts(loose_counts, pitch_stats)
                                    _accumulate_pitch_counts(strict_counts, pitch_stats)

                    hand_pred = F.softmax(out["hand_logits"], dim=-1).argmax(dim=-1)
                    clef_pred = F.softmax(out["clef_logits"], dim=-1).argmax(dim=-1)
                    hand_acc_val = (hand_pred == tgt["hand"]).float().mean().item()
                    clef_acc_val = (clef_pred == tgt["clef"]).float().mean().item()
                    loose_counts["hand_acc"] += hand_acc_val
                    loose_counts["clef_acc"] += clef_acc_val
                    strict_counts["hand_acc"] += hand_acc_val
                    strict_counts["clef_acc"] += clef_acc_val

                    loose_onset_pred, _ = _aggregate_onoff_predictions(
                        onset_logits_eval,
                        thr_on,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_onset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                    )
                    loose_offset_pred, _ = _aggregate_onoff_predictions(
                        offset_logits_eval,
                        thr_off,
                        mode=agg_cfg["mode"],
                        k=agg_cfg["k_offset"],
                        top_k=agg_cfg["top_k"],
                        tau_sum=agg_cfg["tau_sum"],
                        temperature=offset_cal["temperature"],
                        bias=offset_cal["bias"],
                    )
                    strict_onset_pred, _, strict_onset_mask = _aggregate_onoff_predictions(
                        onset_logits_eval,
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
                        offset_logits_eval,
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
                        onset_logits_eval,
                        thr_on,
                        mode="any",
                        k=1,
                        top_k=0,
                        tau_sum=0.0,
                        temperature=onset_cal["temperature"],
                        bias=onset_cal["bias"],
                    )
                    offset_pred_legacy, _ = _aggregate_onoff_predictions(
                        offset_logits_eval,
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
        tp = counts["pitch_pos_tp"]
        fp = counts["pitch_pos_fp"]
        fn = counts["pitch_pos_fn"]
        pitch_denom = 2 * tp + fp + fn
        pitch_pos_f1 = (2 * tp / pitch_denom) if pitch_denom > 0 else 0.0
        frame_total = counts["pitch_frame_total"]
        pitch_frame_acc = (counts["pitch_frame_match"] / frame_total) if frame_total > 0 else 0.0
        # Pitch metric used in reports/papers: per-key micro F1 on positives (pitch_pos_f1).
        # Frame-exact accuracy remains as a debugging aid and is not treated as a headline score.
        branch["pitch_pos_f1"] = pitch_pos_f1
        branch["pitch_frame_exact_acc"] = pitch_frame_acc
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

    onset_event_f1 = event_proxy_accum["onset_sum"] / event_proxy_accum["onset_count"] if event_proxy_accum["onset_count"] > 0 else 0.0
    offset_event_f1 = event_proxy_accum["offset_sum"] / event_proxy_accum["offset_count"] if event_proxy_accum["offset_count"] > 0 else 0.0
    event_f1_mean = 0.5 * (onset_event_f1 + offset_event_f1)

    if not have_valid_metrics:
        warn_msg = "[eval WARNING] No valid clips to score after filtering. Metrics set to 0."
        print(warn_msg, flush=True)
        logger.warning(warn_msg)
        loose_branch = {k: 0.0 for k in loose_branch.keys()}
        strict_branch = {k: 0.0 for k in strict_branch.keys()}
        metrics_any = {k: 0.0 for k in metrics_any.keys()}
        onset_event_f1 = 0.0
        offset_event_f1 = 0.0
        event_f1_mean = 0.0

    loose_prefixed = {f"loose_{k}": v for k, v in loose_branch.items()}
    strict_prefixed = {f"strict_{k}": v for k, v in strict_branch.items()}

    decoder_metrics = {
        "decoder_onset_open": float(decoder_onset.get("open_effective", onset_open_thr)),
        "decoder_onset_hold": float(decoder_onset.get("hold_effective", onset_hold_thr)),
        "decoder_onset_min_on": float(decoder_onset["min_on"]),
        "decoder_onset_merge_gap": float(decoder_onset["merge_gap"]),
        "decoder_offset_open": float(decoder_offset.get("open_effective", offset_open_thr)),
        "decoder_offset_hold": float(decoder_offset.get("hold_effective", offset_hold_thr)),
        "decoder_offset_min_off": float(decoder_offset["min_off"]),
        "decoder_offset_merge_gap": float(decoder_offset["merge_gap"]),
    }

    event_metrics = {
        "event_f1_onset_proxy": float(onset_event_f1),
        "event_f1_offset_proxy": float(offset_event_f1),
        "event_f1_mean_proxy": float(event_f1_mean),
    }

    val_metrics = {**losses, **loose_prefixed, **strict_prefixed, **event_metrics, **decoder_metrics}
    legacy_metrics = {**losses, **metrics_any, **event_metrics, **decoder_metrics}

    def _fmt_metrics_line(items: Mapping[str, float]):
        return " ".join(f"{k}={v:.3f}\t" for k, v in items.items())

    loose_line = _fmt_metrics_line({**losses, **loose_branch})
    strict_line = _fmt_metrics_line({**losses, **strict_branch})
    legacy_line = _fmt_metrics_line(legacy_metrics)
    print(f"[loose] {loose_line}\n")
    print(f"[strict] {strict_line}\n")
    print(f"[any] {legacy_line}\n")
    pitch_tp = loose_counts["pitch_pos_tp"]
    pitch_fp = loose_counts["pitch_pos_fp"]
    pitch_fn = loose_counts["pitch_pos_fn"]
    pitch_targets = loose_counts["pitch_targets_pos"]
    pitch_preds = loose_counts["pitch_preds_pos"]
    pitch_line = (
        f"pitch_pos_f1={loose_branch.get('pitch_pos_f1', 0.0):.3f}\t"
        f"thr={thr_pitch:.3f}\t"
        f"targets_pos={pitch_targets:.0f}\t"
        f"preds_pos={pitch_preds:.0f}\t"
        f"tp={pitch_tp:.0f}\t"
        f"fp={pitch_fp:.0f}\t"
        f"fn={pitch_fn:.0f}"
    )
    print(f"[pitch] {pitch_line}\n")
    event_line = (
        f"onset={onset_event_f1:.3f}\t offset={offset_event_f1:.3f}\t mean={event_f1_mean:.3f} "
        f"(clips_on={event_proxy_accum['onset_count']} clips_off={event_proxy_accum['offset_count']})"
    )
    print(f"[event-proxy] {event_line}\n")
    quiet_extra = {QUIET_INFO_FLAG: True}
    logger.info("[loose] %s", loose_line, extra=quiet_extra)
    logger.info("[strict] %s", strict_line, extra=quiet_extra)
    logger.info("[any] %s", legacy_line, extra=quiet_extra)
    logger.info("[pitch] %s", pitch_line, extra=quiet_extra)
    logger.info("[event-proxy] %s", event_line, extra=quiet_extra)

    calibration_line = (
        f"thr.onset={thr_on:.3f} thr.offset={thr_off:.3f} "
        f"k.onset={int(agg_cfg['k_onset'])} k.offset={int(agg_cfg['k_offset'])} "
        f"temp.onset={float(onset_cal['temperature']):.3f} temp.offset={float(offset_cal['temperature']):.3f} "
        f"bias.onset={float(onset_cal['bias']):+.3f} bias.offset={float(offset_cal['bias']):+.3f}"
    )
    print(f"[calibration] {calibration_line}")
    logger.info("[calibration] %s", calibration_line, extra=quiet_extra)

    decoder_onset_line = (
        f"decoder.onset: open={decoder_onset.get('open_effective', onset_open_thr):.3f} "
        f"hold={decoder_onset.get('hold_effective', onset_hold_thr):.3f} "
        f"min_on={int(decoder_onset['min_on'])} "
        f"merge_gap={int(decoder_onset['merge_gap'])}"
    )
    decoder_offset_line = (
        f"decoder.offset: open={decoder_offset.get('open_effective', offset_open_thr):.3f} "
        f"hold={decoder_offset.get('hold_effective', offset_hold_thr):.3f} "
        f"min_off={int(decoder_offset['min_off'])} "
        f"merge_gap={int(decoder_offset['merge_gap'])}"
    )
    print(f"[decoder] {decoder_onset_line}")
    print(f"[decoder] {decoder_offset_line}\n")
    logger.info("[decoder] %s", decoder_onset_line, extra=quiet_extra)
    logger.info("[decoder] %s", decoder_offset_line, extra=quiet_extra)

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
    global CONFIG_PATH
    CONFIG_PATH = Path(args.config).expanduser()

    args.verbose = configure_verbosity(args.verbose)

    cfg = dict(load_config(CONFIG_PATH))
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

    loss_weights_cfg = cfg["training"]["loss_weights"]
    head_cfgs = _resolve_onoff_loss_config(loss_weights_cfg)
    ema_alpha = float(loss_weights_cfg.get("onoff_pos_weight_ema_alpha", 0.05))

    def _uses_ema(head_cfg: Mapping[str, Any]) -> bool:
        mode = str(head_cfg.get("pos_weight_mode", "adaptive")).lower()
        loss_mode = str(head_cfg.get("loss", "bce_pos")).lower()
        return mode == "ema" and loss_mode in {"bce_pos", "bce", "bce_with_logits"}

    ema_heads = [head for head, cfg_head in head_cfgs.items() if _uses_ema(cfg_head)]
    pos_rate_state = OnOffPosWeightEMA(ema_alpha) if ema_heads else None

    def _describe_head(head: str, cfg_head: Mapping[str, Any]) -> str:
        mode = str(cfg_head.get("loss", "bce_pos")).lower()
        bits: List[str] = [mode]
        if mode in {"focal", "focal_bce"}:
            bits.append(f"gamma={float(cfg_head.get('focal_gamma', 2.0)):.2f}")
            bits.append(f"alpha={float(cfg_head.get('focal_alpha', 0.25)):.2f}")
        else:
            pw_mode = str(cfg_head.get("pos_weight_mode", "adaptive")).lower()
            bits.append(f"pos={pw_mode}")
            if pw_mode == "fixed" and cfg_head.get("pos_weight") is not None:
                bits.append(f"pw={float(cfg_head['pos_weight']):.3f}")
            if pw_mode == "ema" and head in ema_heads:
                bits.append(f"ema={ema_alpha:.4f}")
        prior_w = float(cfg_head.get("prior_weight", 0.0))
        if prior_w > 0.0:
            bits.append(f"prior(mean={float(cfg_head.get('prior_mean', 0.12)):.3f},w={prior_w:.3f})")
        return f"{head}=" + ",".join(bits)

    mode_msg = "[loss:onoff] " + " | ".join(_describe_head(head, cfg_head) for head, cfg_head in head_cfgs.items())
    print(mode_msg)
    logger.info(mode_msg, extra={QUIET_INFO_FLAG: True})

    def _snapshot_trainer_state() -> Optional[Dict[str, Any]]:
        if pos_rate_state is None:
            return None
        return {"pos_rate_ema": pos_rate_state.state_dict()}

    accum_steps = int(cfg.get("train", {}).get("accumulate_steps", 1))
    grad_clip = float(cfg["optim"].get("grad_clip", 1.0))
    freeze_backbone_epochs = int(cfg.get("train", {}).get("freeze_backbone_epochs", 0))
    
    # Train
    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"].get("save_every", 1))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss_value = math.inf
    best_loss_path = ckpt_dir / "tivit_best_by_loss.pt"
    best_proxy_value = float("-inf")
    best_proxy_path = ckpt_dir / "tivit_best_by_eventf1_proxy.pt"
    best_calibrated_value = float("-inf")
    best_calibrated_path = ckpt_dir / "tivit_best_by_eventf1_calibrated.pt"
    primary_best_path = ckpt_dir / "tivit_best.pt"
    last_path = ckpt_dir / "tivit_last.pt"
    last_calibration_epoch = 0
    best_proxy_trigger_value = float("-inf")
    eval_freq = int(cfg["training"].get("eval_freq", 0))

    # Try to resume from checkpoint
    resume_path = primary_best_path
    want_resume = bool(cfg.get("training", {}).get("resume", False))
    if resume_path.exists() and want_resume:
        print(f"[resume] Loading from {resume_path}")
        logger.info("[resume] Loading from %s", resume_path)
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        best_loss_value = ckpt.get("best_val", math.inf)
        best_proxy_value = float(ckpt.get("best_event_f1", float("-inf")))
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"[resume] optimizer groups mismatch; skipping optimizer state. ({e})")
            logger.warning("[resume] optimizer groups mismatch; skipping optimizer state. (%s)", e)
            # scheduler will be re-created fresh below
        trainer_state = ckpt.get("trainer_state")
        if pos_rate_state is not None and isinstance(trainer_state, Mapping):
            pos_snapshot = trainer_state.get("pos_rate_ema")
            if pos_snapshot is not None:
                pos_rate_state.load_state_dict(pos_snapshot)
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
        best_loss_value = math.inf
        best_proxy_value = float("-inf")
        start_epoch = 1

    existing_calibrated_meta = read_selection_metadata(best_calibrated_path)
    if existing_calibrated_meta:
        cal_metrics = existing_calibrated_meta.get("metrics", {}) or {}
        cal_mean = cal_metrics.get("mean_event_f1")
        cal_mean_coerced = _coerce_float(cal_mean)
        if cal_mean_coerced is not None and math.isfinite(cal_mean_coerced):
            best_calibrated_value = cal_mean_coerced

    training_cfg = cast(MutableMapping[str, Any], cfg.setdefault("training", {}))
    metrics_cfg = cast(MutableMapping[str, Any], training_cfg.setdefault("metrics", {}))
    best_sel_cfg = cast(MutableMapping[str, Any], training_cfg.setdefault("best_selection", {}))
    aggregation_cfg = cast(MutableMapping[str, Any], metrics_cfg.setdefault("aggregation", {}))
    k_cfg = cast(MutableMapping[str, Any], aggregation_cfg.setdefault("k", {}))
    selection_mode = str(best_sel_cfg.get("mode", "calibrated_event")).lower()
    if selection_mode not in {"loss", "proxy_event", "calibrated_event"}:
        selection_mode = "calibrated_event"
    selection_trigger = str(best_sel_cfg.get("trigger", "on_proxy_improvement")).lower()
    if selection_trigger not in {"on_proxy_improvement", "every_n_epochs"}:
        selection_trigger = "on_proxy_improvement"
    selection_interval = _coerce_positive_int(best_sel_cfg.get("n")) or 1
    selection_write_back = bool(best_sel_cfg.get("write_back", False))
    selection_frames = _coerce_positive_int(best_sel_cfg.get("frames")) or 96
    selection_max_clips = _coerce_positive_int(best_sel_cfg.get("max_clips")) or 80
    selection_split = str(best_sel_cfg.get("split", "val"))
    sweep_cfg = best_sel_cfg.get("sweep", {}) if isinstance(best_sel_cfg, Mapping) else {}
    sweep_delta = float(_coerce_float(sweep_cfg.get("delta"), 0.05) or 0.05)
    sweep_low_guard = float(_coerce_float(sweep_cfg.get("low_guard"), 0.10) or 0.10)
    sweep_min_prob = float(_coerce_float(sweep_cfg.get("min_prob"), 0.02) or 0.02)
    sweep_max_prob = float(_coerce_float(sweep_cfg.get("max_prob"), 0.98) or 0.98)
    selection_temperature = _coerce_float(best_sel_cfg.get("temperature"))
    selection_bias = _coerce_float(best_sel_cfg.get("bias"))
    decoder_snapshot = decoder_snapshot_from_config(cfg)
    tolerance_snapshot = tolerance_snapshot_from_config(cfg)
    experiment_cfg_raw = cfg.get("experiment", {})
    experiment_cfg = experiment_cfg_raw if isinstance(experiment_cfg_raw, MutableMapping) else {}
    run_id = str(experiment_cfg.get("name", ""))
    selection_seed = _coerce_positive_int(experiment_cfg.get("seed"))
    selection_deterministic = bool(experiment_cfg.get("deterministic", False))
    autopilot_cfg = cfg.get("autopilot", {}) if isinstance(cfg, Mapping) else {}
    autopilot_best_cfg = autopilot_cfg.get("best_selection", {}) if isinstance(autopilot_cfg, Mapping) else {}
    best_owner = str(autopilot_best_cfg.get("owner", "autopilot")).lower()
    if best_owner not in {"autopilot", "train"}:
        best_owner = "autopilot"
    train_is_owner = best_owner == "train"
    primary_owner_label = best_owner
    onset_center = _coerce_float(metrics_cfg.get("prob_threshold_onset"))
    if onset_center is None:
        onset_center = _coerce_float(metrics_cfg.get("prob_threshold"), 0.4) or 0.4
    offset_center = _coerce_float(metrics_cfg.get("prob_threshold_offset"), onset_center)
    k_onset_cfg = _coerce_positive_int(k_cfg.get("onset")) or 1
    if _coerce_positive_int(k_cfg.get("onset")) is None:
        k_cfg["onset"] = k_onset_cfg
    if "offset" not in k_cfg or not int(k_cfg.get("offset", 0)):
        k_cfg["offset"] = 1
    selection_log_dir = ckpt_dir / "selection_logs"
    selection_log_dir.mkdir(parents=True, exist_ok=True)
        
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
        last_saved_for_epoch = False
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
                loss, parts = compute_loss_frame(out, batch, weights=w, pos_rate_state=pos_rate_state)
            else:
                # Guard: if model outputs (B,T,...) but we're using clip loss, pool over time
                if out["pitch_logits"].dim() == 3:
                    out = _time_pool_out_to_clip(out)
                loss, parts = compute_loss(out, tgt, crit, w, pos_rate_state=pos_rate_state)

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
            val_metrics = evaluate_one_epoch(
                model,
                val_loader,
                cfg,
                optimizer=optimizer,
                pos_rate_state=pos_rate_state,
            )
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

                val_total_raw = val_metrics.get("total")
                event_f1_mean_raw = val_metrics.get("event_f1_mean_proxy")
                event_f1_on_raw = val_metrics.get("event_f1_onset_proxy")
                event_f1_off_raw = val_metrics.get("event_f1_offset_proxy")

                event_f1_mean_val = _coerce_float(event_f1_mean_raw)
                event_f1_on_val = _coerce_float(event_f1_on_raw, 0.0) or 0.0
                event_f1_off_val = _coerce_float(event_f1_off_raw, 0.0) or 0.0
                onset_f1_val = _coerce_float(val_metrics.get("onset_f1_proxy"), event_f1_on_val) or event_f1_on_val
                offset_f1_val = _coerce_float(val_metrics.get("offset_f1_proxy"), event_f1_off_val) or event_f1_off_val
                onset_pred_rate_val = _coerce_float(val_metrics.get("onset_pred_rate"), 0.0) or 0.0
                onset_pos_rate_val = _coerce_float(val_metrics.get("onset_pos_rate"), 0.0) or 0.0

                checkpoint_best_event_f1 = max(best_proxy_value, best_calibrated_value)
                save_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    epoch,
                    cfg,
                    best_loss_value,
                    checkpoint_best_event_f1,
                    trainer_state=_snapshot_trainer_state(),
                )
                last_saved_for_epoch = True

                timestamp_now = time.time()
                proxy_result = SelectionResult(
                    onset_threshold=float(onset_center),
                    offset_threshold=float(offset_center if offset_center is not None else onset_center),
                    k_onset=int(k_onset_cfg),
                    onset_event_f1=float(event_f1_on_val),
                    offset_event_f1=float(event_f1_off_val),
                    mean_event_f1=float(event_f1_mean_val or 0.0),
                    onset_f1=float(onset_f1_val),
                    offset_f1=float(offset_f1_val),
                    onset_pred_rate=float(onset_pred_rate_val),
                    onset_pos_rate=float(onset_pos_rate_val),
                    decoder_kind=None,
                    decoder_settings={},
                )
                proxy_context = SelectionContext(
                    split=selection_split,
                    frames=int(selection_frames),
                    max_clips=int(selection_max_clips),
                    seed=selection_seed,
                    deterministic=selection_deterministic,
                    decoder=decoder_snapshot,
                    tolerances=tolerance_snapshot,
                    sweep={
                        "delta": sweep_delta,
                        "clamp_min": sweep_min_prob,
                        "clamp_max": sweep_max_prob,
                        "low_guard": sweep_low_guard,
                        "k_onset": [k_onset_cfg],
                    },
                    temperature=selection_temperature,
                    bias=selection_bias,
                    temperature_onset=selection_temperature,
                    temperature_offset=selection_temperature,
                    bias_onset=selection_bias,
                    bias_offset=selection_bias,
                    run_id=run_id,
                    start_time=timestamp_now,
                    end_time=timestamp_now,
                )

                if val_total_raw is not None:
                    val_total_float = float(val_total_raw)
                    if val_total_float < best_loss_value:
                        best_loss_value = val_total_float
                        record_best(
                            source=last_path,
                            destination=best_loss_path,
                            result=proxy_result,
                            context=proxy_context,
                            repo_root=REPO,
                            metadata_extra={"kind": "loss", "val_total": val_total_float, "epoch": int(epoch)},
                        )
                        if train_is_owner and selection_mode == "loss":
                            record_best(
                                source=last_path,
                                destination=primary_best_path,
                                result=proxy_result,
                                context=proxy_context,
                                repo_root=REPO,
                                metadata_extra={"kind": "primary", "val_total": val_total_float, "epoch": int(epoch)},
                            )

                if event_f1_mean_val is not None and math.isfinite(event_f1_mean_val):
                    if event_f1_mean_val > best_proxy_value + 1e-9:
                        best_proxy_value = event_f1_mean_val
                        record_best(
                            source=last_path,
                            destination=best_proxy_path,
                            result=proxy_result,
                            context=proxy_context,
                            repo_root=REPO,
                            metadata_extra={"kind": "proxy_event", "epoch": int(epoch)},
                        )
                        if train_is_owner and selection_mode == "proxy_event":
                            record_best(
                                source=last_path,
                                destination=primary_best_path,
                                result=proxy_result,
                                context=proxy_context,
                                repo_root=REPO,
                                metadata_extra={"kind": "primary", "epoch": int(epoch)},
                            )

                calibrated_ran = False
                if selection_mode == "calibrated_event" and event_f1_mean_val is not None and math.isfinite(event_f1_mean_val):
                    spacing_ready = (epoch - last_calibration_epoch) >= selection_interval
                    should_calibrate = False
                    if selection_trigger == "every_n_epochs":
                        should_calibrate = spacing_ready
                    elif selection_trigger == "on_proxy_improvement":
                        improved_proxy = event_f1_mean_val > best_proxy_trigger_value + 1e-9
                        should_calibrate = spacing_ready and improved_proxy
                        if improved_proxy:
                            best_proxy_trigger_value = event_f1_mean_val
                    if should_calibrate:
                        sweep = SweepSpec(
                            onset_center=float(onset_center),
                            offset_center=float(offset_center if offset_center is not None else onset_center),
                            delta=sweep_delta,
                            clamp_min=sweep_min_prob,
                            clamp_max=sweep_max_prob,
                            low_guard=sweep_low_guard,
                        )
                        sweep.k_onset_candidates = (int(k_onset_cfg),)
                        selection_request = SelectionRequest(
                            ckpt=last_path,
                            split=selection_split,
                            frames=int(selection_frames),
                            max_clips=int(selection_max_clips),
                            sweep=sweep,
                            decoder=decoder_snapshot,
                            tolerances=tolerance_snapshot,
                            temperature=selection_temperature,
                            bias=selection_bias,
                            seed=selection_seed,
                            deterministic=selection_deterministic,
                            log_path=selection_log_dir / f"epoch_{epoch:03d}.txt",
                            run_id=run_id,
                        )
                        try:
                            calibrated_result, calibrated_context, _ = calibrate_and_score(selection_request)
                        except SelectionError as exc:
                            logger.warning("[train] calibrated selection failed: %s", exc)
                        else:
                            calibrated_ran = True
                            last_calibration_epoch = epoch
                            if calibrated_result.mean_event_f1 > best_calibrated_value + 1e-9:
                                best_calibrated_value = calibrated_result.mean_event_f1
                                record_best(
                                    source=last_path,
                                    destination=best_calibrated_path,
                                    result=calibrated_result,
                                    context=calibrated_context,
                                    repo_root=REPO,
                                    metadata_extra={"kind": "calibrated_event", "epoch": int(epoch)},
                                )
                                if train_is_owner:
                                    record_best(
                                        source=last_path,
                                        destination=primary_best_path,
                                        result=calibrated_result,
                                        context=calibrated_context,
                                        repo_root=REPO,
                                        metadata_extra={"kind": "primary", "epoch": int(epoch)},
                                    )
                            if selection_write_back:
                                _update_selection_in_config(calibrated_result, calibrated_context, cfg, decoder_snapshot)
                    elif selection_trigger == "on_proxy_improvement" and event_f1_mean_val is not None:
                        best_proxy_trigger_value = max(best_proxy_trigger_value, event_f1_mean_val)
                elif selection_mode != "calibrated_event" and event_f1_mean_val is not None:
                    best_proxy_trigger_value = max(best_proxy_trigger_value, event_f1_mean_val)

                updated_best_event_f1 = max(best_proxy_value, best_calibrated_value)
                if calibrated_ran and updated_best_event_f1 > checkpoint_best_event_f1 + 1e-12:
                    checkpoint_best_event_f1 = updated_best_event_f1
                    save_checkpoint(
                        last_path,
                        model,
                        optimizer,
                        epoch,
                        cfg,
                        best_loss_value,
                        checkpoint_best_event_f1,
                        trainer_state=_snapshot_trainer_state(),
                    )

                loss_label = f"{best_loss_value:.3f}" if math.isfinite(best_loss_value) else "inf"
                proxy_label = f"{best_proxy_value:.3f}" if math.isfinite(best_proxy_value) else "n/a"
                calibrated_label = f"{best_calibrated_value:.3f}" if math.isfinite(best_calibrated_value) else "n/a"
                summary_line = (
                    f"loss={loss_label} ({best_loss_path.name}) | "
                    f"proxy_event={proxy_label} ({best_proxy_path.name}) | "
                    f"calibrated_event={calibrated_label} ({best_calibrated_path.name}) | "
                    f"owner={primary_owner_label}"
                )
                print(f"[best-checkpoints] {summary_line}")
                logger.info(
                    "[best-checkpoints] loss=%s path=%s | proxy=%s path=%s | calibrated=%s path=%s owner=%s",
                    loss_label,
                    best_loss_path,
                    proxy_label,
                    best_proxy_path,
                    calibrated_label,
                    best_calibrated_path,
                    primary_owner_label,
                )

                metrics = dict(val_metrics)
                metrics["onset_thr"] = float(proxy_result.onset_threshold)
                metrics["offset_thr"] = float(proxy_result.offset_threshold)
                metrics["ev_f1_mean"] = float(proxy_result.mean_event_f1)
                metrics["onset_event_f1"] = float(proxy_result.onset_event_f1)
                metrics["offset_event_f1"] = float(proxy_result.offset_event_f1)
                metrics["onset_f1"] = float(onset_f1_val)
                metrics["offset_f1"] = float(offset_f1_val)
                metrics["onset_pred_rate"] = float(onset_pred_rate_val)
                metrics["onset_pos_rate"] = float(onset_pos_rate_val)
                metrics["k_onset"] = int(proxy_result.k_onset)
                for head_name, head_values in (decoder_snapshot or {}).items():
                    if not isinstance(head_values, Mapping):
                        continue
                    for key_name, key_value in head_values.items():
                        metric_key = f"decoder_{head_name}_{key_name}"
                        if isinstance(key_value, (int, float)) and math.isfinite(float(key_value)):
                            metrics[metric_key] = float(key_value)

        epoch_finish = getattr(train_base_dataset, "finish_epoch_snapshot", None)
        if callable(epoch_finish):
            try:
                epoch_finish()
            except Exception as exc:
                logger.warning("[train] failed to finalize epoch snapshot: %s", exc)

        # --- always save LAST ---
        state_best_event_f1 = max(best_proxy_value, best_calibrated_value)
        if not last_saved_for_epoch:
            save_checkpoint(
                last_path,
                model,
                optimizer,
                epoch,
                cfg,
                best_loss_value,
                state_best_event_f1,
                trainer_state=_snapshot_trainer_state(),
            )
        if (epoch % save_every) == 0:
            # optional per-epoch named snapshot
            save_checkpoint(
                ckpt_dir / f"tivit_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                cfg,
                best_loss_value,
                state_best_event_f1,
                trainer_state=_snapshot_trainer_state(),
            )

    # close writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
