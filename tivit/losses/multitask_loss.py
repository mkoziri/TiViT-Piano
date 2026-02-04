"""Multitask loss composition for TiViT-Piano heads.

Purpose:
    - Combine pitch/onset/offset/hand/clef losses with adaptive weighting.
    - Support frame/clip modes, per-tile supervision, and optional hand gating.
    - Enforce a single explicit config source at ``training.loss`` and honor ``priors.enabled`` for training-time priors.
Key Functions/Classes:
    - MultitaskLoss: composite loss with time alignment and priors.
    - OnOffPosWeightEMA: EMA tracker for adaptive pos_weight per head/scope.
    - build_loss(): factory that validates and returns a MultitaskLoss.
CLI Arguments:
    (none)
Usage:
    loss_fn = build_loss(cfg)  # cfg with training.loss
    loss, parts = loss_fn(preds, targets)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tivit.decoder.decode import pool_roll_BT
from tivit.priors.base import compute_prior_mean_regularizer
from tivit.priors.hand_gating import build_reach_weights
from tivit.utils.imbalance import map_ratio_to_band

LOGGER = logging.getLogger(__name__)
_ADAPTIVE_BAND_LOGGED: set[Tuple[str, str]] = set()
HEAD_ORDER = ("pitch", "onset", "offset", "hand", "clef")
PITCH_POS_WEIGHT_MODES = {"adaptive", "adaptive_band", "fixed", "sqrt", "off", "none"}
ONOFF_POS_WEIGHT_MODES = {"adaptive", "adaptive_band", "fixed", "ema", "off", "none"}
ONOFF_LOSSES = {"bce", "bce_pos", "bce_with_logits", "focal", "focal_bce"}
PITCH_LOSSES = {"bce", "bce_pos", "bce_with_logits"}
HAND_CLEF_LOSSES = {"ce"}


def _coerce_pos_weight_band(value: Any) -> Optional[Tuple[float, float]]:
    """Normalize a band config into (low, high) floats."""
    if isinstance(value, Mapping):
        value = value.get("band", value.get("values", value.get("range", value)))
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("pos_weight_band must be a 2-element sequence or null")
    if len(value) != 2:
        raise ValueError("pos_weight_band must contain exactly two numeric entries")
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError):
        raise ValueError("pos_weight_band values must be numeric") from None
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("pos_weight_band values must be finite")
    if high < low:
        low, high = high, low
    return (low, high)


def _pos_weight_from_rate(
    rate: torch.Tensor,
    *,
    eps: float = 1e-6,
    clamp_min: float = 1.0,
    clamp_max: float = 100.0,
) -> torch.Tensor:
    """Convert observed positive rate into BCE pos_weight with clamping."""
    rate_clamped = rate.clone().clamp(min=eps, max=1.0 - eps)
    pos_weight = (1.0 - rate_clamped) / rate_clamped
    return pos_weight.clamp_(min=clamp_min, max=clamp_max)


def _adaptive_pos_weight_from_roll(target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute per-class pos_weight from target roll frequencies."""
    flat = target.detach().reshape(-1, target.shape[-1]).float()
    pos = flat.mean(dim=0).clamp(min=eps, max=1.0 - eps)
    return ((1.0 - pos) / (pos + eps)).clamp(1.0, 100.0)


def _banded_pos_weight_from_roll(
    target: torch.Tensor,
    band: Tuple[float, float],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map per-class positive rarity (neg/pos) into a configured band."""
    flat = target.detach().reshape(-1, target.shape[-1]).float()
    pos = flat.mean(dim=0).clamp(min=eps, max=1.0 - eps)
    neg = (1.0 - pos).clamp_min(eps)
    ratio = neg / pos  # larger when positives are rarer
    return map_ratio_to_band(ratio, band, eps=eps)


def _load_pos_weight_file(path: str | Path) -> Mapping[str, Any]:
    """Load precomputed pos_weight values from JSON."""
    path_obj = Path(path).expanduser()
    if not path_obj.exists():
        raise ValueError(f"pos_weight_path not found: {path_obj}")
    data = json.loads(path_obj.read_text())
    if isinstance(data, Mapping) and isinstance(data.get("weights"), Mapping):
        data = data["weights"]
    if not isinstance(data, Mapping):
        raise ValueError("pos_weight file must contain a mapping of head -> weights")
    return data


def _log_banded_weight_stats(head: str, context: str, weights: torch.Tensor, band: Tuple[float, float]) -> None:
    """Log one-time diagnostics for banded weights to avoid noisy logs."""
    key = (context, head)
    if key in _ADAPTIVE_BAND_LOGGED:
        return
    w_min = float(weights.min().item())
    w_mean = float(weights.mean().item())
    w_max = float(weights.max().item())
    LOGGER.info(
        "[onoff_loss] head=%s mode=adaptive_band context=%s band=[%.2f, %.2f] pos_weight min=%.3f mean=%.3f max=%.3f",
        head,
        context,
        band[0],
        band[1],
        w_min,
        w_mean,
        w_max,
    )
    _ADAPTIVE_BAND_LOGGED.add(key)


def _dynamic_pos_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight_override: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BCEWithLogits using adaptive or provided positive class weighting."""
    eps = 1e-6
    target_float = targets.float()
    if pos_weight_override is not None:
        pos_weight = pos_weight_override.detach().to(device=logits.device, dtype=logits.dtype)
    else:
        positive_rate = target_float.mean().clamp(min=eps, max=1.0 - eps)
        pos_weight = ((1.0 - positive_rate) / positive_rate).clamp(1.0, 100.0)

    pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    w = weight.to(device=logits.device, dtype=logits.dtype) if weight is not None else None

    return F.binary_cross_entropy_with_logits(
        logits,
        target_float,
        weight=w,
        pos_weight=pos_weight,
        reduction="mean",
    )




def _pos_weight_tensor(value: Any, size: int, device: torch.device, ctx: str) -> torch.Tensor:
    """Build a per-class pos_weight tensor for BCE losses."""
    if value is None:
        raise ValueError(f"{ctx} pos_weight is required but missing")
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float32).flatten()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tensor = torch.tensor([_as_float(v, f"{ctx}[{i}]") for i, v in enumerate(value)],
                              device=device, dtype=torch.float32)
    else:
        return torch.full((size,), _as_float(value, ctx), dtype=torch.float32, device=device)
    if tensor.numel() != size:
        raise ValueError(f"{ctx} pos_weight expected {size} values, got {tensor.numel()}")
    return tensor


def _time_pool_out_to_clip(out: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """Reduce time-distributed logits when computing clip-mode losses."""
    pooled = dict(out)
    for key in ("pitch_logits", "onset_logits", "offset_logits", "hand_logits", "clef_logits"):
        if key in out and out[key].dim() == 3:
            pooled[key] = out[key].mean(dim=1)
    return pooled


def _interp_labels_BT(x_bt: torch.Tensor, Tprime: int) -> torch.Tensor:
    """Interpolate integer labels along time with nearest sampling."""
    x = x_bt.float().unsqueeze(1)
    x = F.interpolate(x, size=Tprime, mode="nearest")
    return x.squeeze(1).long()


def _interp_mask_BT(mask_bt: torch.Tensor, Tprime: int) -> torch.Tensor:
    """Interpolate a (B,T) mask to T' using nearest sampling."""
    m = mask_bt.float().unsqueeze(1)
    m = F.interpolate(m, size=Tprime, mode="nearest")
    return (m.squeeze(1) > 0.5)


def _match_pitch_dim(x: Optional[torch.Tensor], P: int) -> Optional[torch.Tensor]:
    """Align pitch dimension to model head size, padding/cropping as needed."""
    if x is None:
        return None
    if x.shape[-1] == P:
        return x
    if x.shape[-1] > P:
        start = 0
        if x.shape[-1] == 128 and P == 88:
            start = 21
        return x[..., start : start + P]
    pad = P - x.shape[-1]
    if pad <= 0:
        return x
    return F.pad(x, (0, pad))


class PosRateEMA:
    """Track an exponential moving average for positive class rates."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.value: Optional[torch.Tensor] = None

    def process(self, observation: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        """Update EMA with a new observation and return the smoothed value."""
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
        """Return a copy of the current EMA without updating."""
        return None if self.value is None else self.value.clone()

    def state_dict(self) -> Dict[str, Any]:
        """Serialize EMA state for checkpointing."""
        return {"value": None if self.value is None else self.value.tolist()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore EMA state from checkpoint data."""
        value = state.get("value") if isinstance(state, Mapping) else None
        if value is None:
            self.value = None
        else:
            self.value = torch.tensor(value, dtype=torch.float32)


class OnOffPosWeightEMA:
    """Maintain EMA statistics for onset/offset heads (clip & frame modes)."""

    def __init__(self, alpha: float) -> None:
        """Initialise clip/frame EMA trackers with smoothing factor ``alpha``."""
        alpha = float(alpha)
        self.alpha = alpha
        self.clip = {"onset": PosRateEMA(alpha), "offset": PosRateEMA(alpha)}
        self.frame = {"onset": PosRateEMA(alpha), "offset": PosRateEMA(alpha)}

    @staticmethod
    def _reduce_clip_targets(targets: torch.Tensor) -> torch.Tensor:
        """Reduce clip targets to a scalar positive rate."""
        return torch.tensor([float(targets.float().mean().item())], dtype=torch.float32)

    @staticmethod
    def _reduce_frame_roll(roll: torch.Tensor) -> torch.Tensor:
        """Reduce frame-level roll to per-pitch positive rates."""
        P = roll.shape[-1]
        return roll.reshape(-1, P).float().mean(dim=0).cpu()

    def clip_pos_weight(self, head: str, targets: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        """Compute clip-level pos_weight via EMA for a head."""
        tracker = self.clip[head]
        rate = tracker.process(self._reduce_clip_targets(targets), update=update)
        return _pos_weight_from_rate(rate)

    def frame_pos_weight(self, head: str, roll: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        """Compute frame-level pos_weight via EMA for a head."""
        tracker = self.frame[head]
        rate = tracker.process(self._reduce_frame_roll(roll), update=update)
        return _pos_weight_from_rate(rate)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize nested EMA state for checkpointing."""
        return {
            "alpha": self.alpha,
            "clip": {k: v.state_dict() for k, v in self.clip.items()},
            "frame": {k: v.state_dict() for k, v in self.frame.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore nested EMA state from checkpoint data."""
        if not isinstance(state, Mapping):
            return
        for scope_name, registry in (("clip", self.clip), ("frame", self.frame)):
            saved_scope = state.get(scope_name, {}) if isinstance(state.get(scope_name), Mapping) else {}
            for head, tracker in registry.items():
                tracker_state = saved_scope.get(head)
                if tracker_state is not None:
                    tracker.load_state_dict(tracker_state)


def _validate_required(mapping: Mapping[str, Any], required: set[str], allowed: set[str], ctx: str) -> None:
    missing = required - set(mapping.keys())
    if missing:
        raise ValueError(f"{ctx} missing required keys: {sorted(missing)}")
    unknown = set(mapping.keys()) - allowed
    if unknown:
        raise ValueError(f"{ctx} has unknown keys: {sorted(unknown)}")


def _as_float(value: Any, ctx: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{ctx} must be a float") from None
    if not math.isfinite(out):
        raise ValueError(f"{ctx} must be finite")
    return out


def _validate_per_tile(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {"enabled", "heads", "mask_cushion_keys", "debug"}
    _validate_required(cfg, {"enabled", "heads"}, allowed, "training.loss.per_tile")

    debug_cfg = cfg.get("debug", {}) or {}
    if not isinstance(debug_cfg, Mapping):
        raise ValueError("training.loss.per_tile.debug must be a mapping")
    _validate_required(debug_cfg, set(), {"enabled", "interval"}, "training.loss.per_tile.debug")

    heads_val = cfg.get("heads")
    if not isinstance(heads_val, Sequence) or isinstance(heads_val, (str, bytes)):
        raise ValueError("training.loss.per_tile.heads must be a sequence")
    heads = tuple(str(h).lower() for h in heads_val)

    per_tile = {
        "enabled": bool(cfg.get("enabled", False)),
        "heads": heads,
    }
    if "mask_cushion_keys" in cfg:
        cushion = cfg.get("mask_cushion_keys")
        per_tile["mask_cushion_keys"] = None if cushion is None else int(cushion)
    if debug_cfg:
        per_tile["debug"] = {
            "enabled": bool(debug_cfg.get("enabled", False)),
            "interval": int(debug_cfg.get("interval", 0) or 0),
        }
    return per_tile


def _validate_head_block(name: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if name == "pitch":
        required = {"loss", "pos_weight_mode", "pos_weight", "pos_weight_band"}
        allowed = required
        _validate_required(cfg, required, allowed, f"training.loss.heads.{name}")
        loss = str(cfg["loss"]).lower()
        if loss not in PITCH_LOSSES:
            raise ValueError(f"training.loss.heads.{name}.loss must be one of {sorted(PITCH_LOSSES)}")
        mode = str(cfg["pos_weight_mode"]).lower()
        if mode not in PITCH_POS_WEIGHT_MODES:
            raise ValueError(f"training.loss.heads.{name}.pos_weight_mode must be one of {sorted(PITCH_POS_WEIGHT_MODES)}")
        pos_weight = cfg.get("pos_weight")
        if mode == "fixed":
            if pos_weight is None:
                raise ValueError(f"training.loss.heads.{name}.pos_weight is required when pos_weight_mode='fixed'")
            pos_weight = _as_float(pos_weight, f"training.loss.heads.{name}.pos_weight")
        elif pos_weight is not None:
            pos_weight = _as_float(pos_weight, f"training.loss.heads.{name}.pos_weight")
        band = cfg.get("pos_weight_band")
        band_val = _coerce_pos_weight_band(band) if band is not None else None
        if mode == "adaptive_band" and band_val is None:
            raise ValueError(f"training.loss.heads.{name}.pos_weight_band is required for pos_weight_mode='adaptive_band'")
        return {
            "loss": loss,
            "pos_weight_mode": mode,
            "pos_weight": pos_weight,
            "pos_weight_band": band_val,
        }

    if name in {"onset", "offset"}:
        required = {
            "loss",
            "pos_weight_mode",
            "pos_weight",
            "pos_weight_band",
            "focal_gamma",
            "focal_alpha",
            "prior_mean",
            "prior_weight",
        }
        allowed = required
        _validate_required(cfg, required, allowed, f"training.loss.heads.{name}")
        loss = str(cfg["loss"]).lower()
        if loss not in ONOFF_LOSSES:
            raise ValueError(f"training.loss.heads.{name}.loss must be one of {sorted(ONOFF_LOSSES)}")
        mode = str(cfg["pos_weight_mode"]).lower()
        if mode not in ONOFF_POS_WEIGHT_MODES:
            raise ValueError(f"training.loss.heads.{name}.pos_weight_mode must be one of {sorted(ONOFF_POS_WEIGHT_MODES)}")
        if loss in {"focal", "focal_bce"} and mode not in {"off", "none"}:
            raise ValueError(f"training.loss.heads.{name}: focal loss requires pos_weight_mode='off'")
        pos_weight = cfg.get("pos_weight")
        if mode == "fixed":
            if pos_weight is None:
                raise ValueError(f"training.loss.heads.{name}.pos_weight is required when pos_weight_mode='fixed'")
            pos_weight = _as_float(pos_weight, f"training.loss.heads.{name}.pos_weight")
        elif pos_weight is not None:
            pos_weight = _as_float(pos_weight, f"training.loss.heads.{name}.pos_weight")
        band = cfg.get("pos_weight_band")
        band_val = _coerce_pos_weight_band(band) if band is not None else None
        if mode == "adaptive_band" and band_val is None:
            raise ValueError(f"training.loss.heads.{name}.pos_weight_band is required for pos_weight_mode='adaptive_band'")
        focal_gamma = _as_float(cfg["focal_gamma"], f"training.loss.heads.{name}.focal_gamma")
        focal_alpha = _as_float(cfg["focal_alpha"], f"training.loss.heads.{name}.focal_alpha")
        prior_mean = _as_float(cfg["prior_mean"], f"training.loss.heads.{name}.prior_mean")
        prior_weight = _as_float(cfg["prior_weight"], f"training.loss.heads.{name}.prior_weight")
        return {
            "loss": loss,
            "pos_weight_mode": mode,
            "pos_weight": pos_weight,
            "pos_weight_band": band_val,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            "prior_mean": prior_mean,
            "prior_weight": prior_weight,
        }

    if name in {"hand", "clef"}:
        required = {"loss"}
        allowed = required
        _validate_required(cfg, required, allowed, f"training.loss.heads.{name}")
        loss = str(cfg["loss"]).lower()
        if loss not in HAND_CLEF_LOSSES:
            raise ValueError(f"training.loss.heads.{name}.loss must be one of {sorted(HAND_CLEF_LOSSES)}")
        return {"loss": loss}

    raise ValueError(f"Unknown head '{name}' in training.loss.heads")


def _validate_loss_schema(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg, Mapping):
        raise ValueError("MultitaskLoss expects a mapping config containing training.loss")
    training_cfg = cfg.get("training")
    if not isinstance(training_cfg, Mapping):
        raise ValueError("Config is missing 'training' mapping required for loss construction")
    loss_cfg = training_cfg.get("loss")
    if not isinstance(loss_cfg, Mapping):
        raise ValueError("Config is missing 'training.loss' mapping required for loss construction")

    allowed_top = {"head_weights", "ema_alpha", "neg_smooth_onoff", "per_tile", "heads", "pos_weight_path"}
    _validate_required(loss_cfg, allowed_top, allowed_top, "training.loss")

    head_weights_raw = loss_cfg.get("head_weights")
    if not isinstance(head_weights_raw, Mapping):
        raise ValueError("training.loss.head_weights must be a mapping")
    _validate_required(head_weights_raw, set(HEAD_ORDER), set(HEAD_ORDER), "training.loss.head_weights")
    head_weights = {k: _as_float(head_weights_raw[k], f"training.loss.head_weights.{k}") for k in HEAD_ORDER}

    ema_alpha = _as_float(loss_cfg.get("ema_alpha"), "training.loss.ema_alpha")
    if ema_alpha < 0.0:
        raise ValueError("training.loss.ema_alpha must be non-negative")
    neg_smooth = _as_float(loss_cfg.get("neg_smooth_onoff"), "training.loss.neg_smooth_onoff")
    if neg_smooth < 0.0:
        raise ValueError("training.loss.neg_smooth_onoff must be non-negative")

    per_tile_cfg = loss_cfg.get("per_tile")
    if not isinstance(per_tile_cfg, Mapping):
        raise ValueError("training.loss.per_tile must be a mapping")
    per_tile = _validate_per_tile(per_tile_cfg)

    heads_cfg = loss_cfg.get("heads")
    pos_weight_path = loss_cfg.get("pos_weight_path")
    if pos_weight_path is not None:
        pos_weight_path = str(pos_weight_path)

    if not isinstance(heads_cfg, Mapping):
        raise ValueError("training.loss.heads must be a mapping")
    _validate_required(heads_cfg, set(HEAD_ORDER), set(HEAD_ORDER), "training.loss.heads")

    heads: Dict[str, Dict[str, Any]] = {}
    for head in HEAD_ORDER:
        heads[head] = _validate_head_block(head, heads_cfg.get(head, {}))

    return {
        "head_weights": head_weights,
        "ema_alpha": ema_alpha,
        "neg_smooth_onoff": neg_smooth,
        "per_tile": per_tile,
        "pos_weight_path": pos_weight_path,
        "heads": heads,
    }


def _per_tile_masked_loss(
    logits_tile: torch.Tensor,
    roll: torch.Tensor,
    mask: torch.Tensor,
    cfg: Mapping[str, Any],
    pos_weight: Optional[torch.Tensor],
    *,
    head: str,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """Masked per-tile loss plus minimal debug stats; returns (loss, debug)."""
    if not torch.is_tensor(logits_tile) or not torch.is_tensor(mask):
        return None, None
    if logits_tile.dim() != 4 or mask.dim() != 3:
        return None, None
    B, T_logits, tiles, P_logits = logits_tile.shape
    if mask.shape[0] != B or mask.shape[1] != tiles or mask.shape[2] != P_logits:
        return None, None

    target = roll.unsqueeze(2).expand_as(logits_tile)
    mask_time = mask.to(device=logits_tile.device, dtype=logits_tile.dtype).unsqueeze(1)
    mask_time = mask_time.expand_as(logits_tile)
    supervised = mask_time.sum()
    if supervised <= 0:
        return None, None

    mode = str(cfg.get("loss", "bce_pos")).lower()
    if mode in {"focal", "focal_bce"}:
        gamma = float(cfg.get("focal_gamma", 2.0))
        alpha = float(cfg.get("focal_alpha", 0.25))
        base = F.binary_cross_entropy_with_logits(logits_tile, target, reduction="none")
        probs = torch.sigmoid(logits_tile)
        weight = (alpha * (1.0 - probs).pow(gamma)).detach()
        per_elem_loss = base * weight
    else:
        pos_weight_local = None
        if torch.is_tensor(pos_weight):
            pos_weight_local = pos_weight.to(device=logits_tile.device, dtype=logits_tile.dtype)
        per_elem_loss = F.binary_cross_entropy_with_logits(
            logits_tile,
            target,
            pos_weight=pos_weight_local,
            reduction="none",
        )

    weighted_loss = per_elem_loss * mask_time
    loss = weighted_loss.sum() / supervised.clamp_min(1e-6)

    probs_detached = torch.sigmoid(logits_tile.detach())
    tile_loss = weighted_loss.sum(dim=(0, 1, 3))
    tile_weight = mask_time.sum(dim=(0, 1, 3)).clamp_min(1e-6)
    tile_loss_mean = tile_loss / tile_weight

    active_weight = mask_time * target
    inactive_weight = mask_time * (1.0 - target)
    active_sum = (probs_detached * active_weight).sum(dim=(0, 1, 3))
    inactive_sum = (probs_detached * inactive_weight).sum(dim=(0, 1, 3))
    active_count = active_weight.sum(dim=(0, 1, 3)).clamp_min(1e-6)
    inactive_count = inactive_weight.sum(dim=(0, 1, 3)).clamp_min(1e-6)
    debug = {
        "loss": tile_loss_mean.detach().cpu().tolist(),
        "prob_active": (active_sum / active_count).detach().cpu().tolist(),
        "prob_inactive": (inactive_sum / inactive_count).detach().cpu().tolist(),
        "head": str(head),
    }
    return loss, debug


class MultitaskLoss:
    """Composite loss for pitch/onset/offset/hand/clef heads."""

    def __init__(self, cfg: Mapping[str, Any] | None = None, head_weights: Mapping[str, float] | None = None) -> None:
        """Initialise loss wiring with strict config validation."""
        if head_weights is not None:
            raise ValueError("head_weights override is not supported; configure training.loss.head_weights instead")
        validated = _validate_loss_schema(cfg or {})
        self.head_weights = validated["head_weights"]
        self.neg_smooth = validated["neg_smooth_onoff"]
        self.per_tile_defaults = validated["per_tile"]
        self.head_cfgs = validated["heads"]
        self.pitch_cfg = self.head_cfgs["pitch"]
        self.onset_cfg = self.head_cfgs["onset"]
        self.offset_cfg = self.head_cfgs["offset"]
        self.hand_cfg = self.head_cfgs["hand"]
        self.clef_cfg = self.head_cfgs["clef"]
        uses_ema = any(self.head_cfgs[h]["pos_weight_mode"] == "ema" for h in ("onset", "offset"))
        ema_alpha = validated["ema_alpha"]
        if uses_ema and ema_alpha <= 0.0:
            raise ValueError("training.loss.ema_alpha must be > 0 when any head uses pos_weight_mode='ema'")
        self.pos_rate_state = OnOffPosWeightEMA(ema_alpha) if uses_ema and ema_alpha > 0.0 else None
        priors_cfg = cfg.get("priors", {}) if isinstance(cfg, Mapping) else {}
        if not isinstance(priors_cfg, Mapping):
            priors_cfg = {}
        self.priors_enabled = bool(priors_cfg.get("enabled", False))
        self.hand_gating_cfg = priors_cfg.get("hand_gating", {}) if self.priors_enabled else {}

        self.precomputed_pos_weights = None
        pos_weight_path = validated.get("pos_weight_path")
        if pos_weight_path:
            self.precomputed_pos_weights = _load_pos_weight_file(pos_weight_path)

    def state_dict(self) -> Dict[str, Any]:
        """Return serialisable adaptive state (EMA trackers)."""
        return {"pos_rate_state": None if self.pos_rate_state is None else self.pos_rate_state.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load adaptive state while tolerating partial checkpoints."""
        if not isinstance(state, Mapping):
            return
        pos_state = state.get("pos_rate_state")
        if pos_state and self.pos_rate_state is not None:
            self.pos_rate_state.load_state_dict(pos_state)

    def __call__(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        update_state: bool = True,
        per_tile: Optional[Mapping[str, Any]] = None,
        hand_gating: Optional[Mapping[str, Any]] = None,
        pos_rate_state: Optional[OnOffPosWeightEMA] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Dispatch to clip- or frame-mode loss depending on logits rank."""
        pitch_logits = preds.get("pitch_logits")
        if pitch_logits is None:
            raise ValueError("Missing pitch_logits in predictions")
        if pitch_logits.dim() == 2:
            return self._clip_loss(preds, targets, update_state=update_state, pos_rate_state=pos_rate_state)
        if pitch_logits.dim() == 3:
            return self._frame_loss(
                preds,
                targets,
                update_state=update_state,
                per_tile=per_tile,
                hand_gating=hand_gating,
                pos_rate_state=pos_rate_state,
            )
        raise ValueError(f"Unsupported pitch_logits rank {pitch_logits.dim()}")

    def _pitch_pos_weight(self, target: torch.Tensor, device: torch.device, *, context: str) -> Optional[torch.Tensor]:
        mode = self.pitch_cfg["pos_weight_mode"]
        band = self.pitch_cfg["pos_weight_band"]
        pos_weight = self.pitch_cfg.get("pos_weight")
        eps = 1e-6
        if mode in {"off", "none"}:
            return None
        if mode == "fixed":
            return _pos_weight_tensor(pos_weight, target.shape[-1], device, "pitch pos_weight")
        if mode == "adaptive_band":
            if self.precomputed_pos_weights and "pitch" in self.precomputed_pos_weights:
                return _pos_weight_tensor(self.precomputed_pos_weights["pitch"], target.shape[-1], device, "pitch pos_weight")
            if band is None:
                raise ValueError("pitch pos_weight_mode='adaptive_band' requires pos_weight_band")
            weights_local = _banded_pos_weight_from_roll(target, band, eps)
            _log_banded_weight_stats("pitch", context, weights_local, band)
            return weights_local.to(device=device)
        if mode == "sqrt":
            pos_rate = target.reshape(-1, target.shape[-1]).float().mean(dim=0).clamp_min(eps)
            return ((1.0 - pos_rate) / (pos_rate + eps)).sqrt().clamp(1.0, 50.0).to(device=device)
        return _adaptive_pos_weight_from_roll(target, eps).to(device=device)

    def _onoff_pos_weight(
        self,
        head: str,
        cfg: Mapping[str, Any],
        roll: torch.Tensor,
        device: torch.device,
        state: Optional[OnOffPosWeightEMA],
        *,
        update_state: bool,
        context: str,
    ) -> Optional[torch.Tensor]:
        mode = cfg.get("pos_weight_mode", "adaptive")
        band = cfg.get("pos_weight_band")
        pos_weight = cfg.get("pos_weight")
        if mode in {"off", "none"}:
            return None
        if mode == "fixed":
            return _pos_weight_tensor(pos_weight, roll.shape[-1], device, f"{head} pos_weight")
        if mode == "ema":
            if state is None:
                raise ValueError("pos_weight_mode='ema' requires training.loss.ema_alpha > 0")
            fn = state.clip_pos_weight if context == "clip" else state.frame_pos_weight
            return fn(head, roll, update=update_state).to(device=device)
        if mode == "adaptive_band":
            if self.precomputed_pos_weights and head in self.precomputed_pos_weights:
                return _pos_weight_tensor(self.precomputed_pos_weights[head], roll.shape[-1], device, f"{head} pos_weight")
            if band is None:
                raise ValueError(f"{head} pos_weight_mode='adaptive_band' requires pos_weight_band")
            weights_local = _banded_pos_weight_from_roll(roll, band)
            _log_banded_weight_stats(head, context, weights_local, band)
            return weights_local.to(device=device)
        return _adaptive_pos_weight_from_roll(roll).to(device=device)

    # ---- clip-mode loss -------------------------------------------------
    def _clip_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        update_state: bool,
        pos_rate_state: Optional[OnOffPosWeightEMA],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Clip-mode loss with optional adaptive pos_weight and priors."""
        state = pos_rate_state or self.pos_rate_state
        out = _time_pool_out_to_clip(preds)
        device = next(iter(out.values())).device if out else torch.device("cpu")
        total = torch.tensor(0.0, device=device)
        parts: Dict[str, Any] = {}

        pitch_target = targets.get("pitch")
        if pitch_target is None:
            pitch_target = torch.zeros_like(out["pitch_logits"])
        elif pitch_target.dim() == 3:
            pitch_target = pitch_target.max(dim=1).values

        pitch_pos_weight = self._pitch_pos_weight(pitch_target, device, context="clip")

        def _clip_head_loss(
            head: str,
            cfg: Mapping[str, Any],
            logits: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            mode = cfg.get("loss", "bce")
            if mode in {"focal", "focal_bce"}:
                gamma = float(cfg["focal_gamma"])
                alpha = float(cfg["focal_alpha"])
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                probs = torch.sigmoid(logits)
                weight = (alpha * (1.0 - probs).pow(gamma)).detach()
                return (weight * bce).mean()
            pos_weight_override = self._onoff_pos_weight(
                head, cfg, target, device, state, update_state=update_state, context="clip"
            )
            return _dynamic_pos_weighted_bce(logits, target, pos_weight_override=pos_weight_override)

        loss_pitch = _dynamic_pos_weighted_bce(out["pitch_logits"], pitch_target, pos_weight_override=pitch_pos_weight) * float(
            self.head_weights.get("pitch", 1.0)
        )
        total = total + loss_pitch
        parts["pitch"] = float(loss_pitch.detach().cpu())

        onset_target = targets.get("onset")
        if onset_target is None:
            onset_target = torch.zeros_like(out["onset_logits"])
        elif onset_target.dim() == 3:
            onset_target = onset_target.max(dim=1).values
        offset_target = targets.get("offset")
        if offset_target is None:
            offset_target = torch.zeros_like(out["offset_logits"])
        elif offset_target.dim() == 3:
            offset_target = offset_target.max(dim=1).values

        loss_onset = _clip_head_loss("onset", self.onset_cfg, out["onset_logits"], onset_target) * float(
            self.head_weights.get("onset", 1.0)
        )
        loss_offset = _clip_head_loss("offset", self.offset_cfg, out["offset_logits"], offset_target) * float(
            self.head_weights.get("offset", 1.0)
        )
        total = total + loss_onset + loss_offset
        parts["onset"] = float(loss_onset.detach().cpu())
        parts["offset"] = float(loss_offset.detach().cpu())

        hand_target = targets.get("hand")
        if hand_target is None:
            hand_target = torch.zeros_like(out["hand_logits"][..., 0], dtype=torch.long)
        loss_hand = nn.CrossEntropyLoss()(out["hand_logits"], hand_target) * float(self.head_weights.get("hand", 1.0))
        clef_target = targets.get("clef")
        if clef_target is None:
            clef_target = torch.zeros_like(out["clef_logits"][..., 0], dtype=torch.long)
        loss_clef = nn.CrossEntropyLoss()(out["clef_logits"], clef_target) * float(self.head_weights.get("clef", 1.0))
        total = total + loss_hand + loss_clef
        parts["hand"] = float(loss_hand.detach().cpu())
        parts["clef"] = float(loss_clef.detach().cpu())
        parts["total"] = float(total.detach().cpu())
        return total, parts

    # ---- frame-mode loss ------------------------------------------------
    def _frame_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        update_state: bool,
        per_tile: Optional[Mapping[str, Any]],
        hand_gating: Optional[Mapping[str, Any]],
        pos_rate_state: Optional[OnOffPosWeightEMA],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Frame-mode loss with time alignment, gating, priors, and per-tile paths."""
        state = pos_rate_state or self.pos_rate_state
        pitch_logit = preds["pitch_logits"]
        onset_logit = preds.get("onset_logits")
        offset_logit = preds.get("offset_logits")
        hand_logit = preds.get("hand_logits")
        clef_logit = preds.get("clef_logits")

        device = pitch_logit.device
        B, T_logits, P = pitch_logit.shape

        pitch_roll = targets.get("pitch")
        onset_roll = targets.get("onset")
        offset_roll = targets.get("offset")
        hand_frame = targets.get("hand")
        clef_frame = targets.get("clef")
        hand_frame_mask = targets.get("hand_frame_mask") or targets.get("hand_mask")
        hand_reach = targets.get("hand_reach")
        hand_reach_valid = targets.get("hand_reach_valid")

        def _zero_roll() -> torch.Tensor:
            return torch.zeros((B, T_logits, P), device=device, dtype=pitch_logit.dtype)

        pitch_roll = pitch_roll if torch.is_tensor(pitch_roll) else _zero_roll()
        onset_roll = onset_roll if torch.is_tensor(onset_roll) else _zero_roll()
        offset_roll = offset_roll if torch.is_tensor(offset_roll) else _zero_roll()
        pitch_roll = pitch_roll.to(device=device, dtype=pitch_logit.dtype)
        onset_roll = onset_roll.to(device=device, dtype=onset_logit.dtype if onset_logit is not None else pitch_logit.dtype)
        offset_roll = offset_roll.to(device=device, dtype=offset_logit.dtype if offset_logit is not None else pitch_logit.dtype)

        if pitch_roll.dim() == 2:
            pitch_roll = pitch_roll.unsqueeze(1)
        if onset_roll.dim() == 2:
            onset_roll = onset_roll.unsqueeze(1)
        if offset_roll.dim() == 2:
            offset_roll = offset_roll.unsqueeze(1)

        T_targets = pitch_roll.shape[1]
        if T_targets != T_logits:
            pitch_roll = pool_roll_BT(pitch_roll, T_logits)
            onset_roll = pool_roll_BT(onset_roll, T_logits)
            offset_roll = pool_roll_BT(offset_roll, T_logits)
            if torch.is_tensor(hand_frame):
                hand_frame = _interp_labels_BT(hand_frame, T_logits)
            if torch.is_tensor(clef_frame):
                clef_frame = _interp_labels_BT(clef_frame, T_logits)
            if torch.is_tensor(hand_frame_mask):
                hand_frame_mask = _interp_mask_BT(hand_frame_mask, T_logits)
            if torch.is_tensor(hand_reach):
                hr = hand_reach.float().unsqueeze(1)
                hand_reach = F.interpolate(hr, size=T_logits, mode="nearest").squeeze(1)
            if torch.is_tensor(hand_reach_valid):
                hand_reach_valid = _interp_mask_BT(hand_reach_valid, T_logits)

        pitch_roll = _match_pitch_dim(pitch_roll, P)
        onset_roll = _match_pitch_dim(onset_roll, P)
        offset_roll = _match_pitch_dim(offset_roll, P)
        if pitch_roll is None:
            pitch_roll = _zero_roll()
        if onset_roll is None:
            onset_roll = _zero_roll()
        if offset_roll is None:
            offset_roll = _zero_roll()
        if torch.is_tensor(hand_reach):
            hand_reach = _match_pitch_dim(hand_reach, P)
        if torch.is_tensor(hand_reach_valid) and hand_reach_valid.dim() == 3:
            hand_reach_valid = _match_pitch_dim(hand_reach_valid, P)

        neg_smooth = max(0.0, float(self.neg_smooth))
        if neg_smooth > 0.0:
            onset_roll = onset_roll * (1.0 - neg_smooth) + neg_smooth * (1.0 - onset_roll)
            offset_roll = offset_roll * (1.0 - neg_smooth) + neg_smooth * (1.0 - offset_roll)

        per_tile_ctx: Dict[str, Any] = {}
        if isinstance(self.per_tile_defaults, Mapping):
            per_tile_ctx.update(self.per_tile_defaults)
        if isinstance(per_tile, Mapping):
            per_tile_ctx.update(per_tile)
        per_tile_enabled = bool(per_tile_ctx.get("enabled", False))
        per_tile_heads_raw = per_tile_ctx.get("heads", ())
        if isinstance(per_tile_heads_raw, Sequence) and not isinstance(per_tile_heads_raw, (str, bytes)):
            per_tile_heads = {str(h).lower() for h in per_tile_heads_raw}
        else:
            per_tile_heads = {"pitch", "onset", "offset"}
        tile_mask_tensor = per_tile_ctx.get("mask") if per_tile_enabled else None
        if per_tile_enabled:
            if not torch.is_tensor(tile_mask_tensor):
                per_tile_enabled = False
            else:
                tile_mask_tensor = tile_mask_tensor.to(device=device)
                if tile_mask_tensor.dim() != 3 or tile_mask_tensor.shape[0] != B or tile_mask_tensor.shape[2] != P:
                    per_tile_enabled = False
        per_tile_debug: Dict[str, Dict[str, Any]] = {}
        pitch_tile = per_tile_ctx.get("pitch") if per_tile_enabled else None
        onset_tile = per_tile_ctx.get("onset") if per_tile_enabled else None
        offset_tile = per_tile_ctx.get("offset") if per_tile_enabled else None

        gating_cfg = dict(self.hand_gating_cfg) if isinstance(self.hand_gating_cfg, Mapping) else {}
        if isinstance(hand_gating, Mapping):
            gating_cfg.update(hand_gating)
        reach_weights = build_reach_weights(
            hand_reach if torch.is_tensor(hand_reach) else None,
            hand_reach_valid if torch.is_tensor(hand_reach_valid) else None,
            gating_cfg,
            device=device,
            dtype=pitch_roll.dtype,
        )

        pos_w_pitch = self._pitch_pos_weight(pitch_roll, device, context="frame")

        def _bce_with_reach(logits: torch.Tensor, targets: torch.Tensor, pos_w: Optional[torch.Tensor]) -> torch.Tensor:
            """BCE helper that applies optional reach-based weights."""
            if reach_weights is None or logits.shape != targets.shape:
                pos_w_local = pos_w.to(device=logits.device, dtype=logits.dtype) if pos_w is not None else None
                return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w_local)
            weight = reach_weights.to(device=logits.device, dtype=logits.dtype)
            pos_w_local = pos_w.to(device=logits.device, dtype=logits.dtype) if pos_w is not None else None
            return F.binary_cross_entropy_with_logits(
                logits,
                targets,
                weight=weight,
                pos_weight=pos_w_local,
                reduction="mean",
            )

        use_tile_pitch = per_tile_enabled and "pitch" in per_tile_heads and torch.is_tensor(pitch_tile)
        if use_tile_pitch:
            pitch_tile_tensor = torch.as_tensor(pitch_tile, device=device, dtype=pitch_logit.dtype)
            mask_tensor = torch.as_tensor(tile_mask_tensor, device=device, dtype=pitch_logit.dtype)
            loss_pitch_tile, debug_pitch = _per_tile_masked_loss(
                pitch_tile_tensor,
                pitch_roll,
                mask_tensor,
                {"loss": "bce_pos"},
                pos_w_pitch,
                head="pitch",
            )
            if loss_pitch_tile is not None:
                loss_pitch = loss_pitch_tile * float(self.head_weights.get("pitch", 1.0))
                if debug_pitch:
                    per_tile_debug["pitch"] = debug_pitch
            else:
                loss_pitch = _bce_with_reach(pitch_logit, pitch_roll, pos_w_pitch) * float(self.head_weights.get("pitch", 1.0))
        else:
            loss_pitch = _bce_with_reach(pitch_logit, pitch_roll, pos_w_pitch) * float(self.head_weights.get("pitch", 1.0))

        def _compute_frame_head_loss(
            head: str,
            cfg: Mapping[str, Any],
            logits: torch.Tensor,
            roll: torch.Tensor,
            pos_weight: Optional[torch.Tensor],
        ) -> torch.Tensor:
            """Compute per-head frame loss with optional focal/pos_weight."""
            mode = cfg.get("loss", "bce")
            if mode in {"focal", "focal_bce"}:
                gamma = float(cfg["focal_gamma"])
                alpha = float(cfg["focal_alpha"])
                bce = F.binary_cross_entropy_with_logits(logits, roll, reduction="none")
                probs = torch.sigmoid(logits)
                weight = (alpha * (1.0 - probs).pow(gamma)).detach()
                if reach_weights is not None and reach_weights.shape == bce.shape:
                    weight = weight * reach_weights.to(device=logits.device, dtype=logits.dtype)
                return (weight * bce).mean()

            pos_weight_local = pos_weight.to(device=logits.device, dtype=logits.dtype) if pos_weight is not None else None
            if reach_weights is not None and reach_weights.shape == roll.shape:
                weight = reach_weights.to(device=logits.device, dtype=logits.dtype)
                return F.binary_cross_entropy_with_logits(
                    logits,
                    roll,
                    weight=weight,
                    pos_weight=pos_weight_local,
                    reduction="mean",
                )
            return F.binary_cross_entropy_with_logits(logits, roll, pos_weight=pos_weight_local)

        pos_w_on = self._onoff_pos_weight(
            "onset", self.onset_cfg, onset_roll, device, state, update_state=update_state, context="frame"
        )
        pos_w_off = self._onoff_pos_weight(
            "offset", self.offset_cfg, offset_roll, device, state, update_state=update_state, context="frame"
        )

        use_tile_onset = per_tile_enabled and "onset" in per_tile_heads and torch.is_tensor(onset_tile)
        if use_tile_onset and onset_logit is not None:
            onset_tile_tensor = torch.as_tensor(onset_tile, device=device, dtype=onset_logit.dtype)
            mask_tensor = torch.as_tensor(tile_mask_tensor, device=device, dtype=onset_logit.dtype)
            loss_onset_tile, debug_onset = _per_tile_masked_loss(
                onset_tile_tensor,
                onset_roll,
                mask_tensor,
                self.onset_cfg,
                pos_w_on,
                head="onset",
            )
            if loss_onset_tile is not None:
                loss_onset = loss_onset_tile
                if debug_onset:
                    per_tile_debug["onset"] = debug_onset
            else:
                loss_onset = _compute_frame_head_loss("onset", self.onset_cfg, onset_logit, onset_roll, pos_w_on)
        else:
            loss_onset = _compute_frame_head_loss("onset", self.onset_cfg, onset_logit, onset_roll, pos_w_on) if onset_logit is not None else torch.tensor(0.0, device=device)

        use_tile_offset = per_tile_enabled and "offset" in per_tile_heads and torch.is_tensor(offset_tile)
        if use_tile_offset and offset_logit is not None:
            offset_tile_tensor = torch.as_tensor(offset_tile, device=device, dtype=offset_logit.dtype)
            mask_tensor = torch.as_tensor(tile_mask_tensor, device=device, dtype=offset_logit.dtype)
            loss_offset_tile, debug_offset = _per_tile_masked_loss(
                offset_tile_tensor,
                offset_roll,
                mask_tensor,
                self.offset_cfg,
                pos_w_off,
                head="offset",
            )
            if loss_offset_tile is not None:
                loss_offset = loss_offset_tile
                if debug_offset:
                    per_tile_debug["offset"] = debug_offset
            else:
                loss_offset = _compute_frame_head_loss("offset", self.offset_cfg, offset_logit, offset_roll, pos_w_off)
        else:
            loss_offset = _compute_frame_head_loss("offset", self.offset_cfg, offset_logit, offset_roll, pos_w_off) if offset_logit is not None else torch.tensor(0.0, device=device)

        loss_onset = loss_onset * float(self.head_weights.get("onset", 1.0))
        loss_offset = loss_offset * float(self.head_weights.get("offset", 1.0))

        ce_hand = nn.CrossEntropyLoss(reduction="none")
        ce_clef = nn.CrossEntropyLoss()
        if hand_logit is not None:
            hand_target = hand_frame.to(device=device, dtype=torch.long) if torch.is_tensor(hand_frame) else torch.zeros((B, T_logits), device=device, dtype=torch.long)
            hand_logits_flat = hand_logit.reshape(B * T_logits, -1)
            hand_target_flat = hand_target.reshape(B * T_logits)
            if torch.is_tensor(hand_frame_mask):
                mask = hand_frame_mask.reshape(B * T_logits).to(device=hand_logits_flat.device, dtype=hand_logits_flat.dtype)
                hand_target_flat = hand_target_flat.clamp(min=0, max=hand_logits_flat.shape[-1] - 1)
                hand_loss_elem = ce_hand(hand_logits_flat, hand_target_flat)
                supervised = mask.sum().clamp_min(1e-6)
                loss_hand = (hand_loss_elem * mask).sum() / supervised
            else:
                loss_hand = ce_hand(hand_logits_flat, hand_target_flat).mean()
        else:
            loss_hand = torch.tensor(0.0, device=device)
        loss_hand = loss_hand * float(self.head_weights.get("hand", 1.0))

        if clef_logit is not None:
            clef_target = clef_frame.to(device=device, dtype=torch.long) if torch.is_tensor(clef_frame) else torch.zeros((B, T_logits), device=device, dtype=torch.long)
            clef_logits_flat = clef_logit.reshape(B * T_logits, -1)
            clef_target_flat = clef_target.reshape(B * T_logits)
            loss_clef = ce_clef(clef_logits_flat, clef_target_flat) * float(self.head_weights.get("clef", 1.0))
        else:
            loss_clef = torch.tensor(0.0, device=device)

        total = loss_pitch + loss_onset + loss_offset + loss_hand + loss_clef
        parts: Dict[str, Any] = {
            "pitch": float(loss_pitch.detach().cpu()),
            "onset": float(loss_onset.detach().cpu()),
            "offset": float(loss_offset.detach().cpu()),
            "hand": float(loss_hand.detach().cpu()),
            "clef": float(loss_clef.detach().cpu()),
        }

        reg_terms: Dict[str, torch.Tensor] = {}
        if self.priors_enabled:
            for head, cfg, logits in (
                ("onset", self.onset_cfg, onset_logit),
                ("offset", self.offset_cfg, offset_logit),
            ):
                if logits is None:
                    continue
                reg = compute_prior_mean_regularizer(logits, cfg, default_mean=0.12)
                if reg is None:
                    continue
                total = total + reg
                reg_terms[head] = reg

        if reg_terms:
            reg_sum = sum(float(reg.detach().cpu()) for reg in reg_terms.values())
            parts["reg_onoff"] = reg_sum
            for head, reg in reg_terms.items():
                parts[f"reg_{head}"] = float(reg.detach().cpu())

        if per_tile_debug:
            parts["per_tile_debug"] = per_tile_debug
        parts["total"] = float(total.detach().cpu())
        return total, parts


def build_loss(cfg: Mapping[str, Any] | None = None, head_weights: Mapping[str, float] | None = None) -> MultitaskLoss:
    """Factory for the multitask loss (expects cfg.training.loss schema)."""
    return MultitaskLoss(cfg, head_weights=head_weights)


__all__ = ["MultitaskLoss", "OnOffPosWeightEMA", "build_loss"]
