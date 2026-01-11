"""Per-head loss composition helper."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tivit.decoder.decode import pool_roll_BT

LOGGER = logging.getLogger(__name__)
_ADAPTIVE_BAND_LOGGED: set[Tuple[str, str]] = set()


class PosRateEMA:
    """Track an exponential moving average for positive class rates."""

    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.value: Optional[torch.Tensor] = None

    def process(self, observation: torch.Tensor, *, update: bool = True) -> torch.Tensor:
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
        return {"value": None if self.value is None else self.value.tolist()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        value = state.get("value") if isinstance(state, Mapping) else None
        if value is None:
            self.value = None
        else:
            self.value = torch.tensor(value, dtype=torch.float32)


class OnOffPosWeightEMA:
    """Maintain EMA statistics for onset/offset heads (clip & frame modes)."""

    def __init__(self, alpha: float) -> None:
        alpha = float(alpha)
        self.alpha = alpha
        self.clip = {"onset": PosRateEMA(alpha), "offset": PosRateEMA(alpha)}
        self.frame = {"onset": PosRateEMA(alpha), "offset": PosRateEMA(alpha)}

    @staticmethod
    def _reduce_clip_targets(targets: torch.Tensor) -> torch.Tensor:
        return torch.tensor([float(targets.float().mean().item())], dtype=torch.float32)

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


def _coerce_pos_weight_band(value: Any) -> Optional[Tuple[float, float]]:
    if isinstance(value, Mapping):
        value = value.get("band", value.get("values", value.get("range", value)))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        low = float(value[0])
        high = float(value[1])
    except (TypeError, ValueError):
        return None
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
    rate_clamped = rate.clone().clamp(min=eps, max=1.0 - eps)
    pos_weight = (1.0 - rate_clamped) / rate_clamped
    return pos_weight.clamp_(min=clamp_min, max=clamp_max)


def _adaptive_pos_weight_from_roll(target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = target.detach().reshape(-1, target.shape[-1]).float()
    pos = flat.mean(dim=0).clamp(min=eps, max=1.0 - eps)
    return ((1.0 - pos) / (pos + eps)).clamp(1.0, 100.0)


def _banded_pos_weight_from_roll(
    target: torch.Tensor,
    band: Tuple[float, float],
    eps: float = 1e-6,
) -> torch.Tensor:
    flat = target.detach().reshape(-1, target.shape[-1]).float()
    pos = flat.mean(dim=0).clamp(min=eps, max=1.0 - eps)
    neg = (1.0 - pos).clamp_min(eps)
    ratio = pos / neg
    low, high = band
    if high < low:
        low, high = high, low
    span = ratio.max() - ratio.min()
    if span <= eps:
        mapped = torch.full_like(ratio, 0.5 * (low + high))
    else:
        mapped = (ratio - ratio.min()) / span.clamp_min(eps)
        mapped = low + mapped * (high - low)
    mapped = mapped.clamp(min=min(low, high), max=max(low, high))
    return mapped


def _log_banded_weight_stats(head: str, context: str, weights: torch.Tensor, band: Tuple[float, float]) -> None:
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
    pos_rate_override: Optional[torch.Tensor] = None,
    pos_weight_override: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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

    pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    w = weight.to(device=logits.device, dtype=logits.dtype) if weight is not None else None

    return F.binary_cross_entropy_with_logits(
        logits,
        target_float,
        weight=w,
        pos_weight=pos_weight,
        reduction="mean",
    )


def _time_pool_out_to_clip(out: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    """If logits are time-distributed but using clip losses, reduce over time."""

    pooled = dict(out)
    if "pitch_logits" in out and out["pitch_logits"].dim() == 3:
        pooled["pitch_logits"] = out["pitch_logits"].mean(dim=1)
    if "onset_logits" in out and out["onset_logits"].dim() == 3:
        pooled["onset_logits"] = out["onset_logits"].mean(dim=1)
    if "offset_logits" in out and out["offset_logits"].dim() == 3:
        pooled["offset_logits"] = out["offset_logits"].mean(dim=1)
    if "hand_logits" in out and out["hand_logits"].dim() == 3:
        pooled["hand_logits"] = out["hand_logits"].mean(dim=1)
    if "clef_logits" in out and out["clef_logits"].dim() == 3:
        pooled["clef_logits"] = out["clef_logits"].mean(dim=1)
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
        "pos_weight_band": None,
    }

    per_head_overrides = weights.get("onoff_heads", {})
    if not isinstance(per_head_overrides, Mapping):
        per_head_overrides = {}

    band_defaults = weights.get("onoff_pos_weight_bands", {})
    if not isinstance(band_defaults, Mapping):
        band_defaults = {}

    alias_patterns = {
        "loss": ("{head}_loss", "{head}_onoff_loss"),
        "pos_weight_mode": ("{head}_pos_weight_mode", "{head}_onoff_pos_weight_mode"),
        "pos_weight": ("{head}_pos_weight", "{head}_onoff_pos_weight"),
        "focal_gamma": ("{head}_focal_gamma",),
        "focal_alpha": ("{head}_focal_alpha",),
        "prior_mean": ("{head}_prior_mean", "{head}_onoff_prior_mean"),
        "prior_weight": ("{head}_prior_weight", "{head}_onoff_prior_weight"),
        "pos_weight_band": ("{head}_pos_weight_band",),
    }

    resolved: Dict[str, Dict[str, Any]] = {}
    for head in ("onset", "offset"):
        cfg = dict(default_cfg)
        cfg["pos_weight_band"] = _coerce_pos_weight_band(band_defaults.get(head))

        nested = per_head_overrides.get(head)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                if key == "pos_weight_band":
                    cfg[key] = _coerce_pos_weight_band(value)
                    continue
                if key in cfg:
                    cfg[key] = value

        for key, patterns in alias_patterns.items():
            for pattern in patterns:
                alias = pattern.format(head=head)
                if alias in weights:
                    if key == "pos_weight_band":
                        cfg[key] = _coerce_pos_weight_band(weights[alias])
                    else:
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


def _resolve_head_weights(cfg: Mapping[str, Any] | None, override: Mapping[str, float] | None) -> Dict[str, float]:
    weights = {k: 1.0 for k in ("pitch", "onset", "offset", "hand", "clef")}
    if isinstance(cfg, Mapping):
        for head in weights:
            if head in cfg:
                try:
                    weights[head] = float(cfg[head])
                except (TypeError, ValueError):
                    pass
    if isinstance(override, Mapping):
        for k, v in override.items():
            try:
                weights[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return weights


def _per_tile_masked_loss(
    logits_tile: torch.Tensor,
    roll: torch.Tensor,
    mask: torch.Tensor,
    cfg: Mapping[str, Any],
    pos_weight: Optional[torch.Tensor],
    *,
    head: str,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Compute a masked per-tile loss and lightweight diagnostics.
    Returns (loss_tensor, debug_dict) where debug_dict contains per-tile means.
    """
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
        loss_cfg = {}
        per_tile_cfg = {}
        gating_cfg = {}
        if isinstance(cfg, Mapping):
            loss_cfg = cfg.get("training", {}).get("loss_weights", {}) if isinstance(cfg.get("training"), Mapping) else {}
            per_tile_cfg = cfg.get("training", {}).get("per_tile", {}) if isinstance(cfg.get("training"), Mapping) else {}
            gating_cfg = cfg.get("priors", {}).get("hand_gating", {}) if isinstance(cfg.get("priors"), Mapping) else {}
        self.loss_cfg = loss_cfg
        self.head_cfgs = _resolve_onoff_loss_config(loss_cfg)
        self.head_weights = _resolve_head_weights(loss_cfg, head_weights)
        self.neg_smooth = float(loss_cfg.get("onoff_neg_smooth", 0.0))
        self.pitch_band = _coerce_pos_weight_band((loss_cfg.get("onoff_pos_weight_bands", {}) or {}).get("pitch"))
        self.pitch_pos_mode = str(loss_cfg.get("onoff_pos_weight_mode", "adaptive")).lower()
        ema_alpha = float(loss_cfg.get("onoff_pos_weight_ema_alpha", 0.0))
        if any(cfg.get("pos_weight_mode", "") == "ema" for cfg in self.head_cfgs.values()) and ema_alpha <= 0.0:
            ema_alpha = 0.05
        self.pos_rate_state = OnOffPosWeightEMA(ema_alpha) if ema_alpha > 0.0 else None
        self.per_tile_defaults = per_tile_cfg
        self.hand_gating_cfg = gating_cfg

    def state_dict(self) -> Dict[str, Any]:
        return {"pos_rate_state": None if self.pos_rate_state is None else self.pos_rate_state.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
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

    # ---- clip-mode loss -------------------------------------------------
    def _clip_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        update_state: bool,
        pos_rate_state: Optional[OnOffPosWeightEMA],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        state = pos_rate_state or self.pos_rate_state
        out = _time_pool_out_to_clip(preds)
        device = next(iter(out.values())).device if out else "cpu"
        total = torch.tensor(0.0, device=device)
        parts: Dict[str, Any] = {}

        pitch_target = targets.get("pitch")
        if pitch_target is None:
            pitch_target = torch.zeros_like(out["pitch_logits"])
        elif pitch_target.dim() == 3:
            pitch_target = pitch_target.max(dim=1).values

        head_cfgs = self.head_cfgs
        onset_cfg = head_cfgs["onset"]
        offset_cfg = head_cfgs["offset"]

        def _clip_pos_weight(
            head: str,
            cfg: Mapping[str, Any],
            target: torch.Tensor,
        ) -> Optional[torch.Tensor]:
            mode = str(cfg.get("pos_weight_mode", "adaptive")).lower()
            loss_mode = str(cfg.get("loss", "bce_pos")).lower()
            if loss_mode not in {"bce_pos", "bce", "bce_with_logits"}:
                return None
            if mode == "fixed" and cfg.get("pos_weight") is not None:
                return torch.full((target.shape[-1],), float(cfg["pos_weight"]), dtype=torch.float32)
            if mode == "ema" and state is not None:
                return state.clip_pos_weight(head, target, update=update_state)
            if mode == "adaptive_band":
                band = cfg.get("pos_weight_band")
                if band is None:
                    return _adaptive_pos_weight_from_roll(target)
                weights_local = _banded_pos_weight_from_roll(target, band)
                _log_banded_weight_stats(head, "clip", weights_local, band)
                return weights_local
            if mode in {"none", "off"}:
                return None
            return None

        def _clip_head_loss(
            head: str,
            cfg: Mapping[str, Any],
            logits: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            mode = str(cfg.get("loss", "bce_pos")).lower()
            if mode in {"focal", "focal_bce"}:
                gamma = float(cfg.get("focal_gamma", 2.0))
                alpha = float(cfg.get("focal_alpha", 0.25))
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
                probs = torch.sigmoid(logits)
                weight = (alpha * (1.0 - probs).pow(gamma)).detach()
                return (weight * bce).mean()
            pos_weight_override = _clip_pos_weight(head, cfg, target)
            return _dynamic_pos_weighted_bce(logits, target, pos_weight_override=pos_weight_override)

        loss_pitch = _dynamic_pos_weighted_bce(out["pitch_logits"], pitch_target) * float(self.head_weights.get("pitch", 1.0))
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

        loss_onset = _clip_head_loss("onset", onset_cfg, out["onset_logits"], onset_target) * float(self.head_weights.get("onset", 1.0))
        loss_offset = _clip_head_loss("offset", offset_cfg, out["offset_logits"], offset_target) * float(self.head_weights.get("offset", 1.0))
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

        pitch_roll = _match_pitch_dim(pitch_roll, P) or _zero_roll()
        onset_roll = _match_pitch_dim(onset_roll, P) or _zero_roll()
        offset_roll = _match_pitch_dim(offset_roll, P) or _zero_roll()
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
        per_tile_heads_raw = per_tile_ctx.get("heads")
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

        gating_cfg = dict(self.hand_gating_cfg)
        if isinstance(hand_gating, Mapping):
            gating_cfg.update(hand_gating)
        gating_mode = str(gating_cfg.get("mode", "off")).lower()
        gating_alpha = float(gating_cfg.get("strength", 1.0))
        reach_weights = None
        if gating_mode == "loss_reweight":
            if torch.is_tensor(hand_reach) and torch.is_tensor(hand_reach_valid):
                hr = hand_reach.to(device=device, dtype=pitch_roll.dtype).clamp(0.0, 1.0)
                hr_valid = hand_reach_valid.to(device=device, dtype=pitch_roll.dtype)
                if hr_valid.dim() == 2:
                    hr_valid = hr_valid.unsqueeze(-1).expand_as(hr)
                if hr_valid.shape == hr.shape:
                    neg_weight = 1.0 + gating_alpha * (1.0 - hr)
                    reach_weights = torch.where(hr_valid > 0.5, neg_weight, torch.ones_like(hr))

        eps = 1e-6
        pitch_mode = self.pitch_pos_mode
        pos_rate_pitch = pitch_roll.reshape(-1, pitch_roll.shape[-1]).mean(dim=0).clamp_min(eps)
        if pitch_mode == "adaptive_band" and self.pitch_band is not None:
            pos_w_pitch = _banded_pos_weight_from_roll(pitch_roll, self.pitch_band, eps)
            _log_banded_weight_stats("pitch", "frame", pos_w_pitch, self.pitch_band)
            pos_w_pitch = pos_w_pitch.to(device)
        else:
            pos_w_pitch = ((1.0 - pos_rate_pitch) / (pos_rate_pitch + eps)).sqrt().clamp(1.0, 50.0).to(device)

        def _bce_with_reach(logits: torch.Tensor, targets: torch.Tensor, pos_w: Optional[torch.Tensor]) -> torch.Tensor:
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

        head_cfgs = self.head_cfgs
        onset_cfg = head_cfgs["onset"]
        offset_cfg = head_cfgs["offset"]

        def _frame_pos_weight(head: str, cfg: Mapping[str, Any], roll: torch.Tensor) -> Optional[torch.Tensor]:
            mode = str(cfg.get("pos_weight_mode", "adaptive")).lower()
            loss_mode = str(cfg.get("loss", "bce_pos")).lower()
            if loss_mode not in {"bce_pos", "bce", "bce_with_logits"}:
                return None
            if mode == "fixed" and cfg.get("pos_weight") is not None:
                return torch.full((P,), float(cfg["pos_weight"]), dtype=torch.float32)
            if mode == "ema" and state is not None:
                return state.frame_pos_weight(head, roll, update=update_state)
            if mode == "adaptive_band":
                band = cfg.get("pos_weight_band")
                if band is None:
                    return _adaptive_pos_weight_from_roll(roll, eps)
                weights_local = _banded_pos_weight_from_roll(roll, band, eps).detach()
                _log_banded_weight_stats(head, "frame", weights_local, band)
                return weights_local
            if mode in {"none", "off"}:
                return None
            return _adaptive_pos_weight_from_roll(roll, eps)

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
                if reach_weights is not None and reach_weights.shape == bce.shape:
                    weight = weight * reach_weights.to(device=logits.device, dtype=logits.dtype)
                return (weight * bce).mean()

            if pos_weight is not None:
                pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
            if reach_weights is not None and reach_weights.shape == roll.shape:
                weight = reach_weights.to(device=logits.device, dtype=logits.dtype)
                return F.binary_cross_entropy_with_logits(
                    logits,
                    roll,
                    weight=weight,
                    pos_weight=pos_weight,
                    reduction="mean",
                )
            return F.binary_cross_entropy_with_logits(logits, roll, pos_weight=pos_weight)

        pos_w_on = _frame_pos_weight("onset", onset_cfg, onset_roll)
        pos_w_off = _frame_pos_weight("offset", offset_cfg, offset_roll)

        use_tile_onset = per_tile_enabled and "onset" in per_tile_heads and torch.is_tensor(onset_tile)
        if use_tile_onset and onset_logit is not None:
            onset_tile_tensor = torch.as_tensor(onset_tile, device=device, dtype=onset_logit.dtype)
            mask_tensor = torch.as_tensor(tile_mask_tensor, device=device, dtype=onset_logit.dtype)
            loss_onset_tile, debug_onset = _per_tile_masked_loss(
                onset_tile_tensor,
                onset_roll,
                mask_tensor,
                onset_cfg,
                pos_w_on,
                head="onset",
            )
            if loss_onset_tile is not None:
                loss_onset = loss_onset_tile
                if debug_onset:
                    per_tile_debug["onset"] = debug_onset
            else:
                loss_onset = _compute_frame_head_loss("onset", onset_cfg, onset_logit, onset_roll, pos_w_on)
        else:
            loss_onset = _compute_frame_head_loss("onset", onset_cfg, onset_logit, onset_roll, pos_w_on) if onset_logit is not None else torch.tensor(0.0, device=device)

        use_tile_offset = per_tile_enabled and "offset" in per_tile_heads and torch.is_tensor(offset_tile)
        if use_tile_offset and offset_logit is not None:
            offset_tile_tensor = torch.as_tensor(offset_tile, device=device, dtype=offset_logit.dtype)
            mask_tensor = torch.as_tensor(tile_mask_tensor, device=device, dtype=offset_logit.dtype)
            loss_offset_tile, debug_offset = _per_tile_masked_loss(
                offset_tile_tensor,
                offset_roll,
                mask_tensor,
                offset_cfg,
                pos_w_off,
                head="offset",
            )
            if loss_offset_tile is not None:
                loss_offset = loss_offset_tile
                if debug_offset:
                    per_tile_debug["offset"] = debug_offset
            else:
                loss_offset = _compute_frame_head_loss("offset", offset_cfg, offset_logit, offset_roll, pos_w_off)
        else:
            loss_offset = _compute_frame_head_loss("offset", offset_cfg, offset_logit, offset_roll, pos_w_off) if offset_logit is not None else torch.tensor(0.0, device=device)

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
        for head, cfg, logits in (
            ("onset", onset_cfg, onset_logit),
            ("offset", offset_cfg, offset_logit),
        ):
            if logits is None:
                continue
            prior_w = float(cfg.get("prior_weight", 0.0))
            if prior_w <= 0.0:
                continue
            prior_mean = float(cfg.get("prior_mean", 0.12))
            reg = prior_w * (torch.sigmoid(logits).mean() - prior_mean).abs()
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
    """
    Factory for the multitask loss.

    Backwards-compat: passing a plain mapping without a ``training`` section is
    treated as a head-weight override to match the old signature.
    """
    if cfg is not None and head_weights is None and not (isinstance(cfg, Mapping) and "training" in cfg):
        head_weights = cfg  # type: ignore[assignment]
        cfg = None
    return MultitaskLoss(cfg, head_weights=head_weights)


__all__ = ["MultitaskLoss", "OnOffPosWeightEMA", "build_loss"]
