# scripts/train.py
from utils import load_config
from data import make_dataloader
from models import build_model

import os
from pathlib import Path
from time import perf_counter

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ----------------------- helpers -----------------------
def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dirs(cfg: dict):
    ckpt = Path(cfg["logging"]["checkpoint_dir"]).expanduser()
    logd = Path(cfg["logging"]["log_dir"]).expanduser()
    ckpt.mkdir(parents=True, exist_ok=True)
    logd.mkdir(parents=True, exist_ok=True)
    return ckpt, logd

def make_criterions():
    # Rough pos_weight from your val rates: onset ~2% -> ~50; offset ~1% -> ~100
    pos_w_on  = torch.tensor([50.0])    #later make them configurable through config
    pos_w_off = torch.tensor([100.0])
    # heads:
    # - pitch_logits: (B, 128)  -> CrossEntropy
    # - onset_logits: (B,)      -> BCEWithLogits
    # - offset_logits: (B,)     -> BCEWithLogits
    # - hand_logits:  (B, 2)    -> CrossEntropy
    # - clef_logits:  (B, 3)    -> CrossEntropy
    return {
        "pitch": nn.CrossEntropyLoss(),
        "onset": nn.BCEWithLogitsLoss(),
        "offset": nn.BCEWithLogitsLoss(),
        "hand": nn.CrossEntropyLoss(),
        "clef": nn.CrossEntropyLoss(),
    }

# --- add this helper near the top of scripts/train.py ---
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
        print(f"[GRAD{(':'+step_tag) if step_tag else ''}] onset/offset param norms (top): " +
              ", ".join(f"{k}={v:.4e}" for k,v in top))
    else:
        print(f"[GRAD{(':'+step_tag) if step_tag else ''}] no onset/offset grads found")

def _pool_roll_BT128(x_bt128, Tprime):
    # (B,T,128) -> (B,T',128) with "any positive in window" preserved via max
    # used for pitch/onset/offset rolls when aligning T -> T'
    x = x_bt128.permute(0, 2, 1)           # (B,128,T)
    x = F.adaptive_max_pool1d(x, Tprime)   # (B,128,T')
    return x.permute(0, 2, 1).contiguous() # (B,T',128)

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

def _time_pool_out_to_clip(out: dict) -> dict:
    """
    If logits are time-distributed (B,T,...) but we're about to use clip losses,
    reduce over time with mean so shapes match clip targets.
    """
    pooled = dict(out)  # shallow copy
    if out["pitch_logits"].dim() == 3:
        pooled["pitch_logits"]  = out["pitch_logits"].mean(dim=1)      # (B,128)
    if out["onset_logits"].dim() == 2:
        pooled["onset_logits"]  = out["onset_logits"].mean(dim=1)      # (B,)
    if out["offset_logits"].dim() == 2:
        pooled["offset_logits"] = out["offset_logits"].mean(dim=1)     # (B,)
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

def _set_onoff_head_bias(model, prior=0.12): #CAL prior was 0.01 END CAL
    """Calibrate onset/offset last-layer biases to a small prior P≈1% (negative logits)."""
    import math, torch, torch.nn as nn
    b0 = math.log(prior / (1.0 - prior))
    with torch.no_grad():
        if isinstance(model.head_onset, nn.Sequential) and hasattr(model.head_onset[-1], "bias"):
            model.head_onset[-1].bias.fill_(b0)
        if isinstance(model.head_offset, nn.Sequential) and hasattr(model.head_offset[-1], "bias"):
            model.head_offset[-1].bias.fill_(b0)

def compute_loss(out: dict, tgt: dict, crit: dict, weights: dict):
    # Guard: if logits are time-distributed but we're in clip-loss, pool over time
    if out["pitch_logits"].dim() == 3:  # (B,T,128)
        out = _time_pool_out_to_clip(out)
        
    loss_pitch  = crit["pitch"](out["pitch_logits"],  tgt["pitch"])   * weights["pitch"]
    loss_onset  = crit["onset"](out["onset_logits"],  tgt["onset"])   * weights["onset"]
    loss_offset = crit["offset"](out["offset_logits"], tgt["offset"]) * weights["offset"]
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
    pitch_logit  = out["pitch_logits"]     # (B,T',128)
    onset_logit  = out["onset_logits"]     # (B,T')
    offset_logit = out["offset_logits"]    # (B,T')
    hand_logit   = out["hand_logits"]      # (B,T',C_hand=2)
    clef_logit   = out["clef_logits"]      # (B,T',C_clef=3)

    B, T_logits = onset_logit.shape

    # --- targets (at original T) ---
    pitch_roll   = batch["pitch_roll"].float()   # (B,T,128)
    onset_roll   = batch["onset_roll"].float()   # (B,T,128)
    offset_roll  = batch["offset_roll"].float()  # (B,T,128)
    hand_frame   = batch["hand_frame"].long()    # (B,T)
    clef_frame   = batch["clef_frame"].long()    # (B,T)
    T_targets = pitch_roll.shape[1]

    # --- time alignment: T -> T' (keep your repo behavior) ---
    if T_targets != T_logits:
        pitch_roll  = _pool_roll_BT128(pitch_roll,  T_logits)
        onset_roll  = _pool_roll_BT128(onset_roll,  T_logits)
        offset_roll = _pool_roll_BT128(offset_roll, T_logits)
        hand_frame  = _interp_labels_BT(hand_frame, T_logits)
        clef_frame  = _interp_labels_BT(clef_frame, T_logits)
    # (this matches the alignment already used elsewhere in your code). :contentReference[oaicite:2]{index=2}

    # --- derive ANY-note flags per frame (T') ---
    onset_any  = (onset_roll  > 0).any(dim=-1).float()   # (B,T')
    offset_any = (offset_roll > 0).any(dim=-1).float()   # (B,T')

    # --- optional negative-class smoothing for on/off targets ---
    neg_smooth = float(weights.get("onoff_neg_smooth", 0.0))
    if neg_smooth > 0.0:
        onset_any  = onset_any  * (1.0 - neg_smooth) + neg_smooth * (1.0 - onset_any)
        offset_any = offset_any * (1.0 - neg_smooth) + neg_smooth * (1.0 - offset_any)

    # --- pitch loss: gentle per-pitch pos_weight (sqrt + clamp) ---
    eps = 1e-6
    pos_rate_pitch = pitch_roll.reshape(-1, pitch_roll.shape[-1]).mean(dim=0).clamp_min(eps)  # (128,)
    pos_w_pitch = ((1.0 - pos_rate_pitch) / (pos_rate_pitch + eps)).sqrt().clamp(1.0, 50.0).to(device)
    bce_pitch = nn.BCEWithLogitsLoss(pos_weight=pos_w_pitch)
    loss_pitch = bce_pitch(pitch_logit, pitch_roll) * float(weights.get("pitch", 1.0))

    # --- onset/offset loss: "bce_pos" (adaptive/fixed) OR "focal" ---
    onoff_mode = str(weights.get("onoff_loss", "focal")).lower()  # "bce_pos" | "focal"
    if onoff_mode == "bce_pos":
        pw_mode = str(weights.get("onoff_pos_weight_mode", "adaptive")).lower()  # "adaptive" | "fixed"
        if pw_mode == "fixed":
            pw_val = float(weights.get("onoff_pos_weight", 0.0))
            if pw_val > 0.0:
                pos_w_on  = torch.tensor([pw_val], device=device)
                pos_w_off = torch.tensor([pw_val], device=device)
            else:
                pw_mode = "adaptive"
        if pw_mode == "adaptive":
            p_on  = onset_any.mean().clamp_min(eps)
            p_off = offset_any.mean().clamp_min(eps)
            pos_w_on  = ((1.0 - p_on)  / (p_on  + eps)).clamp(1.0, 100.0).detach()
            pos_w_off = ((1.0 - p_off) / (p_off + eps)).clamp(1.0, 100.0).detach()

        bce_on  = nn.BCEWithLogitsLoss(pos_weight=pos_w_on)
        bce_off = nn.BCEWithLogitsLoss(pos_weight=pos_w_off)
        loss_onset  = bce_on(onset_logit,  onset_any)
        loss_offset = bce_off(offset_logit, offset_any)

    else:
        # focal BCE on logits (your original path)
        gamma = float(weights.get("focal_gamma", 2.0))
        alpha = float(weights.get("focal_alpha", 0.15))
        bce_on  = F.binary_cross_entropy_with_logits(onset_logit,  onset_any,  reduction="none")
        bce_off = F.binary_cross_entropy_with_logits(offset_logit, offset_any, reduction="none")
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
    tgt = {
        "pitch": torch.randint(0, 128, (batch_size,), device=device),  # class ids
        "onset": torch.rand(batch_size, device=device),                # [0,1]
        "offset": torch.rand(batch_size, device=device),               # [0,1]
        "hand": torch.randint(0, 2, (batch_size,), device=device),     # class ids
        "clef": torch.randint(0, 3, (batch_size,), device=device),     # class ids
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
    
# ----------------------- train loop -----------------------

def train_one_epoch(model, train_loader, optimizer, cfg, writer=None, epoch=1):
    model.train()
    crit = make_criterions()  # used only in clip-mode
    w = cfg["training"]["loss_weights"]

    use_amp = bool(cfg["training"].get("amp", False))
    scaler = GradScaler(enabled=use_amp)

    sums = {"total": 0.0, "pitch": 0.0, "onset": 0.0, "offset": 0.0, "hand": 0.0, "clef": 0.0}
    n = 0

    for batch in train_loader:
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

        optimizer.zero_grad(set_to_none=True)

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
        scaler.scale(loss).backward()
        if cfg["training"].get("grad_clip", None):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        # >>> print onset/offset grad norms here <<<
        _print_head_grad_norms(model, step_tag=f"e{epoch}_it{it}")
        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss parts
        for k in sums:
            if k in parts:
                sums[k] += parts[k]
        n += 1

        # --- OPTIONAL: train-side predicted positive rates (only frame mode) ---
        if use_frame and "onset_logits" in out and "offset_logits" in out:
            thr = float(cfg.get("training", {}).get("metrics", {}).get("prob_threshold", 0.5))
            onset_pred  = (torch.sigmoid(out["onset_logits"])  >= thr).float()
            offset_pred = (torch.sigmoid(out["offset_logits"]) >= thr).float()
            if "train_onset_pred_rate" not in sums:
                sums["train_onset_pred_rate"] = 0.0
                sums["train_offset_pred_rate"] = 0.0
            sums["train_onset_pred_rate"]  += onset_pred.mean().item()
            sums["train_offset_pred_rate"] += offset_pred.mean().item()

    # Average over batches
    avg = {k: (sums[k] / max(1, n)) for k in sums}

    # Log to TensorBoard
    if writer is not None:
        for k, v in avg.items():
            writer.add_scalar(f"train/{k}", v, epoch)

    return avg


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: dict, best_val: float | None = None):
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "best_val": best_val,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    
def evaluate_one_epoch(model, loader, cfg):
    model.eval()
    crit = make_criterions()
    #w = cfg["training"]["loss_weights"]   #This was recently proposed but preferred to keep the previous
    w = get_loss_weights(cfg) if "get_loss_weights" in globals() else cfg["training"]["loss_weights"]
    thr = float(cfg.get("training", {}).get("metrics", {}).get("prob_threshold", 0.5))

    sums = {"total": 0.0, "pitch": 0.0, "onset": 0.0, "offset": 0.0, "hand": 0.0, "clef": 0.0}
    n_batches = 0

    # metric accumulators
    metric_counts = {
        "pitch_acc": 0.0,
        "hand_acc":  0.0,
        "clef_acc":  0.0,
        "onset_f1":  0.0,
        "offset_f1": 0.0,
        # additions for masked F1 + imbalance diagnostics
        "onset_pos_rate":  0.0,
        "offset_pos_rate": 0.0,
        "onset_pred_rate": 0.0,
        "offset_pred_rate": 0.0,
        "n_on":  0,   # batches contributing to onset_f1
        "n_off": 0,   # batches contributing to offset_f1
    }
    metric_n = 0

    want = ("pitch", "onset", "offset", "hand", "clef")
    with torch.no_grad():
        for batch in loader:
            x = batch["video"]
            have_all = all(k in batch for k in want)
            use_dummy = bool(cfg["training"].get("debug_dummy_labels", False))
            if have_all and not use_dummy:
                tgt = {k: batch[k] for k in want}
                tgt["pitch"]  = tgt["pitch"].long()
                tgt["hand"]   = tgt["hand"].long()
                tgt["clef"]   = tgt["clef"].long()
                tgt["onset"]  = tgt["onset"].float()
                tgt["offset"] = tgt["offset"].float()
            else:
                tgt = fabricate_dummy_targets(x.shape[0])

            out = model(x)
           # use_frame = (
           #     "pitch_roll" in batch and "onset_roll" in batch and "offset_roll" in batch
           #     and "hand_frame" in batch and "clef_frame" in batch
           #     and hasattr(model, "head_mode") and model.head_mode == "frame"
           # )
           
            use_frame = (
                getattr(model, "head_mode", "clip") == "frame" and
                all(k in batch for k in ("pitch_roll", "onset_roll", "offset_roll", "hand_frame", "clef_frame"))
            )
            
            if use_frame:
                loss, parts = compute_loss_frame(out, batch, weights=w)
            else:
                # guard: if logits are (B,T,...) but we’re using clip loss, pool over time
                if out["pitch_logits"].dim() == 3:
                    out = _time_pool_out_to_clip(out)
                loss, parts = compute_loss(out, tgt, crit, w)

            #loss, parts = compute_loss(out, tgt, crit, w)
            for k in sums: sums[k] += parts[k]
            n_batches += 1

            # --- metrics ---
            if use_frame:
                # --- align frame targets to logits time (T -> T_logits) ---
                B, T_logits = out["onset_logits"].shape

                onset_roll  = batch["onset_roll"].float()   # (B, T, 128)
                offset_roll = batch["offset_roll"].float()  # (B, T, 128)
                hand_frame  = batch["hand_frame"].long()    # (B, T)
                clef_frame  = batch["clef_frame"].long()    # (B, T)

                T_targets = onset_roll.shape[1]

                #if T_targets != T_logits:
                if onset_roll.shape[1] != T_logits:
                    def _pool_bool_BT128(x_bt128, Tprime):
                        # (B,T,128) -> (B,T',128), preserving "any positive in window"
                        x = x_bt128.permute(0, 2, 1)             # (B,128,T)
                        x = F.adaptive_max_pool1d(x, Tprime)     # (B,128,T')
                        return x.permute(0, 2, 1).contiguous()   # (B,T',128)

                    def _interp_labels_BT(x_bt, Tprime):
                        # nearest for class indices: (B,T) -> (B,T')
                        x = x_bt.float().unsqueeze(1)            # (B,1,T)
                        x = F.interpolate(x, size=Tprime, mode="nearest")
                        return x.squeeze(1).long()               # (B,T')

                    onset_roll  = _pool_bool_BT128(onset_roll,  T_logits)
                    offset_roll = _pool_bool_BT128(offset_roll, T_logits)
                    hand_frame  = _interp_labels_BT(hand_frame, T_logits)
                    clef_frame  = _interp_labels_BT(clef_frame, T_logits)
                    #onset_roll  = _pool_bt128(onset_roll,  T_logits)
                    #offset_roll = _pool_bt128(offset_roll, T_logits)
                    #hand_frame  = _interp_bt(hand_frame,   T_logits)
                    #clef_frame  = _interp_bt(clef_frame,   T_logits)

                # --- derive ANY-note targets at logits time ---
                onset_any  = (onset_roll  > 0).any(dim=-1).float()   # (B, T_logits)
                offset_any = (offset_roll > 0).any(dim=-1).float()   # (B, T_logits)

                # --- binarize predictions ---
                thr = float(cfg.get("training", {}).get("metrics", {}).get("prob_threshold", 0.5))
                onset_pred  = (torch.sigmoid(out["onset_logits"])  >= thr).float()  # (B, T_logits)
                offset_pred = (torch.sigmoid(out["offset_logits"]) >= thr).float()  # (B, T_logits)

                # --- masked F1 and positive-rate diagnostics ---
                f1_on  = _binary_f1(onset_pred.reshape(-1),  onset_any.reshape(-1))
                f1_off = _binary_f1(offset_pred.reshape(-1), offset_any.reshape(-1))
                if f1_on is not None:
                    metric_counts["onset_f1"] += f1_on
                    metric_counts["n_on"] += 1
                if f1_off is not None:
                    metric_counts["offset_f1"] += f1_off
                    metric_counts["n_off"] += 1

                metric_counts["onset_pos_rate"]  += onset_any.mean().item()
                metric_counts["offset_pos_rate"] += offset_any.mean().item()
                metric_counts["onset_pred_rate"]  += onset_pred.mean().item()
                metric_counts["offset_pred_rate"] += offset_pred.mean().item()

                # --- per-frame hand/clef accuracy ---
                hand_pred = out["hand_logits"].argmax(dim=-1)   # (B, T_logits)
                clef_pred = out["clef_logits"].argmax(dim=-1)   # (B, T_logits)
                Bx, Tx = hand_pred.shape
                metric_counts["hand_acc"] += (hand_pred.reshape(Bx*Tx) == hand_frame.reshape(Bx*Tx)).float().mean().item()
                metric_counts["clef_acc"] += (clef_pred.reshape(Bx*Tx) == clef_frame.reshape(Bx*Tx)).float().mean().item()

                metric_n += 1


            else:
                # ---- clip-level metrics (existing path) ----
                pitch_pred = out["pitch_logits"].argmax(dim=-1)   # (B,)
                metric_counts["pitch_acc"] += _acc_from_logits(out["pitch_logits"], tgt["pitch"])
                metric_counts["hand_acc"]  += _acc_from_logits(out["hand_logits"],  tgt["hand"])
                metric_counts["clef_acc"]  += _acc_from_logits(out["clef_logits"],  tgt["clef"])

                onset_pred_c  = (torch.sigmoid(out["onset_logits"])  >= thr).float()
                offset_pred_c = (torch.sigmoid(out["offset_logits"]) >= thr).float()

                onset_gt_bin  = (tgt["onset"]  >= 0.5).float()
                offset_gt_bin = (tgt["offset"] >= 0.5).float()

                onset_f1  = _binary_f1(onset_pred_c,  onset_gt_bin)
                offset_f1 = _binary_f1(offset_pred_c, offset_gt_bin)
                if onset_f1 is not None:
                    metric_counts["onset_f1"] += onset_f1
                    metric_counts["n_on"] += 1
                if offset_f1 is not None:
                    metric_counts["offset_f1"] += offset_f1
                    metric_counts["n_off"] += 1

                metric_counts["onset_pos_rate"]  += onset_gt_bin.mean().item()
                metric_counts["offset_pos_rate"] += offset_gt_bin.mean().item()
                metric_n += 1

    # averages
    losses = {k: sums[k] / max(1, n_batches) for k in sums}
    metrics = {}
    metrics["pitch_acc"]  = metric_counts["pitch_acc"] / max(1, metric_n)
    metrics["hand_acc"]   = metric_counts["hand_acc"]  / max(1, metric_n)
    metrics["clef_acc"]   = metric_counts["clef_acc"]  / max(1, metric_n)
    metrics["onset_f1"]   = metric_counts["onset_f1"]  / max(1, metric_counts.get("n_on", 0))
    metrics["offset_f1"]  = metric_counts["offset_f1"] / max(1, metric_counts.get("n_off", 0))
    metrics["onset_pos_rate"]  = metric_counts["onset_pos_rate"]  / max(1, metric_n)
    metrics["offset_pos_rate"] = metric_counts["offset_pos_rate"] / max(1, metric_n)
    metrics["onset_pred_rate"]  = metric_counts["onset_pred_rate"]  / max(1, metric_n)
    metrics["offset_pred_rate"] = metric_counts["offset_pred_rate"] / max(1, metric_n)

    return {**losses, **metrics}

def main():
    set_seed(42)
    cfg = load_config("configs/config.yaml")
    ckpt_dir, _ = ensure_dirs(cfg)

    # Build experiment-specific log dir
    exp_name = cfg.get("experiment", {}).get("name", "default")
    log_root = Path(cfg["logging"]["log_dir"])
    log_dir = log_root / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    use_tb = bool(cfg["logging"].get("tensorboard", False))
    writer = SummaryWriter(log_dir) if use_tb else None
    
    # Data
    #split = cfg["dataset"].get("split", "test")
    #loader = make_dataloader(cfg, split=split)
    train_split = cfg["dataset"].get("split_train", "train")
    val_split   = cfg["dataset"].get("split_val", "val")
    train_loader = make_dataloader(cfg, split=train_split)

    # If you have a dedicated val split, use it; otherwise reuse "test" as a stand-in.
    val_loader = None
    if cfg["training"].get("eval_freq", 0):
        # try to build test or val loader
        try:
            val_loader = make_dataloader(cfg, split=val_split)
        except Exception:
            try:
                val_loader = make_dataloader(cfg, split=cfg["dataset"].get("split_test","test"))
            except Exception:
                val_loader = None

    
    # Model & optimizer
    model = build_model(cfg)
    optimizer = AdamW(model.parameters(),
                      lr=float(cfg["training"]["learning_rate"]),
                      weight_decay=float(cfg["training"]["weight_decay"]))

    # Train
    epochs = int(cfg["training"]["epochs"])
    save_every = int(cfg["training"].get("save_every", 1))
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = math.inf
    best_path = ckpt_dir / "tivit_best.pt"
    last_path = ckpt_dir / "tivit_last.pt"
    eval_freq = int(cfg["training"].get("eval_freq", 0))

    # Try to resume from checkpoint
    resume_path = ckpt_dir / "tivit_best.pt"
    if resume_path.exists():
        print(f"[resume] Loading from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val = ckpt.get("best_val", math.inf)
        start_epoch = ckpt.get("epoch", 0) + 1
    else:
        # Fresh init → apply onset/offset bias if requested
        if cfg.get("training", {}).get("reset_head_bias", True):
            prior_cfg = float(cfg.get("training",{}).get("loss_weights",{}).get("onoff_prior_mean", 0.02))
            _set_onoff_head_bias(model, prior=prior_cfg)
        best_val = math.inf
        start_epoch = 1
        
    for epoch in range(start_epoch, epochs + 1):    
        t0 = perf_counter()
        # --- train one epoch ---
        model.train()
        crit = make_criterions()
        w = get_loss_weights(cfg) if "get_loss_weights" in globals() else cfg["training"]["loss_weights"]
        log_interval = int(cfg["training"].get("log_interval", 20))
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", ncols=100)
        running = {"total": 0.0, "pitch": 0.0, "onset": 0.0, "offset": 0.0, "hand": 0.0, "clef": 0.0}
        count = 0
        avg_events_recent = None  # for TB data stat

        #CAL
        # after you build train_loader / just before the for-it loop
        _first_batch = next(iter(train_loader))
        _have_frame_keys = all(k in _first_batch for k in ("pitch_roll","onset_roll","offset_roll","hand_frame","clef_frame"))
        print(f"[DEBUG] frame-mode keys present: {_have_frame_keys}")
        del _first_batch
        #END CAL
        
        for it, batch in pbar:
            x = batch["video"]
            B = x.shape[0]

            # Prefer real labels; fallback to dummy
            want = ("pitch", "onset", "offset", "hand", "clef")
            have_all = all(k in batch for k in want)
            use_dummy_flag = bool(cfg["training"].get("debug_dummy_labels", False))
            if have_all and not use_dummy_flag:
                tgt = {k: batch[k] for k in want}
                tgt["pitch"]  = tgt["pitch"].long()
                tgt["hand"]   = tgt["hand"].long()
                tgt["clef"]   = tgt["clef"].long()
                tgt["onset"]  = tgt["onset"].float()
                tgt["offset"] = tgt["offset"].float()
            else:
                tgt = fabricate_dummy_targets(B)

            out = model(x)

            #DEB
            if it % 20 == 0:
            with torch.no_grad():
                on = out["onset_logits"]             # expect (B, T') or (B, T', 1)
                if on.dim() == 3 and on.size(-1) == 1: on = on.squeeze(-1)
                std_per_sample = on.std(dim=1).mean().item()   # average temporal std over batch
            pbar.write(f"[diag] onset temporal std (mean over batch) = {std_per_sample:.4f}")

            #END DEB
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gc = float(cfg["training"].get("grad_clip", 0.0))
            if gc and gc > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gc)
             # CAL >>> diagnostics here <<<
            _print_head_grad_norms(model, step_tag=f"e{epoch}_it{it}")
            optimizer.step()

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

        # epoch metrics
        train_metrics = {k: running[k] / max(1, count) for k in running}
        dt = perf_counter() - t0
        print(f"Epoch {epoch} | time {dt:.1f}s | " + " ".join([f"{k}={v:.3f}" for k, v in train_metrics.items()]))

        # TB logging (train scalars + data stat)
        if writer is not None:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            if avg_events_recent is not None:
                writer.add_scalar("data/avg_events_per_clip", avg_events_recent, epoch)

        # --- evaluation & best checkpoint ---
        val_total = None
        if eval_freq and val_loader is not None and (epoch % eval_freq == 0):
            val_metrics = evaluate_one_epoch(model, val_loader, cfg)
            print("Val:", " ".join([f"{k}={v:.3f}" for k, v in val_metrics.items()]))
            if writer is not None:
                for k, v in val_metrics.items():
                    writer.add_scalar(f"val/{k}", v, epoch)
            val_total = val_metrics["total"]
            if val_total < best_val:
                best_val = val_total
                save_checkpoint(best_path, model, optimizer, epoch, cfg, best_val)
                print(f"Saved BEST to: {best_path} (val_total={val_total:.3f})")

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

