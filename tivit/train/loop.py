"""TiViT-Piano training loop entrypoint (new implementation)."""

# Purpose:
# - Load/merge configs, apply CLI overrides, and configure determinism.
# - Build dataloaders/model/loss/optimizer and run AMP-capable training.
# - Support per-tile supervision, evaluation, and checkpoint resume/save.
#
# Key Functions/Classes:
# - run_training: orchestrate the full training flow without legacy dependencies.
# - PerTileSupport: build per-tile masks/context for per-tile loss paths.
#
# CLI Arguments:
# - (none directly; invoked via pipelines/CLI wrappers).
#
# Usage:
# - from tivit.train.loop import run_training

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from tivit.core.config import DEFAULT_CONFIG_PATH, load_experiment_config
from tivit.core.determinism import configure_determinism, resolve_deterministic_flag, resolve_seed
from tivit.data.loaders import make_dataloader
from tivit.data.targets.identifiers import canonical_video_id
from tivit.decoder.global_fusion import build_batch_tile_mask
from tivit.decoder.tile_support_cache import CacheScope, TileSupportCache
from tivit.losses.multitask_loss import MultitaskLoss
from tivit.models import build_model
from tivit.train.callbacks import Callback, CallbackList
from tivit.train.eval_loop import run_evaluation
from tivit.train.optim import build_optimizer
from tivit.utils.amp import autocast, grad_scaler
from tivit.utils.logging import get_logger, log_final_result, log_stage

LOGGER = get_logger(__name__)


def _apply_overrides(
    cfg: Mapping[str, Any],
    train_split: str | None,
    val_split: str | None,
    max_clips: int | None,
    frames: int | None,
    smoke: bool,
) -> tuple[dict[str, Any], str, str | None]:
    """Return a copied config with CLI overrides applied."""
    cfg_copy: dict[str, Any] = dict(copy.deepcopy(cfg))
    dataset_cfg = cfg_copy.setdefault("dataset", {})
    training_cfg = cfg_copy.setdefault("training", {})

    if frames is not None:
        dataset_cfg["frames"] = int(frames)
    if max_clips is not None:
        dataset_cfg["max_clips"] = int(max_clips)

    train_split_name = train_split or dataset_cfg.get("split_train") or dataset_cfg.get("split") or "train"
    val_split_name = val_split or dataset_cfg.get("split_val") or None
    dataset_cfg.setdefault("split_train", train_split_name)
    if val_split_name:
        dataset_cfg.setdefault("split_val", val_split_name)
    dataset_cfg.setdefault("split", train_split_name)

    if smoke:
        training_cfg["epochs"] = 1
        training_cfg["log_interval"] = 1
        dataset_cfg["max_clips"] = min(int(dataset_cfg.get("max_clips", 2) or 2), 2)
        dataset_cfg["batch_size"] = min(int(dataset_cfg.get("batch_size", 1) or 1), 2)
        dataset_cfg["num_workers"] = 0

    return cfg_copy, str(train_split_name), val_split_name


def _resolve_batch_clip_ids(batch: Mapping[str, Any], batch_size: int) -> list[str]:
    ids = batch.get("video_uid") or batch.get("video_uids") or batch.get("uid")
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)):
        resolved = [canonical_video_id(v) for v in ids[:batch_size]]
        if resolved:
            return resolved
    if isinstance(ids, str):
        return [canonical_video_id(ids) for _ in range(batch_size)]
    paths = batch.get("path") or batch.get("paths")
    if isinstance(paths, (list, tuple)):
        resolved = [canonical_video_id(p) for p in paths[:batch_size]]
        if resolved:
            return resolved
    return [f"video_{idx}" for idx in range(batch_size)]


class PerTileSupport:
    """Resolve per-tile masks for optional per-tile supervision."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        dataset: Any,
        *,
        phase: str,
        tile_cache: TileSupportCache | None = None,
    ) -> None:
        training_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
        loss_cfg = training_cfg.get("loss", {}) if isinstance(training_cfg, Mapping) else {}
        per_tile_cfg = loss_cfg.get("per_tile", {}) or {}
        self.enabled = bool(per_tile_cfg.get("enabled", False))
        self.heads = tuple(str(h).lower() for h in per_tile_cfg.get("heads", ("pitch", "onset", "offset")))
        decoder_cfg = cfg.get("decoder", {}).get("global_fusion", {}) if isinstance(cfg, Mapping) else {}
        default_cushion = int((decoder_cfg or {}).get("cushion_keys", 0))
        cushion_override = per_tile_cfg.get("mask_cushion_keys")
        self.mask_cushion = default_cushion if cushion_override is None else int(cushion_override)
        dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
        self.tiles = int(dataset_cfg.get("tiles", 3))
        self.cache_scope: CacheScope = "eval" if str(phase).lower() == "eval" else "train"
        self.tile_cache = tile_cache if tile_cache is not None else TileSupportCache()
        self.reg_refiner = getattr(dataset, "registration_refiner", None)
        self.reg_meta_cache: dict[str, dict[str, Any]] = {}

    @property
    def request_per_tile_outputs(self) -> bool:
        return self.enabled

    def build_context(self, outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if not self.enabled:
            return None
        pitch_tile = outputs.get("pitch_tile")
        onset_tile = outputs.get("onset_tile")
        offset_tile = outputs.get("offset_tile")
        if not (torch.is_tensor(pitch_tile) and torch.is_tensor(onset_tile) and torch.is_tensor(offset_tile)):
            return None
        batch_size = int(pitch_tile.shape[0])
        key_dim = int(pitch_tile.shape[-1])
        canonical_hw = getattr(self.reg_refiner, "canonical_hw", None)
        mask_batch = build_batch_tile_mask(
            _resolve_batch_clip_ids(batch, batch_size),
            cache=self.tile_cache,
            cache_scope=self.cache_scope,
            reg_meta_cache=self.reg_meta_cache,
            reg_refiner=self.reg_refiner,
            num_tiles=self.tiles,
            cushion_keys=self.mask_cushion,
            n_keys=key_dim,
            canonical_hw=canonical_hw,
        )
        return {
            "enabled": True,
            "heads": self.heads,
            "mask": mask_batch.tensor,
            "pitch": pitch_tile,
            "onset": onset_tile,
            "offset": offset_tile,
        }


def _fabricate_dummy_targets(
    outputs: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create lightweight synthetic targets when labels are absent."""
    pitch_logits = outputs.get("pitch_logits")
    if pitch_logits is None:
        raise ValueError("Missing pitch_logits; cannot fabricate dummy targets.")
    if pitch_logits.dim() == 3:
        b, t, p = pitch_logits.shape
        zeros = lambda shape, dtype=torch.float32: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
        return {
            "pitch": zeros((b, t, p)),
            "onset": zeros((b, t, p)),
            "offset": zeros((b, t, p)),
            "hand": zeros((b, t), dtype=torch.long),
            "clef": zeros((b, t), dtype=torch.long),
        }
    b, p = pitch_logits.shape
    zeros = lambda shape, dtype=torch.float32: torch.zeros(shape, device=device, dtype=dtype)  # noqa: E731
    return {
        "pitch": zeros((b, p)),
        "onset": zeros((b, p)),
        "offset": zeros((b, p)),
        "hand": zeros((b,), dtype=torch.long),
        "clef": zeros((b,), dtype=torch.long),
    }


def _prepare_targets(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    debug_dummy_labels: bool,
) -> Mapping[str, torch.Tensor]:
    targets: dict[str, torch.Tensor] = {}
    for key in ("pitch", "onset", "offset", "hand", "clef", "hand_frame_mask", "hand_mask", "hand_reach", "hand_reach_valid"):
        value = batch.get(key)
        if torch.is_tensor(value):
            targets[key] = value.to(device=device, non_blocking=True)

    if targets or not debug_dummy_labels:
        return targets
    return _fabricate_dummy_targets(outputs, device)


def _seed_onoff_head_bias(
    model: torch.nn.Module,
    onset_prior: float,
    offset_prior: float,
) -> None:
    """Initialize onset/offset head biases from Bernoulli priors."""

    def _seed(head: Any, prior: float, name: str) -> None:
        if head is None:
            return
        prior_clamped = min(max(prior, 1e-6), 1.0 - 1e-6)
        logit = math.log(prior_clamped / (1.0 - prior_clamped))
        target = None
        if isinstance(head, torch.nn.Sequential) and len(head) > 0:
            target = head[-1]
        elif isinstance(getattr(head, "net", None), torch.nn.Sequential) and len(head.net) > 0:  # type: ignore[operator]
            target = head.net[-1]  # type: ignore[index]
        bias_tensor = getattr(target, "bias", None)
        if target is None or bias_tensor is None or not torch.is_tensor(bias_tensor):
            return
        with torch.no_grad():
            bias_tensor.fill_(logit)
        LOGGER.info("Seeded %s head bias to prior=%.4f (logit=%.4f)", name, prior_clamped, logit)

    _seed(getattr(model, "head_onset", None), onset_prior, "onset")
    _seed(getattr(model, "head_offset", None), offset_prior, "offset")


def _resolve_prior_mean(cfg: Mapping[str, Any], head: str, default: float = 0.02) -> float:
    """Fetch prior_mean for a head with basic validation."""
    try:
        value = float(
            (cfg.get("training", {}) if isinstance(cfg, Mapping) else {})
            .get("loss", {})
            .get("heads", {})
            .get(head, {})
            .get("prior_mean")
        )
    except (TypeError, ValueError):
        LOGGER.warning("Missing or invalid prior_mean for head '%s'; using default %.3f", head, default)
        return default
    if not (0.0 < value < 1.0) or not math.isfinite(value):
        LOGGER.warning("Out-of-range prior_mean %.4f for head '%s'; using default %.3f", value, head, default)
        return default
    return value


def _compute_loss_for_batch(
    model: torch.nn.Module,
    loss_fn: MultitaskLoss,
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    per_tile_support: PerTileSupport | None,
    amp_enabled: bool,
    debug_dummy_labels: bool,
    update_state: bool,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    video = batch.get("video")
    if not torch.is_tensor(video):
        raise ValueError("Batch is missing tensor key 'video'")
    x = video.to(device=device, non_blocking=True)
    request_per_tile = per_tile_support.request_per_tile_outputs if per_tile_support is not None else False
    with autocast(device, enabled=amp_enabled):
        outputs = model(x, return_per_tile=request_per_tile)
        per_tile_ctx = per_tile_support.build_context(outputs, batch) if per_tile_support is not None else None
        targets = _prepare_targets(outputs, batch, device, debug_dummy_labels=debug_dummy_labels)
        loss, parts = loss_fn(outputs, targets, update_state=update_state, per_tile=per_tile_ctx)
    return loss, parts


def _train_one_epoch(
    model: torch.nn.Module,
    loss_fn: MultitaskLoss,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    *,
    epoch: int,
    grad_clip: float,
    accum_steps: int,
    log_interval: int,
    amp_enabled: bool,
    per_tile_support: PerTileSupport | None,
    debug_dummy_labels: bool,
) -> Mapping[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_batches = 0
    parts_sum: dict[str, float] = {}

    for step, batch in enumerate(loader):
        loss, parts = _compute_loss_for_batch(
            model,
            loss_fn,
            batch,
            device,
            per_tile_support=per_tile_support,
            amp_enabled=amp_enabled,
            debug_dummy_labels=debug_dummy_labels,
            update_state=True,
        )

        loss_to_backprop = loss / max(1, accum_steps)
        scaler.scale(loss_to_backprop).backward()

        should_step = ((step + 1) % max(1, accum_steps)) == 0
        if should_step:
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu())
        total_batches += 1
        for key, value in parts.items():
            try:
                parts_sum[key] = parts_sum.get(key, 0.0) + float(value)
            except (TypeError, ValueError):
                continue

        if log_interval > 0 and ((step + 1) % log_interval) == 0:
            avg_loss = total_loss / max(1, total_batches)
            LOGGER.info("[train] epoch=%d step=%d avg_loss=%.4f", epoch, step + 1, avg_loss)

    # Final optimizer step for leftover grads.
    if total_batches > 0 and (total_batches % max(1, accum_steps)) != 0:
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if total_batches == 0:
        return {}

    metrics = {"loss": total_loss / total_batches}
    for key, value in parts_sum.items():
        metrics[key] = value / total_batches
    return metrics


def _find_latest_checkpoint(path: Path) -> Path | None:
    if not path.exists():
        return None
    candidates = sorted(path.glob("epoch_*.pt"))
    return candidates[-1] if candidates else None


def _load_checkpoint(
    checkpoint: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
) -> int:
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload.get("model", payload))
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "scaler" in payload and payload["scaler"] is not None:
        try:
            scaler.load_state_dict(payload["scaler"])
        except Exception:
            pass
    return int(payload.get("epoch", 0))


def _save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )
    return path


def run_training(
    configs: Sequence[str | Path] | None = None,
    *,
    verbose: str | None = "quiet",  # kept for interface compatibility
    train_split: str | None = None,
    val_split: str | None = None,
    max_clips: int | None = None,
    frames: int | None = None,
    seed: int | None = None,
    deterministic: bool | None = None,
    smoke: bool = False,
    callbacks: Iterable[Callback] | None = None,
) -> None:
    """Train TiViT models using the new layout (no legacy dependencies)."""
    del verbose  # logging is configured by the pipeline entrypoint

    cfg = load_experiment_config(configs)
    cfg, train_split_name, val_split_name = _apply_overrides(cfg, train_split, val_split, max_clips, frames, smoke)

    log_stage("train", f"starting training with configs={configs or [str(DEFAULT_CONFIG_PATH)]}")

    seed_val = resolve_seed(seed, cfg)
    deterministic_flag = resolve_deterministic_flag(deterministic, cfg, default=True)
    configure_determinism(seed_val, deterministic=deterministic_flag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}
    reset_head_bias = bool(training_cfg.get("reset_head_bias", True))

    model = build_model(cfg).to(device)
    loss_fn = MultitaskLoss(cfg)
    optimizer = build_optimizer(model, cfg)
    amp_enabled = bool(training_cfg.get("amp", False))
    scaler = grad_scaler(device, enabled=amp_enabled)
    grad_clip = float(cfg.get("optim", {}).get("grad_clip", 0.0))
    accum_steps = int(cfg.get("train", {}).get("accumulate_steps", 1))
    log_interval = int(training_cfg.get("log_interval", 10))
    save_every = int(training_cfg.get("save_every", 0) or 0)
    eval_freq = max(1, int(training_cfg.get("eval_freq", 1)))
    epochs = int(training_cfg.get("epochs", 1))
    debug_dummy_labels = bool(training_cfg.get("debug_dummy_labels", False))

    train_loader = make_dataloader(cfg, train_split_name, drop_last=True, seed=seed_val)
    val_loader = make_dataloader(cfg, val_split_name, drop_last=False, seed=seed_val) if val_split_name else None

    tile_cache = TileSupportCache()
    per_tile_train = PerTileSupport(cfg, getattr(train_loader, "dataset", None), phase="train", tile_cache=tile_cache)
    per_tile_eval = (
        PerTileSupport(cfg, getattr(val_loader, "dataset", None), phase="eval", tile_cache=tile_cache)
        if val_loader is not None
        else None
    )

    cb = CallbackList(callbacks)

    checkpoint_dir = Path(cfg.get("logging", {}).get("checkpoint_dir", "./checkpoints")).expanduser()
    start_epoch = 1
    if bool(training_cfg.get("resume", False)):
        latest = _find_latest_checkpoint(checkpoint_dir)
        if latest:
            try:
                loaded_epoch = _load_checkpoint(latest, model, optimizer, scaler, device)
                start_epoch = int(loaded_epoch) + 1
                LOGGER.info("Resumed from %s (epoch %d)", latest, loaded_epoch)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to resume from %s: %s", latest, exc)
    if reset_head_bias and start_epoch == 1:
        onset_prior = _resolve_prior_mean(cfg, "onset", default=0.02)
        offset_prior = _resolve_prior_mean(cfg, "offset", default=0.02)
        _seed_onoff_head_bias(model, onset_prior, offset_prior)

    try:
        for epoch in range(start_epoch, epochs + 1):
            cb.on_epoch_start(epoch)
            train_metrics = _train_one_epoch(
                model,
                loss_fn,
                optimizer,
                scaler,
                train_loader,
                device,
                epoch=epoch,
                grad_clip=grad_clip,
                accum_steps=accum_steps,
                log_interval=log_interval,
                amp_enabled=amp_enabled,
                per_tile_support=per_tile_train,
                debug_dummy_labels=debug_dummy_labels,
            )
            if train_metrics:
                LOGGER.info("[train] epoch=%d metrics=%s", epoch, train_metrics)
            eval_metrics = None
            if val_loader is not None and (epoch % eval_freq == 0):
                eval_step = lambda batch: _compute_loss_for_batch(  # noqa: E731
                    model,
                    loss_fn,
                    batch,
                    device,
                    per_tile_support=per_tile_eval,
                    amp_enabled=amp_enabled,
                    debug_dummy_labels=debug_dummy_labels,
                    update_state=False,
                )
                eval_metrics = run_evaluation(eval_step, val_loader, max_batches=None)
                LOGGER.info("[eval] epoch=%d metrics=%s", epoch, eval_metrics)

            cb.on_epoch_end(epoch, metrics={"train": train_metrics, "eval": eval_metrics})

            if save_every > 0 and (epoch % save_every == 0):
                ckpt_path = _save_checkpoint(checkpoint_dir, epoch, model, optimizer, scaler)
                LOGGER.info("Saved checkpoint %s", ckpt_path)

        cb.on_train_end()
    except Exception:
        log_final_result("train", "training run failed")
        raise

    log_final_result("train", "training run finished")


__all__ = ["run_training"]
