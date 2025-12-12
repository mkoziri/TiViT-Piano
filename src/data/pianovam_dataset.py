"""Purpose:
    Provide a PianoVAM loader that mirrors the PianoYT runtime surface so
    pipelines can switch datasets via configuration only.  The loader reuses
    the PianoYT implementation for sampling, target construction, and optional
    augmentations while resolving a PianoVAM-specific root directory.

Key Functions/Classes:
    - PianoVAMDataset: Thin subclass of :class:`PianoYTDataset` that points to
      the local PianoVAM folder (PianoYT-style layout on disk).
    - make_dataloader(): Factory matching :func:`pianoyt_dataset.make_dataloader`
      but defaulting to PianoVAM roots and leaving the rest of the interface
      unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from utils.determinism import DEFAULT_SEED, make_loader_components
from utils.frame_targets import resolve_frame_target_spec, resolve_soft_target_config
from utils.identifiers import canonical_video_id

from .pianoyt_dataset import (
    PianoYTDataset,
    _read_excluded,
    _safe_expanduser,
)
from .sampler_utils import build_onset_balanced_sampler

LOGGER = logging.getLogger(__name__)


def _expand_root(root_dir: Optional[str]) -> Path:
    """Resolve the PianoVAM root directory with environment fallbacks."""

    candidates = []
    if root_dir:
        candidates.append(_safe_expanduser(os.path.expandvars(str(root_dir))))

    env_base = os.environ.get("TIVIT_DATA_DIR") or os.environ.get("DATASETS_HOME")
    if env_base:
        env_base_path = _safe_expanduser(env_base)
        candidates.extend(
            [
                env_base_path / "PianoVAM",
                env_base_path / "PianoVAM_v1.0",
            ]
        )

    project_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            project_root / "data" / "PianoVAM",
            project_root / "data" / "PianoVAM_v1.0",
        ]
    )
    candidates.extend(
        [
            _safe_expanduser("~/datasets/PianoVAM"),
            _safe_expanduser("~/datasets/PianoVAM_v1.0"),
        ]
    )

    for cand in candidates:
        if cand and cand.exists():
            return cand

    msg = (
        "Unable to locate PianoVAM root. Set dataset.root_dir or define "
        "TIVIT_DATA_DIR/DATASETS_HOME to point at a PianoYT-style PianoVAM layout."
    )
    LOGGER.error(msg)
    raise FileNotFoundError(msg)


class PianoVAMDataset(PianoYTDataset):
    """PianoVAM dataset that reuses the PianoYT loader surface."""

    def __init__(
        self,
        root_dir: Optional[str],
        split: str = "test",
        frames: int = 32,
        stride: int = 2,
        resize: Tuple[int, int] = (224, 224),
        tiles: int = 3,
        channels: int = 3,
        normalize: bool = True,
        manifest: Optional[str] = None,
        decode_fps: float = 30.0,
        *,
        dataset_cfg: Optional[Mapping[str, Any]] = None,
        full_cfg: Optional[Mapping[str, Any]] = None,
        only_video: Optional[str] = None,
        avlag_disabled: Optional[bool] = None,
    ):
        resolved_root = _expand_root(root_dir or (dataset_cfg or {}).get("root_dir"))
        super().__init__(
            root_dir=str(resolved_root),
            split=split,
            frames=frames,
            stride=stride,
            resize=resize,
            tiles=tiles,
            channels=channels,
            normalize=normalize,
            manifest=manifest,
            decode_fps=decode_fps,
            dataset_cfg=dataset_cfg,
            full_cfg=full_cfg,
            only_video=only_video,
            avlag_disabled=avlag_disabled,
        )
        self.dataset_name = "pianovam"
        self.root = Path(resolved_root)


def make_dataloader(
    cfg: Mapping[str, Any],
    split: str,
    drop_last: bool = False,
    *,
    seed: Optional[int] = None,
):
    dcfg = cfg["dataset"]
    manifest_cfg = dcfg.get("manifest", {}) or {}
    manifest_path = manifest_cfg.get(split)

    decode_fps = float(dcfg.get("decode_fps", 30.0))
    hop_seconds = float(dcfg.get("hop_seconds", 1.0 / decode_fps))
    stride = int(round(hop_seconds * decode_fps))

    only_video_cfg = dcfg.get("only_video")
    avlag_disabled_cfg = bool(dcfg.get("avlag_disabled", False))

    resolved_root = _expand_root(dcfg.get("root_dir"))
    dataset_cfg = dict(dcfg)
    dataset_cfg["root_dir"] = str(resolved_root)

    dataset = PianoVAMDataset(
        root_dir=str(resolved_root),
        split=split,
        frames=int(dataset_cfg.get("frames", 32)),
        stride=stride,
        resize=tuple(dataset_cfg.get("resize", [224, 224])),
        tiles=int(dataset_cfg.get("tiles", 3)),
        channels=int(dataset_cfg.get("channels", 3)),
        normalize=bool(dataset_cfg.get("normalize", True)),
        manifest=manifest_path,
        decode_fps=decode_fps,
        dataset_cfg=dataset_cfg,
        full_cfg=cfg,
        only_video=only_video_cfg,
        avlag_disabled=avlag_disabled_cfg,
    )
    training_cfg = cfg.get("training", {}) if isinstance(cfg, Mapping) else {}
    soft_cfg_raw = training_cfg.get("soft_targets") if isinstance(training_cfg, Mapping) else None
    dataset.soft_target_cfg = resolve_soft_target_config(soft_cfg_raw)

    include_low = bool(dataset_cfg.get("include_low_res", False))
    excluded_ids = set()
    if not include_low:
        excluded_ids = _read_excluded(dataset.root, dataset_cfg.get("excluded_list"))
    dataset.configure(
        include_low_res=include_low,
        excluded_ids=excluded_ids,
        apply_crop=bool(dataset_cfg.get("apply_crop", True)),
        crop_rescale=str(dataset_cfg.get("crop_rescale", "auto")),
    )

    if only_video_cfg and not getattr(dataset, "_only_filter_applied", False):
        only_canon = canonical_video_id(only_video_cfg)
        if not dataset.filter_to_video(only_canon):
            LOGGER.warning("[PianoVAM] --only filter skipped; id=%s not found", only_canon)

    max_clips = dataset_cfg.get("max_clips")
    dataset.limit_max_clips(max_clips if isinstance(max_clips, int) else None)
    dataset.max_clips = max_clips
    dataset.args_max_clips_or_None = max_clips if isinstance(max_clips, int) else None

    dataset.annotations_root = dataset_cfg.get("annotations_root")
    dataset.label_format = dataset_cfg.get("label_format", "midi")
    dataset.label_targets = dataset_cfg.get("label_targets", ["pitch", "onset", "offset", "hand", "clef"])
    dataset.require_labels = bool(dataset_cfg.get("require_labels", False))
    dataset.frame_targets_cfg = dict(dataset_cfg.get("frame_targets", {}) or {})
    dataset.frame_target_spec = resolve_frame_target_spec(
        dataset.frame_targets_cfg,
        frames=dataset.frames,
        stride=stride,
        fps=decode_fps,
        canonical_hw=dataset.canonical_hw,
    )
    dataset.frame_target_summary = (
        dataset.frame_target_spec.summary()
        if dataset.frame_target_spec is not None
        else "targets_conf: disabled"
    )
    dataset._rebuild_valid_index_cache(log_summary=False)

    def _collate(batch):
        vids = [b["video"] for b in batch]
        paths = [b["path"] for b in batch]
        if not vids:
            raise RuntimeError("Empty batch supplied to PianoVAM collate function")

        dims = vids[0].dim()
        if any(v.dim() != dims for v in vids):
            raise RuntimeError("Inconsistent tensor ranks in PianoVAM batch")

        max_shape = tuple(max(v.shape[d] for v in vids) for d in range(dims))
        x = vids[0].new_zeros((len(vids),) + max_shape)
        for idx, vid in enumerate(vids):
            slices = tuple(slice(0, size) for size in vid.shape)
            x[(idx,) + slices] = vid

        out = {"video": x, "path": paths}
        extra_keys = set().union(*[set(d.keys()) for d in batch]) - {"video", "path"}
        for k in extra_keys:
            vals = [d[k] for d in batch if k in d]
            if len(vals) != len(batch):
                continue
            if k == "labels":
                out[k] = vals
            else:
                v0 = vals[0]
                if torch.is_tensor(v0):
                    try:
                        out[k] = torch.stack(vals, dim=0)
                    except Exception:
                        out[k] = vals
                else:
                    out[k] = vals
        return out

    num_workers = int(dataset_cfg.get("num_workers", 0))
    pin_memory = bool(dataset_cfg.get("pin_memory", False))
    persistent_workers_cfg = bool(dataset_cfg.get("persistent_workers", False))
    persistent_workers = persistent_workers_cfg if num_workers > 0 else False
    prefetch_factor_cfg = dataset_cfg.get("prefetch_factor")
    prefetch_factor: Optional[int] = None
    if num_workers > 0 and prefetch_factor_cfg is not None:
        try:
            prefetch_factor_candidate = int(prefetch_factor_cfg)
        except (TypeError, ValueError):
            prefetch_factor_candidate = None
        if prefetch_factor_candidate is not None and prefetch_factor_candidate > 0:
            prefetch_factor = prefetch_factor_candidate

    base_seed = seed if seed is not None else getattr(dataset, "data_seed", None)
    if base_seed is None:
        base_seed = DEFAULT_SEED
    generator, worker_init_fn = make_loader_components(
        int(base_seed), namespace=f"{dataset.__class__.__name__}:{split}"
    )

    sampler = build_onset_balanced_sampler(
        dataset,
        dataset_cfg.get("sampler"),
        base_seed=int(base_seed),
    )
    shuffle_flag = bool(dataset_cfg.get("shuffle", True)) if split == "train" else False
    if sampler is not None:
        shuffle_flag = False

    loader_kwargs: dict[str, Any] = dict(
        dataset=dataset,
        batch_size=int(dataset_cfg.get("batch_size", 2)),
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**loader_kwargs)
    return loader


__all__ = ["PianoVAMDataset", "make_dataloader"]
