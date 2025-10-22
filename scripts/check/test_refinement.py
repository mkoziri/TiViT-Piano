"""Purpose:
    Exercise the registration refinement pipeline on a small dataset slice and
    report both the observed log output and cached homography statistics.

Key Functions/Classes:
    - run_dataset_probe(): Iterates a limited number of clips to trigger the
      registration refiner and capture basic throughput metrics.
    - inspect_cache(): Summarises recent cache entries, comparing refined
      homographies against the baseline scale transform.
    - main(): CLI entry point wiring together configuration overrides, dataset
      sampling, and cache inspection.

CLI:
    python scripts/check/test_refinement.py --split val --clips 2 --inspect 4
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from data import make_dataloader
from utils import load_config


def _resolve_split(cfg: Mapping[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    dataset = cfg.get("dataset", {})
    if isinstance(dataset, Mapping):
        for key in ("split_val", "split_eval", "split"):
            value = dataset.get(key)
            if isinstance(value, str):
                return value
    return "val"


def _apply_overrides(
    cfg: Mapping[str, Any],
    *,
    frames: Optional[int],
    max_clips: Optional[int],
    batch_size: Optional[int],
) -> None:
    dataset_cfg = cfg.get("dataset")
    if not isinstance(dataset_cfg, MutableMapping):
        raise ValueError("configuration missing 'dataset' mapping")
    dataset_cfg = cast(MutableMapping[str, Any], dataset_cfg)

    if frames is not None:
        dataset_cfg["frames"] = int(frames)

    if max_clips is not None:
        dataset_cfg["max_clips"] = int(max(max_clips, 1))

    if batch_size is not None:
        dataset_cfg["batch_size"] = int(max(batch_size, 1))
        if "shuffle" in dataset_cfg:
            dataset_cfg["shuffle"] = False


def _get_dataset_name(cfg: Mapping[str, Any]) -> str:
    dataset_cfg = cfg.get("dataset")
    if isinstance(dataset_cfg, Mapping):
        name = dataset_cfg.get("name")
        if isinstance(name, str):
            return name.lower()
    return "omaps"


def run_dataset_probe(
    cfg: Mapping[str, Any],
    *,
    split: str,
    clips: int,
) -> Tuple[int, float]:
    """Iterate limited clips to trigger cache updates."""
    loader = make_dataloader(cfg, split=split)
    processed = 0
    start = time.perf_counter()
    for batch in loader:
        video = batch.get("video")
        paths = batch.get("path", [])
        batch_clips = int(video.shape[0]) if torch.is_tensor(video) else len(paths)
        processed += batch_clips
        logging.info(
            "Processed batch: clips=%d tensor_shape=%s first_path=%s",
            batch_clips,
            tuple(video.shape) if torch.is_tensor(video) else "n/a",
            paths[0] if paths else "n/a",
        )
        if processed >= clips:
            break
    elapsed = time.perf_counter() - start
    return processed, elapsed


def _baseline_matrix(source_hw: Iterable[int], target_hw: Iterable[int]) -> np.ndarray:
    src_vec = list(source_hw)
    dst_vec = list(target_hw)
    if len(src_vec) < 2 or len(dst_vec) < 2:
        raise ValueError("source and target shape descriptors must provide (H, W)")
    src_h, src_w = (int(src_vec[0]), int(src_vec[1]))
    dst_h, dst_w = (int(dst_vec[0]), int(dst_vec[1]))
    scale_x = (dst_w - 1) / max(src_w - 1, 1)
    scale_y = (dst_h - 1) / max(src_h - 1, 1)
    return np.array(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def inspect_cache(cache_path: Path, limit: int) -> None:
    if not cache_path.exists():
        logging.warning("Cache file missing: %s", cache_path)
        return

    try:
        payload = json.loads(cache_path.read_text())
    except Exception as exc:  # pragma: no cover - defensive
        logging.error("Failed to read cache %s (%s)", cache_path, exc)
        return

    if not isinstance(payload, Dict):
        logging.warning("Unexpected cache payload type: %s", type(payload).__name__)
        return

    entries = sorted(
        payload.items(),
        key=lambda kv: float(kv[1].get("timestamp", 0.0)) if isinstance(kv[1], Mapping) else 0.0,
        reverse=True,
    )
    if not entries:
        logging.info("Cache is empty at %s", cache_path)
        return

    logging.info("Inspecting %d cached homographies from %s", min(limit, len(entries)), cache_path)
    for video_id, data in entries[:limit]:
        if not isinstance(data, Mapping):
            continue
        H_vals = data.get("homography")
        src_hw = data.get("source_hw")
        dst_hw = data.get("target_hw")
        status = str(data.get("status", "unknown"))
        err_before = float(data.get("err_before", math.nan))
        err_after = float(data.get("err_after", math.nan))
        err_white = float(data.get("err_white_edges", err_after))
        err_black = float(data.get("err_black_gaps", math.nan))
        warp_ctrl = data.get("x_warp_ctrl")
        if isinstance(warp_ctrl, Sequence):
            ctrl_points = len(warp_ctrl)
        elif isinstance(warp_ctrl, Iterable):
            ctrl_points = sum(1 for _ in warp_ctrl)
        else:
            ctrl_points = 0
        frames = int(data.get("frames", 0))
        if not isinstance(H_vals, Iterable) or src_hw is None or dst_hw is None:
            logging.info("%s: status=%s frames=%d (missing homography)", video_id, status, frames)
            continue

        H = np.array(list(H_vals), dtype=np.float32).reshape(3, 3)
        baseline = _baseline_matrix(src_hw, dst_hw)
        delta_norm = float(np.linalg.norm(H - baseline))
        improvement = err_before - err_after
        logging.info(
            (
                "%s: status=%s frames=%d err_before=%.2fpx err_after=%.2fpx "
                "err_white=%.2fpx err_black=%.2fpx Δerr=%.2fpx ||ΔH||=%.3f warp_pts=%d"
            ),
            video_id,
            status,
            frames,
            err_before,
            err_after,
            err_white,
            err_black,
            improvement,
            delta_norm,
            ctrl_points,
        )


def build_argparser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Probe registration refinement and inspect cached homographies.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the experiment configuration YAML.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to probe (defaults to dataset.split_val or 'val').",
    )
    parser.add_argument(
        "--clips",
        type=int,
        default=2,
        help="Number of clips to process before stopping.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override frames per clip for the probe run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        dest="batch_size",
        help="Override batch size for the temporary dataloader.",
    )
    parser.add_argument(
        "--inspect",
        type=int,
        default=5,
        help="Number of cached homographies to summarise.",
    )
    parser.add_argument(
        "--cache",
        default=str(project_root / "reg_refined.json"),
        help="Path to the registration refinement cache.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    cfg_loaded = load_config(args.config)
    cfg = cast(MutableMapping[str, Any], copy.deepcopy(cfg_loaded))

    _apply_overrides(cfg, frames=args.frames, max_clips=args.clips, batch_size=args.batch_size)
    split = _resolve_split(cfg, args.split)

    dataset_name = _get_dataset_name(cfg)
    if dataset_name not in {"omaps", "pianoyt"}:
        logging.error("Unsupported dataset '%s' in configuration", dataset_name)
        return 2

    logging.info("Probing dataset '%s' split='%s' clips=%d", dataset_name, split, args.clips)
    processed, elapsed = run_dataset_probe(cfg, split=split, clips=args.clips)
    if processed == 0:
        logging.warning("No clips processed; check dataset configuration.")
    else:
        logging.info("Probe complete: processed=%d clips in %.2fs", processed, elapsed)

    project_root = Path(__file__).resolve().parents[2]
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = project_root / cache_path
    inspect_cache(cache_path, args.inspect)
    return 0


if __name__ == "__main__":
    sys.exit(main())
