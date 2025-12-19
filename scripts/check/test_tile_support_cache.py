#!/usr/bin/env python3
"""
Lightweight checks for the TileSupportCache helper.

Run directly (no pytest needed):
    python scripts/check/test_tile_support_cache.py

Each check raises AssertionError on failure and prints a short summary on success.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import types
from importlib.util import module_from_spec, spec_from_file_location

try:  # optional torch import; tests still run when torch is absent
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


def _is_tensor(obj: object) -> bool:
    if not _HAS_TORCH or torch is None:
        return False
    try:
        return bool(torch.is_tensor(obj))
    except Exception:
        return False


def _ensure_lightweight_identifiers() -> None:
    """
    Provide a minimal ``utils.identifiers`` module without importing torch-heavy ``utils.__init__``.

    This keeps the check runnable even in bare environments.
    """

    if "utils.identifiers" in sys.modules:
        return
    module_path = REPO_ROOT / "src" / "utils" / "identifiers.py"
    spec = spec_from_file_location("utils.identifiers", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load identifiers helper from {module_path}")
    identifiers_mod = module_from_spec(spec)
    spec.loader.exec_module(identifiers_mod)  # type: ignore[arg-type]
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(module_path.parent)]
    sys.modules["utils"] = utils_pkg
    sys.modules["utils.identifiers"] = identifiers_mod
    setattr(utils_pkg, "identifiers", identifiers_mod)


def canonical_video_id(path_or_id: Optional[str | Path]) -> str:
    """Local copy of the repo helper to avoid importing torch-heavy modules."""

    if path_or_id is None:
        return ""
    text = str(path_or_id).strip()
    if not text:
        return ""

    name = Path(text).name
    base = name
    while True:
        stem = Path(base).stem
        if stem == base or not stem:
            break
        base = stem

    base = base.lower().replace("-", "_").replace(" ", "_").strip("_")
    while "__" in base:
        base = base.replace("__", "_")

    if base.startswith("audio_"):
        base = base[6:]
    elif base.startswith("audio"):
        base = base[5:].lstrip("_")

    if base.startswith("video_"):
        candidate = base[6:]
    elif base.startswith("video"):
        candidate = base[5:].lstrip("_")
    else:
        candidate = base

    candidate = candidate.strip("_")
    digits = "".join(ch for ch in candidate if ch.isdigit()) if candidate else ""
    if digits:
        return f"video_{digits}"
    if candidate:
        return f"video_{candidate}"
    return "video_unknown"


def _resolve_batch_clip_ids_for_test(batch: dict, batch_size: int) -> List[Optional[str]]:
    """Replica of the training helper focusing on stable IDs, without heavy deps."""

    id_fields = ("video_uid", "video_uids", "clip_id", "clip_ids", "video_id", "video_ids")
    for field in id_fields:
        value = batch.get(field)
        if value is None:
            continue
        if _is_tensor(value):
            flat = value.reshape(-1).tolist()
            if len(flat) >= batch_size:
                return [canonical_video_id(str(item)) for item in flat[:batch_size]]
        elif isinstance(value, (list, tuple)) and len(value) >= batch_size:
            return [canonical_video_id(str(item)) if item is not None else None for item in value[:batch_size]]

    paths = batch.get("path")
    clip_ids: List[Optional[str]] = []
    if isinstance(paths, (list, tuple)) and len(paths) >= batch_size:
        for idx in range(batch_size):
            path = paths[idx]
            try:
                stem = Path(str(path)).stem
            except Exception:
                stem = str(path)
            clip_ids.append(canonical_video_id(stem))
    else:
        clip_ids = [None for _ in range(batch_size)]
    return clip_ids


def _load_tile_support_cache():
    """Load TileSupportCache without importing the full tivit package (keeps deps light)."""

    _ensure_lightweight_identifiers()
    module_path = REPO_ROOT / "tivit" / "decoder" / "tile_support_cache.py"
    spec = spec_from_file_location("tile_support_cache_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tile_support_cache from {module_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


_tile_support_cache_mod = _load_tile_support_cache()
TileSupportCache = _tile_support_cache_mod.TileSupportCache
make_tile_cache_key = _tile_support_cache_mod.make_tile_cache_key


def check_train_scope_reuses_video_entry() -> None:
    cache = TileSupportCache()
    key = make_tile_cache_key("Video_001", num_tiles=3, cushion_keys=1, n_keys=88)
    calls = {"count": 0}

    def _compute() -> str:
        calls["count"] += 1
        return f"value-{calls['count']}"

    value_one, cached_one = cache.get_or_compute("train", key, _compute)
    value_two, cached_two = cache.get_or_compute("train", key, _compute)

    assert calls["count"] == 1, "compute_fn should run only once for the same video key"
    assert value_one == value_two
    assert cached_one is False
    assert cached_two is True


def check_eval_clear_preserves_shared_entries() -> None:
    cache = TileSupportCache()
    shared_key = make_tile_cache_key("video_shared", num_tiles=2, cushion_keys=0, n_keys=88)
    eval_key = make_tile_cache_key("video_eval", num_tiles=2, cushion_keys=0, n_keys=88)

    cache.put("train", shared_key, {"value": "shared"})
    cache.put("eval", eval_key, {"value": "eval"})

    assert cache.get("eval", shared_key)["value"] == "shared"
    assert cache.get("eval", eval_key)["value"] == "eval"

    counts_before = cache.counts()
    cleared = cache.clear_eval()
    counts_after = cache.counts()

    assert counts_before.eval == 1
    assert cleared == 1
    assert counts_after.eval == 0
    assert counts_after.shared == counts_before.shared
    assert cache.get("eval", shared_key)["value"] == "shared"


def check_cache_key_normalises_video_uid() -> None:
    lower = make_tile_cache_key("video_123", num_tiles=3, cushion_keys=0, n_keys=88)
    upper = make_tile_cache_key("VIDEO_123", num_tiles=3, cushion_keys=0, n_keys=88)
    assert lower == upper, "cache key should ignore case differences in video uid"


def check_clear_shared_and_all() -> None:
    cache = TileSupportCache()
    shared_key = make_tile_cache_key("video_shared", num_tiles=2, cushion_keys=0, n_keys=88)
    eval_key = make_tile_cache_key("video_eval", num_tiles=2, cushion_keys=0, n_keys=88)
    cache.put("train", shared_key, {"value": "shared"})
    cache.put("eval", eval_key, {"value": "eval"})

    cleared_shared = cache.clear_shared()
    counts_after_shared = cache.counts()
    assert cleared_shared == 1
    assert counts_after_shared.shared == 0
    assert counts_after_shared.eval == 1
    assert cache.get("eval", eval_key)["value"] == "eval"

    cleared_all = cache.clear_all()
    assert cleared_all.shared == 0
    assert cleared_all.eval == 1
    counts_after_all = cache.counts()
    assert counts_after_all.shared == 0 and counts_after_all.eval == 0


def check_batch_id_resolution_prefers_video_uid() -> None:
    batch = {
        "video_uid": ["video_101", "video_202"],
        "clip_id": ["video_101_clip_a", "video_202_clip_b"],
    }
    resolved: List[Optional[str]] = _resolve_batch_clip_ids_for_test(batch, batch_size=2)
    assert resolved == ["video_101", "video_202"], "video_uid should override clip_id when present"


def main() -> None:
    checks = [
        ("train scope reuse", check_train_scope_reuses_video_entry),
        ("eval clear preserves shared", check_eval_clear_preserves_shared_entries),
        ("cache key normalisation", check_cache_key_normalises_video_uid),
        ("clear shared/all", check_clear_shared_and_all),
        ("batch id resolution", check_batch_id_resolution_prefers_video_uid),
    ]
    for label, fn in checks:
        fn()
        print(f"[ok] {label}")
    print("All TileSupportCache checks passed.")


if __name__ == "__main__":
    main()
