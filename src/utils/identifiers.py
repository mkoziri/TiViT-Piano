"""Identifier normalisation helpers for dataset assets and caches."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union


def canonical_video_id(path_or_id: Union[str, Path]) -> str:
    """Return a canonical ``video_XXX`` style identifier for a sample.

    The helper accepts arbitrary strings or paths (including filenames such as
    ``video_220.0.mp4``) and normalises them so that cache keys, metadata, and
    label lookups consistently agree on the same identifier.  Normalisation
    steps:

    * drop any directory components and file extensions (including trailing
      ``.0`` shards)
    * lower-case and collapse non-alphanumeric separators to ``_``
    * coerce "audio" prefixes to "video"
    * ensure the result carries a ``video_`` prefix with the original numeric
      suffix preserved when present
    """

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

    base = base.lower()
    base = base.replace("-", "_").replace(" ", "_")
    base = re.sub(r"__+", "_", base).strip("_")

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

    match = re.search(r"(\d+)$", candidate)
    if match:
        digits = match.group(1)
        return f"video_{digits}"

    if candidate:
        return f"video_{candidate}"

    return "video_unknown"


__all__ = ["canonical_video_id"]
