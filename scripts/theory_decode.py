"""Purpose:
    Apply a music-theory-informed key prior to logits stored in NPZ archives and
    persist the adjusted predictions together with updated metadata.

Key Functions/Classes:
    - load_npz_logits(): Loads logits arrays and metadata while validating the
      expected heads and shapes.
    - maybe_save_key_track(): Optionally writes a CSV summarizing the key
      posterior over time for inspection.
    - main(): Parses CLI arguments, builds the prior configuration, applies the
      prior to requested heads, and saves the results.

CLI:
    Example invocation::

        python scripts/theory_decode.py \
          --in_npz out/val_logits.npz \
          --out_npz out/val_logits_keyaware.npz \
          --apply_to onset,offset \
          --window_sec 3.0 --beta 4.0 --rho_uniform 0.8 --lambda_key 0.5 \
          --fps 30 --midi_low 21 --midi_high 108
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from theory.key_prior import KeyAwarePrior, KeyPriorConfig
else:
    try:
        from tivit.theory import KeyAwarePrior, KeyPriorConfig
    except ModuleNotFoundError:  # pragma: no cover - environment guard
        import sys

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
        from tivit.theory import KeyAwarePrior, KeyPriorConfig


LOGIT_HEADS: Tuple[str, ...] = ("onset", "pitch", "offset")
NPZ_SUFFIX = "_logits"


def parse_apply_to(arg: str | None) -> List[str]:
    """Parse a comma-separated list of heads to adjust."""

    if not arg:
        return []
    heads: List[str] = []
    for item in arg.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in LOGIT_HEADS:
            raise ValueError(
                f"Invalid head '{name}' in --apply_to. Expected a subset of {LOGIT_HEADS}."
            )
        if name not in heads:
            heads.append(name)
    return heads


def load_npz_logits(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Load logits and metadata from an NPZ file."""

    with np.load(path, allow_pickle=True) as data:
        arrays: Dict[str, np.ndarray] = {}
        for head in LOGIT_HEADS:
            key = f"{head}{NPZ_SUFFIX}"
            if key in data:
                arrays[head] = np.asarray(data[key])

        if not arrays:
            raise ValueError(
                "Input NPZ does not contain any of the expected logits arrays: "
                + ", ".join(f"{h}{NPZ_SUFFIX}" for h in LOGIT_HEADS)
            )

        meta_dict: Dict[str, object]
        if "meta" in data:
            meta_array = np.asarray(data["meta"]).ravel()
            if meta_array.size == 0:
                meta_str = ""
            elif meta_array.ndim == 0 or meta_array.size == 1:
                meta_str = str(meta_array.item())
            else:
                raise ValueError("meta array must contain a single JSON string.")
            try:
                meta_obj = json.loads(meta_str) if meta_str else {}
            except json.JSONDecodeError as exc:
                raise ValueError("Failed to decode meta JSON from NPZ.") from exc
            if not isinstance(meta_obj, dict):
                raise ValueError("Decoded meta JSON must be an object.")
            meta_dict = cast(Dict[str, object], meta_obj)
        else:
            meta_dict = {}

    return arrays, meta_dict


def validate_logits_shapes(arrays: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """Ensure all available logits share the same (T, P) shape."""

    shape: Tuple[int, int] | None = None
    for head, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(
                f"Logits array for head '{head}' must have shape (T, P); got {array.shape}."
            )
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                "All logits arrays must share the same shape; "
                f"mismatch between '{head}' ({array.shape}) and previous {shape}."
            )
    if shape is None:
        raise RuntimeError("No logits arrays were loaded.")
    return shape


def select_reference_head(
    requested: str | None, arrays: Dict[str, np.ndarray]
) -> str:
    """Determine which head to use for estimating key posteriors."""

    if requested:
        if requested not in LOGIT_HEADS:
            raise ValueError(
                f"Invalid --ref_head '{requested}'. Expected one of {LOGIT_HEADS}."
            )
        if requested not in arrays:
            raise ValueError(
                f"Reference head '{requested}' not found in input NPZ. Available: {list(arrays)}"
            )
        return requested

    for candidate in ("onset", "pitch", "offset"):
        if candidate in arrays:
            return candidate
    raise RuntimeError("Unable to select a reference head; no logits available.")


def build_key_prior_config(
    args: argparse.Namespace,
    fps: float,
    midi_low: int,
    midi_high: int,
) -> KeyPriorConfig:
    """Create a :class:`KeyPriorConfig` from CLI arguments and metadata."""

    return KeyPriorConfig(
        window_sec=args.window_sec,
        beta=args.beta,
        rho_uniform=args.rho_uniform,
        prior_strength=args.lambda_key,
        fps=fps,
        midi_low=midi_low,
        midi_high=midi_high,
    )


def save_npz(
    path: Path,
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, object],
) -> None:
    """Write logits arrays and metadata to an NPZ file."""

    np_arrays: Dict[str, np.ndarray] = {
        f"{head}{NPZ_SUFFIX}": value for head, value in arrays.items()
    }
    meta_json = json.dumps(meta)
    np_arrays["meta"] = np.asarray(meta_json)
    np.savez_compressed(str(path), **cast(Dict[str, Any], np_arrays))


def maybe_save_key_track(
    path: Path | None,
    fps: float,
    posteriors: np.ndarray,
    key_names: List[str],
) -> None:
    """Optionally write a CSV summary of the key posterior."""

    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = posteriors.shape[0]
    times = (
        np.arange(num_frames, dtype=np.float64) / fps if num_frames else np.empty(0)
    )

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_sec", "top1_key", "top1_prob", "top2_key", "top2_prob"])
        if num_frames == 0:
            return

        sorted_indices = np.argsort(posteriors, axis=1)[:, ::-1]
        sorted_probs = np.take_along_axis(posteriors, sorted_indices, axis=1)

        for frame, time in enumerate(times):
            top1_idx = int(sorted_indices[frame, 0])
            if sorted_indices.shape[1] >= 2:
                top2_idx = int(sorted_indices[frame, 1])
                top2_prob = float(sorted_probs[frame, 1])
            else:
                top2_idx = top1_idx
                top2_prob = float(sorted_probs[frame, 0])

            writer.writerow(
                [
                    float(time),
                    key_names[top1_idx],
                    float(sorted_probs[frame, 0]),
                    key_names[top2_idx],
                    top2_prob,
                ]
            )

    print(f"[theory_decode] wrote key posterior track -> {path}")


def maybe_plot_key_posteriors(
    enabled: bool, fps: float, posteriors: np.ndarray, key_names: List[str]
) -> None:
    """Display a heatmap of the key posterior when requested."""

    if not enabled:
        return

    if posteriors.size == 0:
        print("[theory_decode] skipping plot: no frames available.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for --plot") from exc

    num_frames = posteriors.shape[0]
    duration = num_frames / fps if fps > 0 else float(num_frames)
    extent = (0.0, duration, -0.5, len(key_names) - 0.5)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        posteriors.T,
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    ax.set_xlabel("Time (s)" if fps > 0 else "Frame")
    ax.set_ylabel("Key")
    ax.set_yticks(np.arange(len(key_names)))
    ax.set_yticklabels(key_names)
    fig.colorbar(im, ax=ax, label="Posterior")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a key-aware pitch-class prior to saved logits."
    )
    parser.add_argument("--in_npz", required=True, type=Path, help="Input NPZ file path.")
    parser.add_argument(
        "--out_npz", required=True, type=Path, help="Path to write the rescored NPZ."
    )
    parser.add_argument(
        "--apply_to",
        default="onset",
        help="Comma-separated list of heads to rescore (subset of onset, offset, pitch).",
    )
    parser.add_argument("--window_sec", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--rho_uniform", type=float, default=0.8)
    parser.add_argument("--lambda_key", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=None, help="Override frame rate (Hz).")
    parser.add_argument(
        "--midi_low", type=int, default=None, help="Lowest MIDI pitch index included."
    )
    parser.add_argument(
        "--midi_high", type=int, default=None, help="Highest MIDI pitch index included."
    )
    parser.add_argument(
        "--ref_head",
        type=str,
        default=None,
        help="Reference head to estimate the key posterior (onset, pitch, or offset).",
    )
    parser.add_argument(
        "--save_key_track",
        type=Path,
        default=None,
        help="Optional CSV path to save per-frame key posterior summaries.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display a matplotlib heatmap of the key posterior over time.",
    )
    args = parser.parse_args()

    arrays, meta = load_npz_logits(args.in_npz)
    _, num_pitches = validate_logits_shapes(arrays)

    # Resolve configuration parameters using CLI overrides first, then metadata, then defaults.
    fps_override = args.fps
    if fps_override is not None:
        fps = float(fps_override)
    else:
        meta_fps_obj = meta.get("fps")
        fps = float(meta_fps_obj) if isinstance(meta_fps_obj, (int, float)) else 30.0

    midi_low_override = args.midi_low
    if midi_low_override is not None:
        midi_low = int(midi_low_override)
    else:
        meta_midi_low = meta.get("midi_low")
        midi_low = int(meta_midi_low) if isinstance(meta_midi_low, int) else 21

    midi_high_override = args.midi_high
    if midi_high_override is not None:
        midi_high = int(midi_high_override)
    else:
        meta_midi_high = meta.get("midi_high")
        midi_high = int(meta_midi_high) if isinstance(meta_midi_high, int) else 108

    expected_pitches = midi_high - midi_low + 1
    if expected_pitches <= 0:
        raise ValueError("midi_high must be greater than or equal to midi_low.")
    if num_pitches != expected_pitches:
        raise ValueError(
            "Pitch dimension of logits does not match MIDI range."
            f" Expected {expected_pitches} pitches for [{midi_low}, {midi_high}],"
            f" but found {num_pitches}."
        )

    apply_heads = parse_apply_to(args.apply_to)
    ref_head = select_reference_head(args.ref_head, arrays)

    config = build_key_prior_config(args, fps=fps, midi_low=midi_low, midi_high=midi_high)
    prior = KeyAwarePrior(config)

    ref_logits = arrays[ref_head]
    P_keys, key_names = prior.estimate_key_posteriors(ref_logits)
    Pc_t = prior.pc_prior_from_keys(P_keys, key_names)
    maybe_save_key_track(args.save_key_track, fps, P_keys, key_names)
    maybe_plot_key_posteriors(args.plot, fps, P_keys, key_names)
    
    applied_heads: List[str] = []
    for head in apply_heads:
        if head not in arrays:
            continue
        original = arrays[head]
        rescored = prior.rescore_logits_with_pc_prior(original, Pc_t)
        arrays[head] = rescored.astype(original.dtype, copy=False)
        applied_heads.append(head)

    # Ensure metadata carries through defaults and the key prior configuration.
    meta = {**meta}
    meta["fps"] = fps
    meta["midi_low"] = midi_low
    meta["midi_high"] = midi_high
    meta["key_prior"] = {
        "window_sec": args.window_sec,
        "beta": args.beta,
        "rho_uniform": args.rho_uniform,
        "lambda_key": args.lambda_key,
        "fps": fps,
        "midi_low": midi_low,
        "midi_high": midi_high,
        "apply_to": applied_heads,
        "ref_head": ref_head,
    }

    save_npz(args.out_npz, arrays, meta)
    print(f"[theory_decode] wrote rescored logits -> {args.out_npz}")


if __name__ == "__main__":
    main()
