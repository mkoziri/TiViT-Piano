"""Post-processing utilities for decoding.

Purpose:
    - Expose decoding helpers, thresholding utilities, and optional constraints.
    - Provide a single import surface for post-processing components.

Key Functions/Classes:
    - build_decoder(), build_decoder_from_logits(): Hysteresis decoding helpers.
    - build_threshold_mask(), median_filter_time(): Thresholding utilities.
    - KeySignaturePrior: Decode-time key-prior adapter.

CLI Arguments:
    (none)

Usage:
    from tivit.postproc import build_decoder_from_logits
    decoder = build_decoder_from_logits(cfg)
"""

from .thresholding import build_threshold_mask, median_filter_time, pool_roll_BT, topk_mask
from .event_decode import (
    DECODER_DEFAULTS,
    decode_hysteresis,
    normalize_decoder_params,
    resolve_decoder_from_config,
    resolve_decoder_gates,
    build_decoder,
    build_decoder_from_logits,
)
from .key_signature import KeySignaturePrior, build_prior as build_key_signature
from .min_note_length import build_constraint as build_min_note_length
from .harmony_filter import build_constraint as build_harmony_filter

__all__ = [
    "build_threshold_mask",
    "median_filter_time",
    "pool_roll_BT",
    "topk_mask",
    "DECODER_DEFAULTS",
    "decode_hysteresis",
    "normalize_decoder_params",
    "resolve_decoder_from_config",
    "resolve_decoder_gates",
    "build_decoder",
    "build_decoder_from_logits",
    "KeySignaturePrior",
    "build_key_signature",
    "build_min_note_length",
    "build_harmony_filter",
]
