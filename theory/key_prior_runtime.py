"""Purpose:
    Provide reusable helpers that parse configuration toggles for the
    key-aware prior and apply :class:`KeyAwarePrior` to per-head logits inside
    decoding/evaluation flows without duplicating the standalone CLI script.

Key Functions/Classes:
    - KeyPriorRuntimeSettings: Captures runtime configuration and optional
      overrides for FPS/MIDI ranges.
    - resolve_key_prior_settings(): Parse decoder.key_prior dictionaries into
      a strongly typed runtime settings object.
    - apply_key_prior_to_logits(): Run the prior over batched logits and return
      rescored tensors for the requested heads.

CLI:
    None.  Import these utilities from training/evaluation scripts that need
    to enable or disable the prior at runtime via ``configs/config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from .key_prior import KeyAwarePrior, KeyPriorConfig

HeadMap = Mapping[str, torch.Tensor]
HeadTensorMap = Dict[str, torch.Tensor]


@dataclass(slots=True)
class KeyPriorRuntimeSettings:
    """Configuration wrapper controlling when and how the key prior runs."""

    enabled: bool = False
    ref_head: str = "onset"
    apply_to: tuple[str, ...] = ("onset", "offset")
    window_sec: float = 3.0
    beta: float = 4.0
    rho_uniform: float = 0.8
    prior_strength: float = 0.5
    epsilon: float = 1e-8
    fps_override: float | None = None
    midi_low_override: int | None = None
    midi_high_override: int | None = None

    def resolve_config(self, *, fps: float, midi_low: int, midi_high: int) -> KeyPriorConfig:
        """Convert runtime settings + environment metadata into ``KeyPriorConfig``."""

        fps_eff = float(self.fps_override if self.fps_override is not None else fps)
        midi_low_eff = int(self.midi_low_override if self.midi_low_override is not None else midi_low)
        midi_high_eff = int(self.midi_high_override if self.midi_high_override is not None else midi_high)
        if fps_eff <= 0.0:
            raise ValueError("key prior requires fps>0; set decoder.key_prior.fps or dataset.decode_fps")
        if midi_high_eff < midi_low_eff:
            raise ValueError("key prior requires midi_high>=midi_low")
        return KeyPriorConfig(
            window_sec=self.window_sec,
            beta=self.beta,
            rho_uniform=self.rho_uniform,
            prior_strength=self.prior_strength,
            epsilon=self.epsilon,
            fps=fps_eff,
            midi_low=midi_low_eff,
            midi_high=midi_high_eff,
        )


def resolve_key_prior_settings(raw_cfg: Mapping[str, object] | None) -> KeyPriorRuntimeSettings:
    """Parse decoder.key_prior dictionaries into :class:`KeyPriorRuntimeSettings`."""

    if not isinstance(raw_cfg, Mapping):
        return KeyPriorRuntimeSettings()

    def _coerce_float(value: object, default: float | None = None) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_int(value: object, default: int | None = None) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    enabled = bool(raw_cfg.get("enabled", False))
    ref_head = str(raw_cfg.get("ref_head", "onset") or "onset")
    apply_to_raw = raw_cfg.get("apply_to")
    if isinstance(apply_to_raw, Sequence) and not isinstance(apply_to_raw, (str, bytes)):
        apply_to = tuple(
            str(item).strip()
            for item in apply_to_raw
            if str(item).strip()
        )
    else:
        apply_to = ("onset", "offset")

    return KeyPriorRuntimeSettings(
        enabled=enabled,
        ref_head=ref_head,
        apply_to=apply_to or (ref_head,),
        window_sec=_coerce_float(raw_cfg.get("window_sec"), 3.0) or 3.0,
        beta=_coerce_float(raw_cfg.get("beta"), 4.0) or 4.0,
        rho_uniform=_coerce_float(raw_cfg.get("rho_uniform"), 0.8) or 0.8,
        prior_strength=_coerce_float(raw_cfg.get("prior_strength"), 0.5) or 0.5,
        epsilon=_coerce_float(raw_cfg.get("epsilon"), 1e-8) or 1e-8,
        fps_override=_coerce_float(raw_cfg.get("fps")),
        midi_low_override=_coerce_int(raw_cfg.get("midi_low")),
        midi_high_override=_coerce_int(raw_cfg.get("midi_high")),
    )


def apply_key_prior_to_logits(
    logits_by_head: HeadMap,
    settings: KeyPriorRuntimeSettings,
    *,
    fps: float,
    midi_low: int | None,
    midi_high: int | None,
) -> HeadTensorMap:
    """Apply the key-aware prior to selected heads and return updated tensors."""

    if not settings.enabled:
        return {}

    ref_tensor = logits_by_head.get(settings.ref_head)
    if not torch.is_tensor(ref_tensor):
        raise ValueError(
            f"key prior reference head '{settings.ref_head}' not present in logits map."
        )

    apply_heads = tuple(head for head in settings.apply_to if torch.is_tensor(logits_by_head.get(head)))
    if not apply_heads:
        return {}

    def _as_batch_view(tensor: torch.Tensor, *, label: str) -> tuple[torch.Tensor, bool]:
        view = tensor.detach()
        squeezed = False
        if view.ndim == 2:
            view = view.unsqueeze(0)
            squeezed = True
        elif view.ndim != 3:
            raise ValueError(f"{label} logits must have shape (B,T,P) or (T,P); got {tuple(view.shape)}")
        return view.to(dtype=torch.float32, device="cpu", non_blocking=False).contiguous(), squeezed

    with torch.inference_mode():
        ref_view, ref_squeezed = _as_batch_view(ref_tensor, label=settings.ref_head)
        batch_size, _, num_pitches = ref_view.shape

        midi_low_eff = int(midi_low if midi_low is not None else 21)
        if midi_low_eff < 0:
            midi_low_eff = 0
        midi_high_eff = (
            settings.midi_high_override
            if settings.midi_high_override is not None
            else (midi_high if midi_high is not None else midi_low_eff + num_pitches - 1)
        )
        midi_high_eff = int(midi_high_eff)
        expected_pitches = midi_high_eff - midi_low_eff + 1
        if expected_pitches != num_pitches:
            raise ValueError(
                f"key prior pitch mismatch: logits have {num_pitches} bins but MIDI range "
                f"[{midi_low_eff}, {midi_high_eff}] implies {expected_pitches}."
            )
        fps_eff = fps if fps > 0 else 0.0
        config = settings.resolve_config(
            fps=fps_eff,
            midi_low=midi_low_eff,
            midi_high=midi_high_eff,
        )
        prior = KeyAwarePrior(config)

        head_views: dict[str, dict[str, object]] = {}
        for head in apply_heads:
            tensor = logits_by_head[head]
            view, squeezed = _as_batch_view(tensor, label=head)
            if view.shape != ref_view.shape:
                raise ValueError(
                    f"key prior expects head '{head}' to match reference shape {tuple(ref_view.shape)}; "
                    f"got {tuple(view.shape)}."
                )
            head_views[head] = {
                "view": view,
                "squeezed": squeezed,
                "device": tensor.device,
                "dtype": tensor.dtype,
            }

        updated: HeadTensorMap = {}
        for batch_idx in range(batch_size):
            ref_np = np.asarray(ref_view[batch_idx].numpy(), dtype=np.float64)
            key_posteriors, key_names = prior.estimate_key_posteriors(ref_np)
            pc_prior = prior.pc_prior_from_keys(key_posteriors, key_names)

            for head, meta in head_views.items():
                view = meta["view"]
                head_np = np.asarray(view[batch_idx].numpy(), dtype=np.float64)
                rescored = prior.rescore_logits_with_pc_prior(head_np, pc_prior)
                clip_tensor = torch.from_numpy(rescored.astype(np.float32))
                updated.setdefault(head, []).append(clip_tensor)

        for head, clips in updated.items():
            stacked = torch.stack(clips, dim=0)
            meta = head_views[head]
            if meta["squeezed"]:
                stacked = stacked.squeeze(0)
            stacked = stacked.to(
                device=meta["device"],
                dtype=meta["dtype"],
                non_blocking=False,
            )
            updated[head] = stacked

    return updated


__all__ = [
    "KeyPriorRuntimeSettings",
    "resolve_key_prior_settings",
    "apply_key_prior_to_logits",
]
