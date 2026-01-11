"""ViT-S/DeiT-S tile backend (GCR, registry-backed).

Purpose:
    - Expose the tiled ViT-S/DeiT-S backbone with temporal and cross-tile mixing while matching legacy behaviour and shapes.
    - Serve as the ``backend: vits_tile``/``backend: vit_small`` builder in the new registry/factory.
Key Functions/Classes:
    - ``SharedTileViTEncoder`` / ``TileTemporalEncoder`` / ``CrossTileMixer`` / ``ViTSTilePiano``: shared ViT feature extractor, temporal transformer, tile mixer, and multi-task heads.
    - ``build_model``: Instantiate the backend from a nested config mapping.
CLI Arguments:
    (none)
Usage:
    from tivit.models.backbones.vit_small import build_model
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tivit.data.roi import tile_vertical_token_aligned
from tivit.models.heads.common import MultiLayerHead

LOGGER = logging.getLogger(__name__)


class SharedTileViTEncoder(nn.Module):
    """Shared 2D ViT encoder that is reused across tiles and frames."""

    def __init__(
        self,
        backbone_name: str,
        *,
        input_hw: Sequence[int],
        embed_dim: int,
        input_channels: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "timm is required for the ViT-S tile backend; install the training requirements."
            ) from exc

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=bool(pretrained),
            num_classes=0,
            in_chans=int(input_channels),
            global_pool="",
        )
        resolved_dim = getattr(self.backbone, "num_features", None)
        if resolved_dim is None:
            resolved_dim = getattr(self.backbone, "embed_dim", None)
        if resolved_dim is None:
            resolved_dim = int(embed_dim)
        if int(embed_dim) != int(resolved_dim):
            raise ValueError(
                f"Configured embed_dim={embed_dim} does not match backbone ({resolved_dim}). "
                "Update model.vits_tile.embed_dim to the ViT backbone's width."
            )
        self.embed_dim = int(resolved_dim)
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> SharedTileViTEncoder:  # type: ignore[override]
        """Keep the frozen backbone in eval mode when training the head stack."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            tiles: Tensor shaped (B, T, tiles, C, H, W)

        Returns:
            Tensor of embeddings shaped (B, T, tiles, embed_dim)
        """
        if tiles.dim() != 6:
            raise ValueError(f"Expected tiled tensor rank 6, got {tiles.dim()}")
        b, t, m, c, h, w = tiles.shape
        frames = rearrange(tiles, "b tt tile c hh ww -> (b tt tile) c hh ww").contiguous()
        if frames.shape[-2:] != self.input_hw:
            frames = F.interpolate(frames, size=self.input_hw, mode="bilinear", align_corners=False)
        if frames.dtype != torch.float32:
            frames = frames.float()
        feat_fn = getattr(self.backbone, "forward_features", None)
        if callable(feat_fn):
            feats = cast(torch.Tensor, feat_fn(frames))
        else:  # pragma: no cover - timm backends expose forward_features
            feats = cast(torch.Tensor, self.backbone(frames))
        if feats.dim() == 3 and feats.shape[1] > 1:
            feats = feats[:, 0]
        if feats.dim() != 2:
            raise ValueError(
                f"Unexpected backbone output shape {tuple(feats.shape)}; expected (N, D) or (N, tokens, D)."
            )
        feats = feats.view(b, t, m, self.embed_dim)
        return feats


class TileTemporalEncoder(nn.Module):
    """Temporal transformer stacked per tile."""

    def __init__(self, embed_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                batch_first=True,
                dropout=dropout,
                dim_feedforward=embed_dim * 4,
                activation="gelu",
            )
            for _ in range(max(0, layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not self.layers:
            return x
        b, t, m, d = x.shape
        seq = rearrange(x, "b tt tile d -> (b tile) tt d")
        for layer in self.layers:
            seq = layer(seq)
        return rearrange(seq, "(b tile) tt d -> b tt tile d", b=b, tile=m)


class CrossTileMixer(nn.Module):
    """Mix tile embeddings per time step and emit global features."""

    def __init__(self, embed_dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=heads,
                batch_first=True,
                dropout=dropout,
                dim_feedforward=embed_dim * 4,
                activation="gelu",
            )
            for _ in range(max(0, layers))
        )
        self.post_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        Args:
            x: Tensor shaped (B, T, tiles, D)

        Returns:
            global_embeddings: (B, T, D)
            per_tile_embeddings: (B, T, tiles, D)
        """
        b, t, m, _ = x.shape
        if self.layers:
            seq = rearrange(x, "b tt tile d -> (b tt) tile d")
            for layer in self.layers:
                seq = layer(seq)
            x = rearrange(seq, "(b tt) tile d -> b tt tile d", b=b, tt=t)
        x = self.post_norm(x)
        return x.mean(dim=2), x


class ViTSTilePiano(nn.Module):
    """Drop-in replacement for TiViTPiano powered by a ViT-S tile encoder."""

    def __init__(
        self,
        *,
        tiles: int,
        input_channels: int,
        head_mode: str,
        tiling_cfg: Optional[Dict[str, Any]],
        backbone_name: str,
        embed_dim: int,
        pretrained: bool,
        freeze_backbone: bool,
        input_hw: Sequence[int],
        temporal_layers: int,
        temporal_heads: int,
        global_mixing_layers: int,
        dropout: float,
        pitch_classes: int = 88,
        clef_classes: int = 3,
    ) -> None:
        super().__init__()
        self.tiles = int(tiles)
        self.head_mode = str(head_mode or "frame").lower()
        tiling_cfg = dict(tiling_cfg or {})
        tokens_split_cfg = tiling_cfg.get("tokens_split", "auto")
        if isinstance(tokens_split_cfg, Sequence) and not isinstance(tokens_split_cfg, str):
            self.tiling_tokens_split: Sequence[int] | str = [int(v) for v in tokens_split_cfg]
        else:
            self.tiling_tokens_split = tokens_split_cfg
        self.tiling_overlap_tokens = int(tiling_cfg.get("overlap_tokens", 0))
        self.tiling_patch_w = int(tiling_cfg.get("patch_w", 16))
        self._tile_aligned_width: Optional[int] = None
        self._tile_bounds: Optional[List[Tuple[int, int]]] = None
        self._tile_token_counts: Optional[List[int]] = None
        self._tiling_debug_enabled = False
        self._tiling_debug_logged = False

        self.embed_dropout = nn.Dropout(dropout)
        self.tile_encoder = SharedTileViTEncoder(
            backbone_name=backbone_name,
            input_hw=input_hw,
            embed_dim=embed_dim,
            input_channels=input_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.temporal_encoder = TileTemporalEncoder(embed_dim, temporal_layers, temporal_heads, dropout)
        self.tile_mixer = CrossTileMixer(embed_dim, global_mixing_layers, temporal_heads, dropout)

        base_hidden = max(embed_dim // 2, 128)
        self.head_pitch = MultiLayerHead(embed_dim, pitch_classes, hidden_dims=(base_hidden,), dropout=dropout)
        self.head_onset = MultiLayerHead(embed_dim, 88, hidden_dims=(base_hidden, base_hidden), dropout=dropout)
        self.head_offset = MultiLayerHead(embed_dim, 88, hidden_dims=(base_hidden, base_hidden), dropout=dropout)
        self.head_hand = MultiLayerHead(embed_dim, 2, hidden_dims=(base_hidden // 2,), dropout=dropout)
        self.head_clef = MultiLayerHead(embed_dim, clef_classes, hidden_dims=(base_hidden // 2,), dropout=dropout)

    def enable_tiling_debug(self) -> None:
        """Enable a single round of tiling debug logs to aid troubleshooting."""
        self._tiling_debug_enabled = True
        self._tiling_debug_logged = False

    def _tile_split_token_aligned(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split and optionally pad tiles on the token grid to keep widths aligned."""
        b, t, c, h, w = x.shape
        cache_width = self._tile_aligned_width
        bounds = self._tile_bounds
        if cache_width is None or bounds is None or cache_width != w:
            sample = x[0]
            _, tokens_per_tile, _, bounds, aligned_w, _ = tile_vertical_token_aligned(
                sample,
                self.tiles,
                patch_w=self.tiling_patch_w,
                tokens_split=self.tiling_tokens_split,
                overlap_tokens=self.tiling_overlap_tokens,
            )
            cache_width = aligned_w
            self._tile_aligned_width = cache_width
            self._tile_bounds = bounds
            self._tile_token_counts = tokens_per_tile
        if cache_width is None or bounds is None:
            raise RuntimeError("Token-aligned tiling failed to compute bounds")
        if w != cache_width:
            x = x[..., :cache_width]
        tiles = [x[..., left:right] for (left, right) in bounds]
        token_counts = self._tile_token_counts or []
        if token_counts:
            max_tokens = max(token_counts)
            if any(tok != max_tokens for tok in token_counts):
                padded = []
                for tile, tok in zip(tiles, token_counts):
                    extra = max_tokens - tok
                    if extra > 0:
                        pad_px = extra * self.tiling_patch_w
                        tile = F.pad(tile, (0, pad_px))
                    padded.append(tile)
                tiles = padded
        return tiles

    def _maybe_log_tiling_debug(self, tiles: Sequence[torch.Tensor]) -> None:
        """Log tiling details once without holding references to the tensors."""
        if not self._tiling_debug_enabled or self._tiling_debug_logged:
            return
        if not tiles:
            return
        shapes = [tuple(int(dim) for dim in tile.shape) for tile in tiles]
        LOGGER.debug("[vits-tile] tiles=%d shapes=%s", len(tiles), shapes)
        widths = [int(tile.shape[-1]) for tile in tiles]
        heights = [int(tile.shape[-2]) for tile in tiles]
        LOGGER.debug("[vits-tile] tile_heights=%s tile_widths=%s", heights, widths)
        token_counts = self._tile_token_counts
        if not token_counts:
            token_counts = [int(w // max(self.tiling_patch_w, 1)) for w in widths]
        LOGGER.debug("[vits-tile] patch_columns_per_tile=%s (patch_w=%d)", token_counts, self.tiling_patch_w)
        self._tiling_debug_logged = True

    def forward(self, x: torch.Tensor, return_per_tile: bool = False, **kwargs: Any) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        if return_per_tile and self.head_mode != "frame":
            raise ValueError("return_per_tile=True is only supported when head_mode=='frame'")

        if x.dim() == 6:
            tiles = [x[:, :, i, :, :, :] for i in range(x.shape[2])]
        elif x.dim() == 5:
            tiles = self._tile_split_token_aligned(x)
        else:
            raise ValueError(f"Unexpected input rank {x.dim()} for ViTSTilePiano forward")
        self._maybe_log_tiling_debug(tiles)
        if len(tiles) != self.tiles:
            raise ValueError(f"Configured tiles={self.tiles} but received {len(tiles)} slices")
        tile_stack = torch.stack(tiles, dim=2)

        tile_embeddings = self.tile_encoder(tile_stack)
        tile_embeddings = self.embed_dropout(tile_embeddings)
        tile_embeddings = self.temporal_encoder(tile_embeddings)
        global_feats, tile_feats = self.tile_mixer(tile_embeddings)
        tile_feats = tile_feats.contiguous()

        if self.head_mode == "frame":
            pitch_tile = self.head_pitch(tile_feats)
            onset_tile = self.head_onset(tile_feats)
            offset_tile = self.head_offset(tile_feats)
            pitch_global = pitch_tile.mean(dim=2)
            onset_global = onset_tile.mean(dim=2)
            offset_global = offset_tile.mean(dim=2)
            hand_global = self.head_hand(global_feats)
            clef_global = self.head_clef(global_feats)

            outputs: Dict[str, torch.Tensor] = {
                "pitch_logits": pitch_global,
                "onset_logits": onset_global,
                "offset_logits": offset_global,
                "hand_logits": hand_global,
                "clef_logits": clef_global,
                "pitch_global": pitch_global,
                "onset_global": onset_global,
                "offset_global": offset_global,
                "hand_global": hand_global,
                "clef_global": clef_global,
            }
            if return_per_tile:
                outputs.update(
                    {
                        "pitch_tile": pitch_tile,
                        "onset_tile": onset_tile,
                        "offset_tile": offset_tile,
                    }
                )
            return outputs

        clip_global = global_feats.mean(dim=1)
        return {
            "pitch_logits": self.head_pitch(clip_global),
            "onset_logits": self.head_onset(clip_global),
            "offset_logits": self.head_offset(clip_global),
            "hand_logits": self.head_hand(clip_global),
            "clef_logits": self.head_clef(clip_global),
        }


def _get(d: Mapping[str, Any], key: str, default: Any) -> Any:
    """Fetch ``key`` from mapping with ``default``, mirroring legacy semantics."""
    v = d.get(key, default)
    return v if v is not None else default


def build_model(cfg: Mapping[str, Any]) -> ViTSTilePiano:
    """Instantiate the ViT-S tile backend from the provided config."""

    if "model" not in cfg:
        raise ValueError("Config must have a model section")
    mcfg = cfg["model"]
    vcfg = mcfg.get("vits_tile", {}) or {}
    tcfg = mcfg.get("transformer", {}) or {}
    input_hw = vcfg.get("input_hw", (145, 342))
    if not isinstance(input_hw, Sequence):
        raise ValueError("model.vits_tile.input_hw must be a 2-element sequence")
    if len(input_hw) < 2:
        raise ValueError("model.vits_tile.input_hw must provide height and width")
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    tiling_cfg = cfg.get("tiling", {}) if isinstance(cfg, Mapping) else {}
    dropout_default = _get(tcfg, "dropout", 0.1)
    tiles = int(_get(dataset_cfg, "tiles", 3))
    input_channels = int(_get(dataset_cfg, "channels", 3))

    return ViTSTilePiano(
        tiles=tiles,
        input_channels=input_channels,
        head_mode=mcfg.get("head_mode", "frame"),
        tiling_cfg=tiling_cfg,
        backbone_name=str(vcfg.get("backbone_name", "vit_small_patch16_224")),
        embed_dim=int(vcfg.get("embed_dim", 384)),
        pretrained=bool(vcfg.get("pretrained", True)),
        freeze_backbone=bool(vcfg.get("freeze_backbone", True)),
        input_hw=input_hw,
        temporal_layers=int(vcfg.get("temporal_layers", 2)),
        temporal_heads=int(vcfg.get("temporal_heads", 4)),
        global_mixing_layers=int(vcfg.get("global_mixing_layers", 1)),
        dropout=float(vcfg.get("dropout", dropout_default)),
    )


__all__ = [
    "ViTSTilePiano",
    "SharedTileViTEncoder",
    "TileTemporalEncoder",
    "CrossTileMixer",
    "build_model",
]
