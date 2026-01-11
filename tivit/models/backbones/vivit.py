"""ViViT-style backbone (GCR, registry-backed).

Purpose:
    - Provide the factorized ViViT/TiViT-Piano encoder with token-aligned tiling while preserving legacy behaviour and shapes.
    - Serve as the ``backend: vivit`` builder for the new registry/factory.
Key Functions/Classes:
    - ``TubeletEmbed`` / ``FactorizedSpaceTimeEncoder`` / ``TiViTPiano``: core modules for patchifying, encoding, and predicting multi-task logits.
    - ``build_model``: Instantiate the backbone from a nested config mapping.
CLI Arguments:
    (none)
Usage:
    from tivit.models.backbones.vivit import build_model
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tivit.data.roi import tile_vertical_token_aligned
from tivit.models.heads.common import MultiLayerHead

LOGGER = logging.getLogger(__name__)


class TubeletEmbed(nn.Module):
    """3D patchify with Conv3d to produce ViViT tubelets."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 768, patch_size: int = 16, tube_size: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_ch,
            embed_dim,
            kernel_size=(tube_size, patch_size, patch_size),
            stride=(tube_size, patch_size, patch_size),
            padding=0,
            bias=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[override]
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, D, T', H', W')
        x = rearrange(x, "b d t h w -> b (t h w) d")
        x = self.norm(x)
        return x, None


class TransformerBlock(nn.Module):
    """Standard encoder block used across the factorized encoder stages."""

    def __init__(self, d_model: int = 768, nhead: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.attn(x)


class FactorizedSpaceTimeEncoder(nn.Module):
    """
    Factorized attention like ViViT Model-3/2 with an extra global stage:
      - Temporal encoding within each (tile, spatial_position)
      - Spatial (across tiles) encoding for each (time, spatial_position)
      - Cross-tile aggregation via learnable global tokens
    """

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        depth_temporal: int = 2,
        depth_spatial: int = 2,
        depth_global: int = 1,
        global_tokens: int = 2,
        tiles: int = 3,
        t_tokens: Optional[int] = None,
        s_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tiles = tiles
        self.depth_temporal = depth_temporal
        self.depth_spatial = depth_spatial
        self.depth_global = depth_global
        self.global_tokens = global_tokens
        self.t_tokens = t_tokens
        self.s_tokens = s_tokens

        self.temporal_blocks = nn.ModuleList(
            TransformerBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(depth_temporal)
        )
        self.spatial_blocks = nn.ModuleList(
            TransformerBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(depth_spatial)
        )
        self.global_blocks = (
            nn.ModuleList(TransformerBlock(d_model, nhead, mlp_ratio, dropout) for _ in range(depth_global))
            if depth_global > 0
            else None
        )
        if global_tokens > 0:
            self.global_ctx = nn.Parameter(torch.randn(global_tokens, d_model))
        else:
            self.register_parameter("global_ctx", None)

    def forward(self, x_tiles: Sequence[torch.Tensor], t_tokens: int, s_tokens: int) -> torch.Tensor:
        """
        Args:
            x_tiles: list length=tiles, each (B, Ntokens, D) where Ntokens = t_tokens * s_tokens

        Returns:
            (B, tiles * Ntokens, D)
        """

        x_tiles_list = x_tiles if isinstance(x_tiles, (list, tuple)) else list(x_tiles)
        if not x_tiles_list:
            raise ValueError("x_tiles must be non-empty")
        b = x_tiles_list[0].shape[0]
        x = torch.stack(x_tiles_list, dim=1)  # (B, tiles, Ntokens, D)

        # Temporal attention per tile and spatial position.
        x_t = rearrange(x, "b m (t s) d -> (b m s) t d", t=t_tokens, s=s_tokens)
        for blk in self.temporal_blocks:
            x_t = blk(x_t)
        x = rearrange(x_t, "(b m s) t d -> b m (t s) d", b=b, m=self.tiles, s=s_tokens)

        # Spatial attention across tiles for each temporal/spatial position.
        x_s = rearrange(x, "b m n d -> (b n) m d")
        for blk in self.spatial_blocks:
            x_s = blk(x_s)
        x = rearrange(x_s, "(b n) m d -> b m n d", b=b)

        # Global aggregation with optional learnable tokens.
        x = rearrange(x, "b m n d -> b (m n) d")
        if self.depth_global > 0 and self.global_ctx is not None and self.global_blocks is not None:
            g = self.global_ctx.unsqueeze(0).expand(b, -1, -1)
            x = torch.cat([g, x], dim=1)
            for blk in self.global_blocks:
                x = blk(x)
            x = x[:, self.global_tokens :, :]
        return x


class TiViTPiano(nn.Module):
    """Factorized ViViT encoder with multi-task heads."""

    def __init__(
        self,
        tiles: int = 3,
        input_channels: int = 3,
        patch_size: int = 16,
        tube_size: int = 2,
        d_model: int = 768,
        nhead: int = 8,
        depth_temporal: int = 2,
        depth_spatial: int = 2,
        depth_global: int = 1,
        global_tokens: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pitch_classes: int = 88,
        clef_classes: int = 3,
        head_mode: str = "clip",
        tiling_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.tiles = tiles
        self.head_mode = head_mode

        tiling_cfg = dict(tiling_cfg or {})
        tokens_split_cfg = tiling_cfg.get("tokens_split", "auto")
        if isinstance(tokens_split_cfg, Sequence) and not isinstance(tokens_split_cfg, str):
            self.tiling_tokens_split: Union[str, List[int]] = [int(v) for v in tokens_split_cfg]
        else:
            self.tiling_tokens_split = tokens_split_cfg
        self.tiling_overlap_tokens = int(tiling_cfg.get("overlap_tokens", 0))
        self.tiling_patch_w = int(tiling_cfg.get("patch_w", patch_size))
        self._tile_aligned_width: Optional[int] = None
        self._tile_bounds: Optional[List[Tuple[int, int]]] = None
        self._tile_token_counts: Optional[List[int]] = None
        self._tiling_debug_enabled = False
        self._tiling_debug_logged = False

        self.embed = TubeletEmbed(
            in_ch=input_channels,
            embed_dim=d_model,
            patch_size=patch_size,
            tube_size=tube_size,
        )
        self.pos_drop = nn.Dropout(dropout)

        self.encoder: Optional[FactorizedSpaceTimeEncoder] = None
        self.encoder_cfg = dict(
            d_model=d_model,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            depth_temporal=depth_temporal,
            depth_spatial=depth_spatial,
            depth_global=depth_global,
            global_tokens=global_tokens,
            tiles=tiles,
        )

        base_hidden = max(d_model // 2, 128)
        self.head_pitch = MultiLayerHead(d_model, pitch_classes, hidden_dims=(base_hidden,), dropout=dropout)
        self.head_onset = MultiLayerHead(d_model, 88, hidden_dims=(base_hidden, base_hidden), dropout=dropout)
        self.head_offset = MultiLayerHead(d_model, 88, hidden_dims=(base_hidden, base_hidden), dropout=dropout)
        self.head_hand = MultiLayerHead(d_model, 2, hidden_dims=(base_hidden // 2,), dropout=dropout)
        self.head_clef = MultiLayerHead(d_model, clef_classes, hidden_dims=(base_hidden // 2,), dropout=dropout)

        self._tube_k = self.embed.proj.kernel_size[0]
        self._tube_s = self.embed.proj.stride[0]
        self._patch_kh = self.embed.proj.kernel_size[1]
        self._patch_kw = self.embed.proj.kernel_size[2]
        self._patch_sh = self.embed.proj.stride[1]
        self._patch_sw = self.embed.proj.stride[2]

    def _init_encoder_if_needed(self, t_tokens: int, s_tokens: int) -> None:
        """Lazy-init the encoder once token factors are known to save memory."""
        if self.encoder is None:
            self.encoder = FactorizedSpaceTimeEncoder(
                t_tokens=int(t_tokens),
                s_tokens=int(s_tokens),
                d_model=int(self.encoder_cfg["d_model"]),
                nhead=int(self.encoder_cfg["nhead"]),
                mlp_ratio=self.encoder_cfg["mlp_ratio"],
                dropout=self.encoder_cfg["dropout"],
                depth_temporal=int(self.encoder_cfg["depth_temporal"]),
                depth_spatial=int(self.encoder_cfg["depth_spatial"]),
                depth_global=int(self.encoder_cfg["depth_global"]),
                global_tokens=int(self.encoder_cfg["global_tokens"]),
                tiles=int(self.encoder_cfg["tiles"]),
            )

    def enable_tiling_debug(self) -> None:
        """Allow a single detailed tiling log when logger level is DEBUG."""

        self._tiling_debug_enabled = True
        self._tiling_debug_logged = False

    def _tile_split_token_aligned(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split ``x`` using token-aligned tiling consistent with the data pipeline."""

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

        if self._tile_token_counts:
            max_tokens = max(self._tile_token_counts)
            if any(tok != max_tokens for tok in self._tile_token_counts):
                patch_w = self.tiling_patch_w
                padded_tiles = []
                for tile, tok in zip(tiles, self._tile_token_counts):
                    extra_tokens = max_tokens - tok
                    if extra_tokens > 0:
                        pad_px = extra_tokens * patch_w
                        tile = F.pad(tile, (0, pad_px))
                    padded_tiles.append(tile)
                tiles = padded_tiles

        return tiles

    def _maybe_log_tiling_debug(self, tiles: Sequence[torch.Tensor]) -> None:
        """Emit a one-time tiling debug log without holding onto tensors."""
        if not self._tiling_debug_enabled or self._tiling_debug_logged:
            return
        if not tiles:
            return
        try:
            tile_shapes = [tuple(int(dim) for dim in tile.shape) for tile in tiles]
        except Exception:
            tile_shapes = []
        heights = [int(tile.shape[-2]) for tile in tiles]
        widths = [int(tile.shape[-1]) for tile in tiles]
        LOGGER.debug("[tiling-debug] tiles=%d tile_shapes=%s", len(tiles), tile_shapes or "n/a")
        LOGGER.debug("[tiling-debug] tile_heights=%s tile_widths=%s", [h for h in heights], [w for w in widths])
        token_counts = self._tile_token_counts
        if not token_counts:
            patch_w = max(int(self.tiling_patch_w), 1)
            token_counts = [int(w // patch_w) for w in widths]
        LOGGER.debug(
            "[tiling-debug] patch_columns_per_tile=%s (patch_w=%d)",
            token_counts,
            self.tiling_patch_w,
        )
        self._tiling_debug_logged = True

    def _infer_token_factors(self, t_cf: torch.Tensor) -> tuple[int, int]:
        """
        Infer the number of temporal tokens T' and spatial tokens S' (= H'*W')
        produced by the tubelet/patch embedding for a channels-first 5D tensor.
        """

        t_in = int(t_cf.shape[2])
        h_in = int(t_cf.shape[3])
        w_in = int(t_cf.shape[4])

        kernel_t, kernel_h, kernel_w = self.embed.proj.kernel_size
        kt, kh, kw = int(kernel_t), int(kernel_h), int(kernel_w)

        stride_t, stride_h, stride_w = self.embed.proj.stride
        st, sh, sw = int(stride_t), int(stride_h), int(stride_w)

        padding_attr = getattr(self.embed.proj, "padding", (0, 0, 0))
        if isinstance(padding_attr, (list, tuple)):
            pt_raw, ph_raw, pw_raw = padding_attr
        else:
            pt_raw = ph_raw = pw_raw = padding_attr
        pt, ph, pw = int(pt_raw), int(ph_raw), int(pw_raw)

        tprime = (t_in + 2 * pt - kt) // st + 1
        hprime = (h_in + 2 * ph - kh) // sh + 1
        wprime = (w_in + 2 * pw - kw) // sw + 1

        if tprime <= 0 or hprime <= 0 or wprime <= 0:
            raise ValueError(
                f"Invalid token sizing: got T'={tprime}, H'={hprime}, W'={wprime} "
                f"from input (T={t_in},H={h_in},W={w_in}) with "
                f"kernel={(kt, kh, kw)}, stride={(st, sh, sw)}, padding={(pt, ph, pw)}."
            )

        return int(tprime), int(hprime * wprime)

    def forward(self, x: torch.Tensor, return_per_tile: bool = False, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        x: (B, T, tiles, C, H, W)  OR  (B, T, C, H, W)
        Returns dict of logits; shapes depend on head_mode ("clip" or "frame").
        """
        if return_per_tile and getattr(self, "head_mode", "clip") != "frame":
            raise ValueError("return_per_tile=True is only supported when head_mode=='frame'")

        if x.dim() == 6:
            _, _, m, _, _, _ = x.shape
            tiles = [x[:, :, i, :, :, :] for i in range(m)]
        elif x.dim() == 5:
            tiles = self._tile_split_token_aligned(x)
        else:
            raise ValueError(f"Unexpected input rank {x.dim()} for TiViTPiano forward")
        self._maybe_log_tiling_debug(tiles)

        tile_tokens: List[torch.Tensor] = []
        t_tokens = s_tokens = None
        for t in tiles:
            t_cf = rearrange(t, "b tt c h w -> b c tt h w")
            tok, _ = self.embed(t_cf)
            if t_tokens is None or s_tokens is None:
                t_tokens, s_tokens = self._infer_token_factors(t_cf)
            tile_tokens.append(tok)

        if t_tokens is None or s_tokens is None:
            raise RuntimeError("Failed to infer token factors for encoder initialisation")
        self._init_encoder_if_needed(t_tokens=t_tokens, s_tokens=s_tokens)
        tile_tokens = [self.pos_drop(tok) for tok in tile_tokens]

        if self.encoder is None:
            raise RuntimeError("Encoder was not initialized properly.")
        enc = self.encoder(tile_tokens, t_tokens=t_tokens, s_tokens=s_tokens)

        enc_tiles = rearrange(enc, "b (m n) d -> b m n d", m=self.tiles)
        enc_5d = rearrange(enc_tiles, "b m (t s) d -> b m t s d", t=t_tokens)

        if getattr(self, "head_mode", "clip") == "frame":
            tile_feats = enc_5d.mean(dim=3)
            tile_feats = tile_feats.permute(0, 2, 1, 3).contiguous()
            pitch_tile = self.head_pitch(tile_feats)
            onset_tile = self.head_onset(tile_feats)
            offset_tile = self.head_offset(tile_feats)

            g_t = tile_feats.mean(dim=2)
            pitch_global = pitch_tile.mean(dim=2)
            onset_global = onset_tile.mean(dim=2)
            offset_global = offset_tile.mean(dim=2)
            hand_global = self.head_hand(g_t)
            clef_global = self.head_clef(g_t)

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
                outputs.update({"pitch_tile": pitch_tile, "onset_tile": onset_tile, "offset_tile": offset_tile})
            return outputs

        g = enc.mean(dim=1)
        return {
            "pitch_logits": self.head_pitch(g),
            "onset_logits": self.head_onset(g),
            "offset_logits": self.head_offset(g),
            "hand_logits": self.head_hand(g),
            "clef_logits": self.head_clef(g),
        }


def _get(d: Mapping[str, Any], key: str, default: Any) -> Any:
    """Fetch ``key`` from mapping with ``default``, mirroring legacy semantics."""
    v = d.get(key, default)
    return v if v is not None else default


def build_model(cfg: Mapping[str, Any]) -> TiViTPiano:
    """Instantiate the ViViT backbone from a nested config mapping."""

    if "model" not in cfg:
        raise ValueError("Config must have a model section")

    mcfg = cfg["model"]
    if "transformer" not in mcfg:
        raise ValueError("model.transformer is required for the ViViT backend")
    tcfg = mcfg["transformer"]
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, Mapping) else {}
    tiling_cfg = cfg.get("tiling", {}) if isinstance(cfg, Mapping) else {}
    tiles = int(_get(dataset_cfg, "tiles", 3))
    input_channels = int(_get(dataset_cfg, "channels", 3))

    return TiViTPiano(
        tiles=tiles,
        input_channels=input_channels,
        patch_size=_get(tcfg, "input_patch_size", 16),
        tube_size=_get(tcfg, "tube_size", 2),
        d_model=_get(tcfg, "d_model", 768),
        nhead=_get(tcfg, "num_heads", 8),
        depth_temporal=_get(tcfg, "depth_temporal", 2),
        depth_spatial=_get(tcfg, "depth_spatial", 2),
        depth_global=_get(tcfg, "depth_global", 1),
        global_tokens=_get(tcfg, "global_tokens", 2),
        mlp_ratio=_get(tcfg, "mlp_ratio", 4.0),
        dropout=_get(tcfg, "dropout", 0.1),
        head_mode=mcfg.get("head_mode", "clip"),
        tiling_cfg=tiling_cfg,
    )


__all__ = ["TiViTPiano", "TubeletEmbed", "FactorizedSpaceTimeEncoder", "build_model"]
