"""Purpose:
    Define the TiViT-Piano neural network architecture including tubelet
    embedding, factorized space-time attention, and multi-task prediction
    heads.

Key Functions/Classes:
    - TubeletEmbed: Performs 3D convolutional patchification over temporal
      windows to produce token sequences per tile.
    - FactorizedSpaceTimeEncoder: Applies temporal, spatial, and global
      transformer blocks in a ViViT-style factorization.
    - TiViTPiano: Wraps embedding, encoder, and head modules to produce clip or
      frame-level predictions for pitch, onset, offset, hand, and clef tasks.

CLI:
    Not a CLI module; instantiate :class:`TiViTPiano` via configuration
    factories for training or evaluation scripts.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.tiling import tile_vertical_token_aligned


LOGGER = logging.getLogger(__name__)


# -------- Tubelet embedding (3D patchify) ----------
class TubeletEmbed(nn.Module):
    """
    3D patchify: Conv3d with kernel=(t,p,p), stride=(t,p,p)
    Expects input per tile as: B C T H W  (channels-first)
    Produces tokens of dim D.
    """
    def __init__(self, in_ch=3, embed_dim=768, patch_size=16, tube_size=2):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim,
                              kernel_size=(tube_size, patch_size, patch_size),
                              stride=(tube_size, patch_size, patch_size),
                              padding=0, bias=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: B C T H W
        x = self.proj(x)   # B D T' H' W'
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        x = self.norm(x)
        return x, None  # (tokens, pos_embed placeholder if needed)

# -------- Factorized ViViT-style encoder blocks ----------
class TransformerBlock(nn.Module):
    """Standard encoder block."""
    def __init__(self, d_model=768, nhead=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=int(d_model*mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu"
        )

    def forward(self, x):  # x: (B, N, D)
        return self.attn(x)

class FactorizedSpaceTimeEncoder(nn.Module):
    """
    Factorized attention like ViViT Model-3/2 with an extra global stage:
    - Temporal encoding within each (tile, spatial_position)
    - Spatial (across tiles) encoding for each (time, spatial_position)
    - Cross-tile aggregation via learnable global tokens
    Shapes:
      Input tokens come per-tile from TubeletEmbed:
        per tile: X_tile -> (B, Ntokens, D) with Ntokens = T' * H' * W'
      We also need to know the factorization counts: T', S = H'*W'
    """
    def __init__(self, d_model=768, nhead=8, mlp_ratio=4.0, dropout=0.1,
                 depth_temporal=2, depth_spatial=2, depth_global=1,
                 global_tokens=2,
                 tiles=3, t_tokens=None, s_tokens=None):
        super().__init__()
        self.tiles = tiles
        self.depth_temporal = depth_temporal
        self.depth_spatial = depth_spatial
        self.depth_global = depth_global
        self.global_tokens = global_tokens
        self.t_tokens = t_tokens   # number of temporal tokens after tubeleting
        self.s_tokens = s_tokens   # number of spatial tokens (H'*W')

        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, mlp_ratio, dropout)
            for _ in range(depth_temporal)
        ])
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, mlp_ratio, dropout)
            for _ in range(depth_spatial)
        ])

    # Global aggregation blocks and context tokens
        self.global_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, mlp_ratio, dropout)
            for _ in range(depth_global)
        ]) if depth_global > 0 else None
        if global_tokens > 0:
            self.global_ctx = nn.Parameter(torch.randn(global_tokens, d_model))
        else:
            self.register_parameter('global_ctx', None)
            
    def forward(self, x_tiles, t_tokens, s_tokens):
        """
        x_tiles: list length=tiles, each (B, Ntokens, D) where Ntokens = t_tokens * s_tokens
        Returns: (B, tiles * Ntokens, D)
        """
        B = x_tiles[0].shape[0]
        D = x_tiles[0].shape[-1]
        Tt, Ss = t_tokens, s_tokens

        # Stack tiles -> (B, tiles, Ntokens, D)
        x = torch.stack(x_tiles, dim=1)

        # ---- Temporal attention: per tile, per spatial position sequence over Tt ----
        # reshape to (B, tiles, Ss, Tt, D) then group (B*tiles*Ss, Tt, D)
        x_t = rearrange(x, 'b m (t s) d -> (b m s) t d', t=Tt, s=Ss)
        for blk in self.temporal_blocks:
            x_t = blk(x_t)  # (B*tiles*Ss, Tt, D)
        # back to (B, tiles, Ss, Tt, D) -> (B, tiles, Tt*Ss, D)
        x = rearrange(x_t, '(b m s) t d -> b m (t s) d', b=B, m=self.tiles, s=Ss)

        # ---- Spatial attention: across tiles, for each (timestep, spatial position) ----
        # Need sequences length=tiles. Reshape to (B, Tt*Ss, tiles, D) -> group (B*Tt*Ss, tiles, D)
        x_s = rearrange(x, 'b m n d -> b n m d')  # n = Tt*Ss
        x_s = rearrange(x_s, 'b n m d -> (b n) m d')
        for blk in self.spatial_blocks:
            x_s = blk(x_s)  # (B*Tt*Ss, tiles, D)
        # back to (B, n, m, d) then (B, m, n, d)
        x = rearrange(x_s, '(b n) m d -> b m n d', b=B)

        # ---- Cross-tile aggregation with global context tokens ----
        x = rearrange(x, 'b m n d -> b (m n) d')  # (B, tiles*Ntokens, D)
        if self.depth_global > 0 and self.global_ctx is not None and self.global_blocks is not None:
            g = self.global_ctx.unsqueeze(0).expand(B, -1, -1)  # (B, G, D)
            x = torch.cat([g, x], dim=1)  # prepend global tokens
            for blk in self.global_blocks:
                x = blk(x)
            x = x[:, self.global_tokens:, :]  # drop globals
        return x

# -------- Shared multi-layer prediction head ----------
class MultiLayerHead(nn.Module):
    """Light-weight MLP head with normalization, hidden layers, and dropout."""

    def __init__(
        self,
        d_model: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (512,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(d_model)]

        in_dim = d_model
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KeyAwareWidthPool(nn.Module):
    """
    Learnable soft assignment from spatial width bins to piano keys.

    Maintains a bank of key-by-width logits initialised from canonical keyboard
    geometry, row-wise softmaxed at runtime to yield per-key distributions.
    """

    def __init__(
        self,
        num_keys: int = 88,
        canonical_width: float = 800.0,
        smoothness_lambda: float = 5e-5,
    ) -> None:
        super().__init__()
        self.num_keys = int(num_keys)
        self.canonical_width = float(canonical_width)
        self.white_key_width = float(self.canonical_width / 52.0)
        self.sigma = max(self.white_key_width * 0.7, 1.0)
        self.smoothness_lambda = max(float(smoothness_lambda), 0.0)

        self.register_parameter("weight", None)
        self.register_buffer("key_centers", self._compute_key_centers(), persistent=False)
        self.register_buffer("width_coords_ref", None, persistent=False)

        self._logged = False

    def _compute_key_centers(self) -> torch.Tensor:
        """Return canonical x-centres (px) for MIDI 21..108."""

        black_mods = {1, 3, 6, 8, 10}
        centers: List[float] = []
        white_positions: Dict[int, float] = {}
        white_idx = -1

        for midi in range(21, 109):
            note_mod = midi % 12
            if note_mod in black_mods:
                prev_midi = midi - 1
                while prev_midi >= 21 and (prev_midi % 12) in black_mods:
                    prev_midi -= 1
                next_midi = midi + 1
                while next_midi <= 108 and (next_midi % 12) in black_mods:
                    next_midi += 1
                prev_center = white_positions.get(prev_midi)
                next_center = white_positions.get(next_midi)
                if prev_center is not None and next_center is not None:
                    center = 0.5 * (prev_center + next_center)
                elif prev_center is not None:
                    center = prev_center
                elif next_center is not None:
                    center = next_center
                else:
                    center = 0.0
                centers.append(float(center))
            else:
                white_idx += 1
                center = (white_idx + 0.5) * self.white_key_width
                white_positions[midi] = center
                centers.append(float(center))

        return torch.tensor(centers, dtype=torch.float32)

    def _init_weight(self, width_coords: torch.Tensor, device: torch.device) -> None:
        coords = width_coords.to(device=device, dtype=torch.float32)
        centers = self.key_centers.to(device=device, dtype=torch.float32)

        diff = coords.unsqueeze(0) - centers.unsqueeze(1)
        gauss = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        gauss = gauss / gauss.sum(dim=1, keepdim=True).clamp_min(1e-6)
        logits = torch.log(gauss.clamp_min(1e-6))

        self.weight = nn.Parameter(logits)
        self.width_coords_ref = coords.detach().clone()
        self._logged = False

    def _ensure_weight(self, width_coords: torch.Tensor) -> None:
        if width_coords.ndim != 1:
            raise ValueError(f"width_coords must be 1D (got shape {tuple(width_coords.shape)})")

        P = int(width_coords.numel())
        device = width_coords.device

        if self.weight is None or self.weight.shape[1] != P:
            self._init_weight(width_coords, device)
            LOGGER.info("keypool: P=%d init=geometry learnable=True", P)
            self._logged = True
        elif self.width_coords_ref is None or self.width_coords_ref.shape[0] != P:
            self.width_coords_ref = width_coords.detach().clone()
        elif not self._logged:
            LOGGER.info("keypool: P=%d init=geometry learnable=True", P)
            self._logged = True

    def forward(
        self,
        feat: torch.Tensor,
        width_coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            feat: (B, T, P, D) tensor of features across width bins.
            width_coords: (P,) canonical x coordinate per width bin.
            valid_mask: Optional (P,) bool mask flagging valid width bins.

        Returns:
            key_features: (B, T, K, D)
            smooth_loss: Optional scalar regulariser.
        """

        if feat.ndim != 4:
            raise ValueError(f"Expected feat with rank 4 (B,T,P,D), got shape {tuple(feat.shape)}")

        width_coords = width_coords.to(device=feat.device, dtype=torch.float32)
        self._ensure_weight(width_coords)

        logits = self.weight.to(device=feat.device)
        if logits.shape[1] != feat.shape[2]:
            raise RuntimeError(
                f"Key pool width mismatch: logits has {logits.shape[1]} bins but features have {feat.shape[2]}"
            )

        if valid_mask is not None:
            mask = valid_mask.to(device=feat.device, dtype=torch.bool).unsqueeze(0)
            masked_logits = logits.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(masked_logits, dim=1)
            attn = attn * mask.float()
            denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
            attn = attn / denom
        else:
            attn = torch.softmax(logits, dim=1)

        attn = attn.to(dtype=feat.dtype)
        key_feat = torch.einsum("btpd,kp->btkd", feat, attn)

        smooth_loss: Optional[torch.Tensor] = None
        if self.smoothness_lambda > 0.0 and attn.shape[1] > 1:
            diff = attn[:, 1:] - attn[:, :-1]
            smooth_loss = self.smoothness_lambda * diff.pow(2).mean()

        return key_feat, smooth_loss
    
# -------- TiViT-Piano: tiling wrapper + ViViT backbone + multi-task head ----------
class TiViTPiano(nn.Module):
    """
    Expected input from dataloader:
      - Either: x of shape (B, T, tiles, C, H, W)
      - Or:     x of shape (B, T, C, H, W)  (will auto-split horizontally into 3 tiles)

    Pipeline:
      - For each tile: TubeletEmbed (shared weights) -> tokens of dim D
      - Factorized space-time-cross encoder (temporal -> spatial -> global cross-tile)
      - Heads:
          * head_mode="clip": global mean-pool over all tokens -> one prediction per clip
          * head_mode="frame": average over tiles & spatial -> per-time features (B, T', D),
                               then time-distributed heads produce per-time logits

    Outputs (dict):
      clip mode:
        - pitch_logits:  (B, 88)
        - onset_logits:  (B, 88)
        - offset_logits: (B, 88)
        - hand_logits:   (B, 2)
        - clef_logits:   (B, 3)

      frame mode:
        - pitch_logits:  (B, T', 88)
        - onset_logits:  (B, T', 88)
        - offset_logits: (B, T', 88)
        - hand_logits:   (B, T', 2)
        - clef_logits:   (B, T', 3)
    """
    def __init__(
        self,
        tiles=3,
        input_channels=3,
        patch_size=16,
        tube_size=2,
        d_model=768,
        nhead=8,
        depth_temporal=2,
        depth_spatial=2,
        depth_global=1,
        global_tokens=2,
        mlp_ratio=4.0,
        dropout=0.1,
        pitch_classes=88,
        clef_classes=3,
        head_mode: str = "clip",   # "clip" or "frame"
        per_key_head: bool = False,
        tiling_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.tiles = tiles
        self.head_mode = head_mode  # runtime switch for clip vs frame heads
        self.per_key_head = bool(per_key_head)

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
        self._tile_widths_px: Optional[List[int]] = None

        # Tubelet / patch embed
        self.embed = TubeletEmbed(in_ch=input_channels, embed_dim=d_model,
                                  patch_size=patch_size, tube_size=tube_size)
        self.pos_drop = nn.Dropout(dropout)

        # Encoder will be initialized lazily when we infer t_tokens & s_tokens
        self.encoder = None
        self.encoder_cfg = dict(d_model=d_model, nhead=nhead,
                                mlp_ratio=mlp_ratio, dropout=dropout,
                                depth_temporal=depth_temporal,
                                depth_spatial=depth_spatial,
                                depth_global=depth_global,
                                global_tokens=global_tokens,
                                tiles=tiles)

        # Heads. These work with (B, D) and (B, T, D).
        base_hidden = max(d_model // 2, 128)

        self.num_keys = int(pitch_classes)

        head_out_dim = 1 if self.per_key_head else self.num_keys

        self.head_pitch = MultiLayerHead(d_model, head_out_dim, hidden_dims=(base_hidden,), dropout=dropout)
        self.head_onset = MultiLayerHead(d_model, head_out_dim, hidden_dims=(base_hidden, base_hidden), dropout=dropout)
        self.head_offset = MultiLayerHead(d_model, head_out_dim, hidden_dims=(base_hidden, base_hidden), dropout=dropout)
        self.head_hand = MultiLayerHead(d_model, 2, hidden_dims=(base_hidden//2,), dropout=dropout)
        self.head_clef = MultiLayerHead(d_model, clef_classes, hidden_dims=(base_hidden//2,), dropout=dropout)

        self.key_pool = (
            KeyAwareWidthPool(num_keys=self.num_keys, canonical_width=800.0, smoothness_lambda=5e-5)
            if self.per_key_head
            else None
        )

        # Cache Tubelet conv sizes for token shape inference
        self._tube_k   = self.embed.proj.kernel_size[0]
        self._tube_s   = self.embed.proj.stride[0]
        self._patch_kh = self.embed.proj.kernel_size[1]
        self._patch_kw = self.embed.proj.kernel_size[2]
        self._patch_sh = self.embed.proj.stride[1]
        self._patch_sw = self.embed.proj.stride[2]
        self._spatial_hw: Optional[Tuple[int, int]] = None
        self._keypool_warned = False

    def _init_encoder_if_needed(self, t_tokens, s_tokens):
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
                tiles=int(self.encoder_cfg["tiles"])
            )

    def _tile_split_token_aligned(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split ``x`` using token-aligned tiling consistent with the data pipeline."""

        B, T, C, H, W = x.shape
        cache_width = self._tile_aligned_width
        bounds = self._tile_bounds
        if cache_width is None or bounds is None or cache_width != W:
            sample = x[0]
            _, tokens_per_tile, widths_px, bounds, aligned_w, _ = tile_vertical_token_aligned(
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
            self._tile_widths_px = widths_px
        if cache_width is None or bounds is None:
            raise RuntimeError("Token-aligned tiling failed to compute bounds")
        if W != cache_width:
            x = x[..., :cache_width]
        tiles = [x[..., left:right] for (left, right) in bounds]

        # Pipeline v2 clips can yield token counts that differ by at most one
        # between tiles when the overall width is not perfectly divisible by the
        # number of tiles. This causes issues later when we stack the token
        # sequences, so we zero-pad narrower tiles to match the widest one. The
        # padding size is computed in token units to keep alignment with the ViT
        # patch grid.
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

    def _infer_token_factors(self, t_cf: torch.Tensor) -> tuple[int, int]:
        """
        Infer the number of temporal tokens T' and spatial tokens S' (= H'*W')
        produced by the tubelet/patch embedding for a channels-first 5D tensor.

        Args:
            t_cf: Tensor of shape (B, C, T, H, W)

        Returns:
            (t_tokens, s_tokens): (T', H'*W') as integers
        """
        # Input sizes
        T_in = int(t_cf.shape[2])
        H_in = int(t_cf.shape[3])
        W_in = int(t_cf.shape[4])

        # Conv3d (tubelet) parameters — assuming padding=0 (as in TubeletEmbed)
        kernel_t, kernel_h, kernel_w = self.embed.proj.kernel_size
        kt, kh, kw = int(kernel_t), int(kernel_h), int(kernel_w)

        stride_t, stride_h, stride_w = self.embed.proj.stride
        st, sh, sw = int(stride_t), int(stride_h), int(stride_w)
        # If you ever change TubeletEmbed padding, adjust here accordingly:
        if hasattr(self.embed.proj, "padding"):
            padding_attr = self.embed.proj.padding
        else:
            padding_attr = (0, 0, 0)

        if isinstance(padding_attr, (list, tuple)):
            pt_raw, ph_raw, pw_raw = padding_attr
        else:
            pt_raw = ph_raw = pw_raw = padding_attr

        pt, ph, pw = int(pt_raw), int(ph_raw), int(pw_raw)
        # Standard conv output-size arithmetic (floor division)
        # T' = floor((T + 2*pt - kt)/st) + 1, etc.
        Tprime = (T_in + 2 * pt - kt) // st + 1
        Hprime = (H_in + 2 * ph - kh) // sh + 1
        Wprime = (W_in + 2 * pw - kw) // sw + 1

        # Guard against negative/zero (shouldn’t happen with valid configs)
        if Tprime <= 0 or Hprime <= 0 or Wprime <= 0:
            raise ValueError(
                f"Invalid token sizing: got T'={Tprime}, H'={Hprime}, W'={Wprime} "
                f"from input (T={T_in},H={H_in},W={W_in}) with "
                f"kernel={(kt,kh,kw)}, stride={(st,sh,sw)}, padding={(pt,ph,pw)}."
            )

        self._spatial_hw = (int(Hprime), int(Wprime))
        return int(Tprime), int(Hprime * Wprime)

    def _compute_width_coords(self, w_tokens: int) -> Optional[torch.Tensor]:
        bounds = self._tile_bounds
        if not bounds or w_tokens <= 0:
            return None
        patch_w = max(int(self.tiling_patch_w), 1)
        coords: List[float] = []
        for left, right in bounds:
            width_px = max(int(right - left), 0)
            tokens_est = int(round(width_px / patch_w))
            tokens_est = max(1, min(tokens_est, w_tokens))
            for idx in range(w_tokens):
                if idx < tokens_est:
                    center = left + (idx + 0.5) * patch_w
                else:
                    last_center = left + (tokens_est - 0.5) * patch_w
                    center = last_center if tokens_est > 0 else left + 0.5 * patch_w
                coords.append(float(center))
        if not coords:
            return None
        return torch.tensor(coords, dtype=torch.float32)

    def _compute_width_valid_mask(self, w_tokens: int) -> Optional[torch.Tensor]:
        bounds = self._tile_bounds
        if not bounds or w_tokens <= 0:
            return None
        patch_w = max(int(self.tiling_patch_w), 1)
        mask_vals: List[bool] = []
        for left, right in bounds:
            width_px = max(int(right - left), 0)
            tokens_est = int(round(width_px / patch_w))
            tokens_est = max(1, min(tokens_est, w_tokens))
            mask_vals.extend([True] * tokens_est)
            mask_vals.extend([False] * max(0, w_tokens - tokens_est))
        if not mask_vals:
            return None
        return torch.tensor(mask_vals, dtype=torch.bool)

    def _prepare_width_features(
        self, enc_5d: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        spatial_hw = self._spatial_hw
        if spatial_hw is None:
            return None, None, None
        h_tokens, w_tokens = spatial_hw
        if h_tokens <= 0 or w_tokens <= 0:
            return None, None, None
        try:
            enc_hw = rearrange(enc_5d, "b m t (h w) d -> b m t h w d", h=h_tokens, w=w_tokens)
        except RuntimeError:
            return None, None, None
        enc_height_avg = enc_hw.mean(dim=3)  # avg over height bins
        width_feat = rearrange(enc_height_avg, "b m t w d -> b t (m w) d")
        width_coords = self._compute_width_coords(w_tokens)
        valid_mask = self._compute_width_valid_mask(w_tokens)
        return width_feat, width_coords, valid_mask

    def _apply_key_head(self, head: nn.Module, key_feat: torch.Tensor) -> torch.Tensor:
        """
        Apply a shared head independently to each key feature vector.

        Args:
            head: module mapping from D -> 1.
            key_feat: (B, T, K, D) tensor.

        Returns:
            (B, T, K) logits tensor.
        """
        if key_feat.ndim != 4:
            raise ValueError(f"Expected key features with rank 4 (B,T,K,D), got {tuple(key_feat.shape)}")

        key_feat_flat = key_feat.reshape(-1, key_feat.shape[-1])
        logits = head(key_feat_flat)
        logits = logits.reshape(*key_feat.shape[:-1], -1)
        return logits.squeeze(-1)


    def forward(self, x):
        """
        x: (B, T, tiles, C, H, W)  OR  (B, T, C, H, W)
        Returns dict of logits; shapes depend on head_mode ("clip" or "frame").
        """
        # Normalize input to a list of per-tile tensors with shape (B, T, C, H, W)
        if x.dim() == 6:
            B, T, M, C, H, W = x.shape
            tiles = [x[:, :, i, :, :, :] for i in range(M)]
        elif x.dim() == 5:
            B, T, C, H, W = x.shape
            tiles = self._tile_split_token_aligned(x)
        else:
            raise ValueError(f"Unexpected input rank {x.dim()} for TiViTPiano forward")
            
        # Per-tile tubelet embedding (shared weights)
        tile_tokens = []
        t_tokens = s_tokens = None
        for t in tiles:
            # (B, T, C, H, W) -> (B, C, T, H, W)
            t_cf = rearrange(t, 'b tt c h w -> b c tt h w')
            tok, _ = self.embed(t_cf)  # (B, Ntokens, D) where Ntokens = T' * S'
            if t_tokens is None or s_tokens is None:
                t_tokens, s_tokens = self._infer_token_factors(t_cf)
            tile_tokens.append(tok)

        # Ensure encoder is initialized before use
        self._init_encoder_if_needed(t_tokens=t_tokens, s_tokens=s_tokens)
        tile_tokens = [self.pos_drop(tok) for tok in tile_tokens]

        # Factorized encoding across time then tiles
        if self.encoder is None:
            raise RuntimeError("Encoder was not initialized properly.")
        enc = self.encoder(tile_tokens, t_tokens=t_tokens, s_tokens=s_tokens)  # (B, tiles*Ntok, D)

        # Recover 5D token grid: (B, tiles, T', S', D)
        enc_tiles = rearrange(enc, 'b (m n) d -> b m n d', m=self.tiles)          # (B, tiles, T'*S', D)
        enc_5d    = rearrange(enc_tiles, 'b m (t s) d -> b m t s d', t=t_tokens)  # (B, tiles, T', S', D)

        width_feat: Optional[torch.Tensor] = None
        width_coords_cpu: Optional[torch.Tensor] = None
        valid_mask_cpu: Optional[torch.Tensor] = None
        if self.per_key_head:
            width_feat, width_coords_cpu, valid_mask_cpu = self._prepare_width_features(enc_5d)

        # ----- Frame mode: per-time features and time-distributed heads -----
        if getattr(self, "head_mode", "clip") == "frame":
            if width_feat is not None:
                g_t = width_feat.mean(dim=2)
            else:
                g_t = enc_5d.mean(dim=(1, 3))

            if self.per_key_head:
                keypool_reg: Optional[torch.Tensor] = None
                key_features: Optional[torch.Tensor] = None
                if (
                    self.key_pool is not None
                    and width_feat is not None
                    and width_coords_cpu is not None
                ):
                    width_coords = width_coords_cpu.to(device=width_feat.device, dtype=width_feat.dtype)
                    valid_mask = valid_mask_cpu.to(device=width_feat.device) if valid_mask_cpu is not None else None
                    key_features, keypool_reg = self.key_pool(width_feat, width_coords, valid_mask=valid_mask)
                else:
                    if not self._keypool_warned:
                        LOGGER.warning("keypool: width grid unavailable; using global width average.")
                        self._keypool_warned = True

                if key_features is not None:
                    frame_key_features = key_features
                else:
                    frame_key_features = g_t.unsqueeze(2).expand(-1, -1, self.num_keys, -1).contiguous()

                pitch_logits = self._apply_key_head(self.head_pitch, frame_key_features)
                onset_logits = self._apply_key_head(self.head_onset, frame_key_features)
                offset_logits = self._apply_key_head(self.head_offset, frame_key_features)

                hand_logits = self.head_hand(g_t)
                clef_logits = self.head_clef(g_t)

                out = {
                    "pitch_logits": pitch_logits,
                    "onset_logits": onset_logits,
                    "offset_logits": offset_logits,
                    "hand_logits": hand_logits,
                    "clef_logits": clef_logits,
                }
                if keypool_reg is not None:
                    out["keypool_reg"] = keypool_reg
                return out

            pitch_logits = self.head_pitch(g_t)
            onset_logits = self.head_onset(g_t)
            offset_logits = self.head_offset(g_t)
            hand_logits = self.head_hand(g_t)
            clef_logits = self.head_clef(g_t)
            return {
                "pitch_logits": pitch_logits,
                "onset_logits": onset_logits,
                "offset_logits": offset_logits,
                "hand_logits": hand_logits,
                "clef_logits": clef_logits,
            }

        # ----- Clip mode: global mean-pool over all tokens -----
        if width_feat is not None:
            g = width_feat.mean(dim=(1, 2))
        else:
            g = enc.mean(dim=1)

        if self.per_key_head:
            keypool_reg: Optional[torch.Tensor] = None
            key_features_clip: Optional[torch.Tensor] = None
            if (
                self.key_pool is not None
                and width_feat is not None
                and width_coords_cpu is not None
            ):
                width_coords = width_coords_cpu.to(device=width_feat.device, dtype=width_feat.dtype)
                valid_mask = valid_mask_cpu.to(device=width_feat.device) if valid_mask_cpu is not None else None
                pooled_clip, keypool_reg = self.key_pool(width_feat.mean(dim=1, keepdim=True), width_coords, valid_mask=valid_mask)
                key_features_clip = pooled_clip
            else:
                if not self._keypool_warned:
                    LOGGER.warning("keypool: width grid unavailable; using global width average.")
                    self._keypool_warned = True

            if key_features_clip is not None:
                clip_key_features = key_features_clip
            else:
                clip_key_features = g.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_keys, -1).contiguous()

            pitch_logits = self._apply_key_head(self.head_pitch, clip_key_features).squeeze(1)
            onset_logits = self._apply_key_head(self.head_onset, clip_key_features).squeeze(1)
            offset_logits = self._apply_key_head(self.head_offset, clip_key_features).squeeze(1)

            out = {
                "pitch_logits": pitch_logits,
                "onset_logits": onset_logits,
                "offset_logits": offset_logits,
                "hand_logits":   self.head_hand(g),
                "clef_logits":   self.head_clef(g),
            }
            if keypool_reg is not None:
                out["keypool_reg"] = keypool_reg
            return out

        pitch_logits = self.head_pitch(g)
        onset_logits = self.head_onset(g)
        offset_logits = self.head_offset(g)
        return {
            "pitch_logits": pitch_logits,
            "onset_logits": onset_logits,
            "offset_logits": offset_logits,
            "hand_logits":   self.head_hand(g),
            "clef_logits":   self.head_clef(g),
        }
