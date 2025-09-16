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

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        if self.depth_global > 0 and self.global_ctx is not None:
            g = self.global_ctx.unsqueeze(0).expand(B, -1, -1)  # (B, G, D)
            x = torch.cat([g, x], dim=1)  # prepend global tokens
            for blk in self.global_blocks:
                x = blk(x)
            x = x[:, self.global_tokens:, :]  # drop globals
        return x

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
    ):
        super().__init__()
        self.tiles = tiles
        self.head_mode = head_mode  # runtime switch for clip vs frame heads

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

        # Heads (LayerNorm -> Linear). These work with (B, D) and (B, T, D).
        self.head_pitch  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, pitch_classes))
        self.head_onset  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 88))
        self.head_offset = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 88))
        self.head_hand   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2))
        self.head_clef   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, clef_classes))

        # Cache Tubelet conv sizes for token shape inference
        self._tube_k   = self.embed.proj.kernel_size[0]
        self._tube_s   = self.embed.proj.stride[0]
        self._patch_kh = self.embed.proj.kernel_size[1]
        self._patch_kw = self.embed.proj.kernel_size[2]
        self._patch_sh = self.embed.proj.stride[1]
        self._patch_sw = self.embed.proj.stride[2]

    def _init_encoder_if_needed(self, t_tokens, s_tokens):
        if self.encoder is None:
            self.encoder = FactorizedSpaceTimeEncoder(
                t_tokens=t_tokens, s_tokens=s_tokens, **self.encoder_cfg
            )

    @staticmethod
    def _tile_split(x):
        """
        Split frame horizontally into 3 tiles if needed.
        Input:  x (B, T, C, H, W)
        Return: list of 3 tensors each (B, T, C, H, W/3)
        """
        B, T, C, H, W = x.shape
        w3 = W // 3
        return [
            x[..., 0:w3],
            x[..., w3:2*w3],
            x[..., 2*w3:3*w3],
        ]

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
        kt, kh, kw = self.embed.proj.kernel_size
        st, sh, sw = self.embed.proj.stride
        # If you ever change TubeletEmbed padding, adjust here accordingly:
        pt, ph, pw = self.embed.proj.padding if hasattr(self.embed.proj, "padding") else (0, 0, 0)

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

        return int(Tprime), int(Hprime * Wprime)


    def forward(self, x):
        """
        x: (B, T, tiles, C, H, W)  OR  (B, T, C, H, W)
        Returns dict of logits; shapes depend on head_mode ("clip" or "frame").
        """
        # Normalize input to a list of per-tile tensors with shape (B, T, C, H, W)
        if x.dim() == 6:
            B, T, M, C, H, W = x.shape
            tiles = [x[:, :, i, :, :, :] for i in range(M)]
        else:
            B, T, C, H, W = x.shape
            tiles = self._tile_split(x)

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

        # Init encoder if needed and apply dropout to tokens
        self._init_encoder_if_needed(t_tokens=t_tokens, s_tokens=s_tokens)
        tile_tokens = [self.pos_drop(tok) for tok in tile_tokens]

        # Factorized encoding across time then tiles
        enc = self.encoder(tile_tokens, t_tokens=t_tokens, s_tokens=s_tokens)  # (B, tiles*Ntok, D)

        # Recover 5D token grid: (B, tiles, T', S', D)
        enc_tiles = rearrange(enc, 'b (m n) d -> b m n d', m=self.tiles)          # (B, tiles, T'*S', D)
        enc_5d    = rearrange(enc_tiles, 'b m (t s) d -> b m t s d', t=t_tokens)  # (B, tiles, T', S', D)

        # ----- Frame mode: per-time features and time-distributed heads -----
        if getattr(self, "head_mode", "clip") == "frame":
            # Avg over tiles and spatial positions -> (B, T', D)
            g_t = enc_5d.mean(dim=(1, 3))  # mean over tiles (m) and spatial (s)

            # Heads accept (B,T,D) thanks to LayerNorm -> Linear on last dim
            pitch_logits  = self.head_pitch(g_t)              # (B, T', 88)
            onset_logits  = self.head_onset(g_t)             # (B, T', 88)
            offset_logits = self.head_offset(g_t)            # (B, T', 88)
            hand_logits   = self.head_hand(g_t)              # (B, T', 2)
            clef_logits   = self.head_clef(g_t)              # (B, T', 3)

            return {
                "pitch_logits":  pitch_logits,
                "onset_logits":  onset_logits,
                "offset_logits": offset_logits,
                "hand_logits":   hand_logits,
                "clef_logits":   clef_logits,
            }

        # ----- Clip mode: global mean-pool over all tokens -----
        g = enc.mean(dim=1)  # (B, D)
        out = {
            "pitch_logits":  self.head_pitch(g),                 # (B, 88)
            "onset_logits":  self.head_onset(g),                 # (B, 88)
            "offset_logits": self.head_offset(g),                # (B, 88)
            "hand_logits":   self.head_hand(g),                  # (B, 2)
            "clef_logits":   self.head_clef(g),                  # (B, 3)
        }
        return out


