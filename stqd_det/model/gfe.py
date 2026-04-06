"""Global Feature Enhancement (GFE) Module.

Applied only to the highest FPN layer (stride-32, lowest resolution).
Enhances N frames' feature maps via contextual grouping, multi-head attention,
and dynamic convolution, then merges back into the full FPN output.

Architecture per frame n:
  1. Tokenize:  f_n → v_n  (flatten to vector)
  2. Group:     ν_n = [v_{n-1}, v_n, v_{n+1}]  (with boundary duplication)
  3. MHA:       v_n' = LayerNorm(MHA(Q=v_n, K=v_{n-1}, V=v_{n+1}) + v_n)
  4. DynConv:   f_n'' = LayerNorm(FC(DC(f_n')) + f_n')
  5. Merge:     Replace top FPN layer with enhanced features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class DynamicConv(nn.Module):
    """Dynamic Convolution: generates per-sample kernel weights from input.

    For each input feature map, a small FC generates depth-wise convolution
    kernel weights conditioned on the global average of the input. This makes
    the convolution adaptive to the content.

    Args:
        channels: Number of input/output channels.
        kernel_size: Spatial kernel size for the depth-wise conv.
        groups: Number of groups (not used in depth-wise mode, kept for API).
    """

    def __init__(self, channels: int, kernel_size: int = 3, groups: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Kernel generator: from global-avg-pooled features → depthwise kernel weights
        # Depthwise: one k×k kernel per channel → C * k * k weights
        kernel_numel = channels * kernel_size * kernel_size
        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, kernel_numel),
        )

        # Pointwise (1×1) projection after dynamic conv
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(32, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) feature maps.

        Returns:
            (N, C, H, W) dynamically-convolved features.
        """
        N, C, H, W = x.shape
        k = self.kernel_size
        pad = k // 2

        # Generate per-sample depthwise kernels: (N, C * k * k)
        kernels = self.kernel_gen(x)
        # Reshape to (N*C, 1, k, k) for depthwise conv
        kernels = kernels.reshape(N * C, 1, k, k)

        # Pad input for 'same' convolution
        x_padded = F.pad(x, [pad, pad, pad, pad])

        # Reshape to (1, N*C, H+2p, W+2p) for batched depthwise conv
        x_grouped = x_padded.reshape(1, N * C, H + 2 * pad, W + 2 * pad)

        # Depthwise conv: groups = N*C (one kernel per channel per sample)
        out = F.conv2d(x_grouped, kernels, groups=N * C)
        out = out.reshape(N, C, H, W)

        out = self.proj(out)
        out = self.norm(out)
        return out


class GFEAttention(nn.Module):
    """Multi-Head Attention block for contextually-grouped frame features.

    For frame n, the three grouped feature maps (flattened to spatial token
    sequences of dim C) serve as:
      K = f_{n-1}, Q = f_n, V = f_{n+1}

    Args:
        embed_dim: Channel dimension C (each spatial position is a token).
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, v_prev: torch.Tensor, v_curr: torch.Tensor, v_next: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            v_prev: (N, S, C) — previous frame spatial tokens (Key).
            v_curr: (N, S, C) — current frame spatial tokens (Query).
            v_next: (N, S, C) — next frame spatial tokens (Value).
              N = batch of frames, S = H*W spatial positions, C = channels.

        Returns:
            v_enhanced: (N, S, C) — attention-enhanced tokens with residual.
        """
        attn_out, _ = self.mha(v_curr, v_prev, v_next)  # (N, S, C)

        # Residual + LayerNorm
        v_enhanced = self.layer_norm(attn_out + v_curr)
        return v_enhanced


class GFEModule(nn.Module):
    """Global Feature Enhancement applied to top FPN layer across N frames.

    Pipeline:
      1. Extract top-layer feature maps for all N frames → (N, C, H, W)
      2. Flatten to tokens v_n → (N, C*H*W)
      3. For each frame n: MHA with contextual grouping → v_n'
      4. Reshape v_n' → (N, C, H, W), apply Dynamic Convolution → f_n''
      5. f_n'' = LayerNorm(FC(DC(f_n')) + f_n')  (residual)
      6. Return updated FPN dict with enhanced top layer

    Args:
        cfg: Config with C, gfe_num_heads, gfe_dropout, gfe_dc_groups,
             top_fpn_spatial.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        C = cfg.C
        spatial = cfg.top_fpn_spatial  # H = W = 16 for 512 input

        # Multi-Head Attention over spatial tokens (each token is C-dim)
        self.attention = GFEAttention(
            embed_dim=C,
            num_heads=cfg.gfe_num_heads,
            dropout=cfg.gfe_dropout,
        )

        # Dynamic Convolution on the spatial feature maps
        self.dynamic_conv = DynamicConv(
            channels=C,
            kernel_size=3,
            groups=cfg.gfe_dc_groups,
        )

        # Final FC + LayerNorm for DC residual (Eq. 7)
        self.dc_fc = nn.Sequential(
            nn.Conv2d(C, C, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1),
        )
        self.dc_norm = nn.LayerNorm([C, spatial, spatial])

        self.spatial = spatial

    def forward(
        self, fpn_features: dict, num_frames: int
    ) -> dict:
        """
        Args:
            fpn_features: OrderedDict with keys "0".."3", each (B*N, C, H_i, W_i).
                B is batch size, N is number of frames. Features are interleaved
                as [b0_f0, b0_f1, ..., b0_fN-1, b1_f0, ...].
            num_frames: N (number of frames per batch element).

        Returns:
            Updated fpn_features with enhanced top layer "3".
        """
        top_key = "3"  # stride-32, lowest resolution
        feat = fpn_features[top_key]  # (B*N, C, H, W)
        BN, C, H, W = feat.shape
        N = num_frames
        B = BN // N

        # Reshape to (B, N, C, H, W)
        feat_5d = feat.reshape(B, N, C, H, W)

        enhanced_frames = []
        for b in range(B):
            frame_feats = feat_5d[b]  # (N, C, H, W)

            # Step 1: Tokenize — flatten spatial dims to token sequence
            # Each spatial position is a C-dim token: (N, H*W, C)
            tokens = frame_feats.reshape(N, C, H * W).permute(0, 2, 1)  # (N, S, C)

            # Step 2: Contextual grouping + MHA for each frame
            enhanced_tokens = []
            for n in range(N):
                prev_idx = max(0, n - 1)
                next_idx = min(N - 1, n + 1)

                v_prev = tokens[prev_idx].unsqueeze(0)  # (1, S, C)
                v_curr = tokens[n].unsqueeze(0)          # (1, S, C)
                v_next = tokens[next_idx].unsqueeze(0)   # (1, S, C)

                v_enhanced = self.attention(v_prev, v_curr, v_next)  # (1, S, C)
                enhanced_tokens.append(v_enhanced)

            enhanced_tokens = torch.cat(enhanced_tokens, dim=0)  # (N, S, C)

            # Step 3: Reshape back to spatial feature maps
            feat_attn = enhanced_tokens.permute(0, 2, 1).reshape(N, C, H, W)

            # Step 4: Dynamic Convolution + residual (Eq. 7)
            feat_dc = self.dynamic_conv(feat_attn)            # (N, C, H, W)
            feat_dc = self.dc_norm(self.dc_fc(feat_dc) + feat_attn)

            enhanced_frames.append(feat_dc)

        # Stack back: (B, N, C, H, W) → (B*N, C, H, W)
        enhanced = torch.stack(enhanced_frames, dim=0)  # (B, N, C, H, W)
        enhanced = enhanced.reshape(BN, C, H, W)

        # Update FPN dict
        fpn_features = dict(fpn_features)  # make mutable copy
        fpn_features[top_key] = enhanced
        return fpn_features
