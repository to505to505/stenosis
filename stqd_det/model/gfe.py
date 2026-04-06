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
    """Multi-Head Attention block for contextually-grouped frame vectors.

    For frame n, the three grouped vectors serve as:
      K = v_{n-1}, Q = v_n, V = v_{n+1}

    Args:
        embed_dim: Dimension of each frame's feature vector.
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
            v_prev: (N, D) — previous frame token (Key).
            v_curr: (N, D) — current frame token (Query).
            v_next: (N, D) — next frame token (Value).
              Here N is the number of frames being processed (can be batched).

        Returns:
            v_enhanced: (N, D) — attention-enhanced token with residual.
        """
        # MHA expects (batch, seq_len, dim). Each frame has seq_len=1 token,
        # but K/V come from adjacent frames. Stack as seq_len=1 for Q,
        # and seq_len=1 for K and V individually.
        # Actually: Q is current, K is previous, V is next — each is a single
        # token, so we treat this as cross-attention with seq_len=1.
        Q = v_curr.unsqueeze(1)  # (N, 1, D)
        K = v_prev.unsqueeze(1)  # (N, 1, D)
        V = v_next.unsqueeze(1)  # (N, 1, D)

        attn_out, _ = self.mha(Q, K, V)  # (N, 1, D)
        attn_out = attn_out.squeeze(1)    # (N, D)

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
        token_dim = C * spatial * spatial  # 256 * 16 * 16 = 65536

        # Project to a manageable attention dimension
        self.attn_dim = min(token_dim, 512)
        self.token_proj = nn.Linear(token_dim, self.attn_dim)
        self.token_unproj = nn.Linear(self.attn_dim, token_dim)

        # Multi-Head Attention over projected tokens
        self.attention = GFEAttention(
            embed_dim=self.attn_dim,
            num_heads=cfg.gfe_num_heads,
            dropout=cfg.gfe_dropout,
        )

        # Dynamic Convolution on the spatial feature maps
        self.dynamic_conv = DynamicConv(
            channels=C,
            kernel_size=3,
            groups=cfg.gfe_dc_groups,
        )

        # Final FC + LayerNorm for DC residual
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

            # Step 1: Tokenize — flatten spatial dims
            tokens = frame_feats.reshape(N, C * H * W)  # (N, token_dim)

            # Step 2: Project to attention dimension
            tokens_proj = self.token_proj(tokens)  # (N, attn_dim)

            # Step 3: Contextual grouping + MHA for each frame
            enhanced_tokens = []
            for n in range(N):
                prev_idx = max(0, n - 1)
                next_idx = min(N - 1, n + 1)

                v_prev = tokens_proj[prev_idx].unsqueeze(0)  # (1, attn_dim)
                v_curr = tokens_proj[n].unsqueeze(0)          # (1, attn_dim)
                v_next = tokens_proj[next_idx].unsqueeze(0)   # (1, attn_dim)

                v_enhanced = self.attention(v_prev, v_curr, v_next)  # (1, attn_dim)
                enhanced_tokens.append(v_enhanced)

            enhanced_tokens = torch.cat(enhanced_tokens, dim=0)  # (N, attn_dim)

            # Step 4: Unproject back to full token dim and reshape to spatial
            tokens_full = self.token_unproj(enhanced_tokens)  # (N, C*H*W)
            feat_attn = tokens_full.reshape(N, C, H, W)

            # Step 5: Dynamic Convolution + residual
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
