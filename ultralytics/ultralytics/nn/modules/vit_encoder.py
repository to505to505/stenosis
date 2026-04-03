"""ViT-Small encoder modules for YOLO integration.

Wraps a timm VisionTransformer (ViT-Small: 384d, 12 blocks, 6 heads, patch_size=16)
as Ultralytics backbone layers.

Two variants:
  1. ViTEncoder       — single stride-16 feature map (original, backward-compatible).
  2. ViTMultiScale    — extracts features from 3 intermediate transformer blocks
                        (default: blocks 3, 7, 11) and returns them one at a time
                        via ViTMultiScaleTap helper modules.

Uses dynamic_img_size=True for automatic positional embedding interpolation,
which is required because DetectionModel probes strides with 256×256 dummy input
while training uses 512×512.
"""

import copy

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class ViTEncoder(nn.Module):
    """ViT-Small backbone producing a stride-16 spatial feature map.

    Args:
        c1 (int): Input channels (3 for RGB / repeated grayscale).
        c2 (int): Embedding dimension / output channels (384 for ViT-Small).
        img_size (int): Training image size (used to initialize pos_embed grid).
    """

    def __init__(self, c1: int = 3, c2: int = 384, img_size: int = 512):
        super().__init__()
        self.patch_size = 16
        self.embed_dim = c2

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=c1,
            embed_dim=c2,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images.
        Returns:
            (B, embed_dim, H//16, W//16) spatial feature map.
        """
        B, _, H, W = x.shape
        gh, gw = H // self.patch_size, W // self.patch_size

        tokens = self.vit.forward_features(x)  # (B, 1+gh*gw, D)
        tokens = tokens[:, 1:]                  # drop CLS token

        return tokens.transpose(1, 2).reshape(B, self.embed_dim, gh, gw)


# ---------------------------------------------------------------------------
# Multi-scale ViT: extract features from different transformer block depths
# ---------------------------------------------------------------------------

class ViTMultiScale(nn.Module):
    """ViT-Small backbone that caches features from 3 intermediate blocks.

    In the YAML, place this as layer 0. It runs the full ViT forward pass and
    stores 3 intermediate feature maps (from ``tap_blocks``) in ``_ms_cache``.
    Subsequent ``ViTMultiScaleTap`` layers (indices 0/1/2) retrieve them.

    Args:
        c1 (int): Input channels.
        c2 (int): Embedding dimension (384 for ViT-Small).
        img_size (int): Training image size.
        tap_blocks (tuple[int,int,int]): Block indices to extract (0-indexed).
            Default (3, 7, 11) — early / middle / late.
    """

    def __init__(self, c1: int = 3, c2: int = 384, img_size: int = 512,
                 tap_blocks: tuple = (3, 7, 11)):
        super().__init__()
        self.patch_size = 16
        self.embed_dim = c2
        self.tap_blocks = tap_blocks

        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=c1,
            embed_dim=c2,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )
        # Separate LayerNorms for each tap point (block outputs aren't normalized)
        self.norms = nn.ModuleList([nn.LayerNorm(c2) for _ in tap_blocks])

        # Shared cache — filled during forward, read by ViTMultiScaleTap
        self._ms_cache: list[torch.Tensor] = []

    def __deepcopy__(self, memo):
        """Skip _ms_cache during deepcopy (contains non-leaf computation graph tensors)."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_ms_cache":
                setattr(result, k, [])
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run full ViT and cache multi-scale features.

        Returns the *last* tap feature map (deepest block) so downstream
        layers see a valid tensor. The other taps are retrieved via
        ViTMultiScaleTap.
        """
        B, _, H, W = x.shape
        gh, gw = H // self.patch_size, W // self.patch_size

        # --- manual forward through timm ViT internals ---
        vit = self.vit
        x = vit.patch_embed(x)
        x = vit._pos_embed(x)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)

        tap_set = set(self.tap_blocks)
        cache = []
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in tap_set:
                tap_idx = self.tap_blocks.index(i)
                tokens = x[:, 1:]  # drop CLS
                tokens = self.norms[tap_idx](tokens)
                feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, gh, gw)
                cache.append(feat)

        self._ms_cache = cache
        return cache[-1]  # return deepest as default output


class ViTMultiScaleTap(nn.Module):
    """Retrieve a cached multi-scale feature from ViTMultiScale.

    Place after ViTMultiScale in the YAML with ``from: 0`` (the ViTMultiScale
    layer index). Each tap has a ``tap_index`` (0 = early, 1 = mid, 2 = late).

    Args:
        c1 (int): Input channels (ignored — comes from cache).
        c2 (int): Output channels (ignored — same as embed_dim).
        tap_index (int): Which cached feature to return (0, 1, or 2).
    """

    def __init__(self, c1: int = 384, c2: int = 384, tap_index: int = 0):
        super().__init__()
        self.tap_index = tap_index
        self._vit_ms = None  # set during model construction in tasks.py

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is the output of ViTMultiScale (last tap). Walk up to find the cache."""
        # During YOLO's _predict_once, `x` is the output of the ViTMultiScale layer.
        # We need to access the cache. The ViTMultiScale module is stored in
        # self._vit_ms which gets set during model construction.
        return self._vit_ms._ms_cache[self.tap_index]
