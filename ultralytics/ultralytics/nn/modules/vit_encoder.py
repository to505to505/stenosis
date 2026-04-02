"""ViT-Small encoder module for YOLO integration.

Wraps a timm VisionTransformer (ViT-Small: 384d, 12 blocks, 6 heads, patch_size=16)
as a single Ultralytics backbone layer. Outputs a 2D spatial feature map at stride 16.

Uses dynamic_img_size=True for automatic positional embedding interpolation,
which is required because DetectionModel probes strides with 256×256 dummy input
while training uses 512×512.
"""

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

        # forward_features handles patch_embed, cls_token, pos_embed interpolation,
        # transformer blocks, and final norm (dynamic_img_size=True)
        tokens = self.vit.forward_features(x)  # (B, 1+gh*gw, D)
        tokens = tokens[:, 1:]                  # drop CLS token

        return tokens.transpose(1, 2).reshape(B, self.embed_dim, gh, gw)
