"""ResNet-50 backbone with FPN for Stenosis-DetNet.

Grayscale (1ch) → 3ch adapter → ResNet-50 → FPN (256ch @ 4 scales).
Outputs feature maps at strides 4, 8, 16, 32 plus a pool level.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)

from ..config import Config


class ResNet50FPN(nn.Module):
    """ResNet50 backbone with FPN producing multi-scale feature maps (C=256)."""

    def __init__(self, gradient_checkpointing: bool = False):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Keep layers up to layer4, strip fc and avgpool
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # stride 4,  out 256
        self.layer2 = backbone.layer2  # stride 8,  out 512
        self.layer3 = backbone.layer3  # stride 16, out 1024
        self.layer4 = backbone.layer4  # stride 32, out 2048

        # FPN: 4 scale inputs → 256ch outputs + pool level
        in_channels_list = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """
        Args:
            x: (N, 3, H, W) images (after channel adapter)

        Returns:
            OrderedDict: FPN feature maps "0".."3" + "pool"
              "0": (N, 256, H/4,  W/4)   = 256×128×128 for 512 input
              "1": (N, 256, H/8,  W/8)   = 256×64×64
              "2": (N, 256, H/16, W/16)  = 256×32×32
              "3": (N, 256, H/32, W/32)  = 256×16×16
        """
        c1 = self.layer0(x)
        if self.gradient_checkpointing and self.training:
            c2 = checkpoint(self.layer1, c1, use_reentrant=False)
            c3 = checkpoint(self.layer2, c2, use_reentrant=False)
            c4 = checkpoint(self.layer3, c3, use_reentrant=False)
            c5 = checkpoint(self.layer4, c4, use_reentrant=False)
        else:
            c2 = self.layer1(c1)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)

        fpn_input = OrderedDict([
            ("0", c2), ("1", c3), ("2", c4), ("3", c5),
        ])
        return self.fpn(fpn_input)


class Backbone(nn.Module):
    """Full backbone: grayscale adapter + ResNet50-FPN."""

    def __init__(self, cfg: Config):
        super().__init__()
        # 1-channel → 3-channel adapter (initialized to 1/3 for neutral start)
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.channel_adapter.weight, 1.0 / 3.0)

        self.resnet_fpn = ResNet50FPN(
            gradient_checkpointing=cfg.gradient_checkpointing
        )

    def forward(self, images: torch.Tensor) -> OrderedDict:
        """
        Args:
            images: (N, 1, H, W) grayscale images

        Returns:
            FPN feature maps OrderedDict with keys "0".."3" + "pool"
        """
        x = self.channel_adapter(images)
        return self.resnet_fpn(x)
