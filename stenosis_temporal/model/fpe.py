"""Feature and Proposal Extraction (FPE) module.

ResNet50 backbone + FPN + RPN, with a 1→3 channel adapter for grayscale input.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign
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
        # Keep layers up to layer4
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # stride 4,  out 256
        self.layer2 = backbone.layer2  # stride 8,  out 512
        self.layer3 = backbone.layer3  # stride 16, out 1024
        self.layer4 = backbone.layer4  # stride 32, out 2048

        in_channels_list = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=256,
            extra_blocks=LastLevelMaxPool(),
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
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
        return self.fpn(fpn_input)  # keys "0".."3" + "pool"


class FPE(nn.Module):
    """Feature and Proposal Extraction: grayscale adapter + ResNet50-FPN + RPN."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # 1-channel → 3-channel adapter
        self.channel_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        nn.init.constant_(self.channel_adapter.weight, 1.0 / 3.0)

        self.backbone = ResNet50FPN(gradient_checkpointing=cfg.gradient_checkpointing)

        # Anchor generator — one set of sizes per FPN level
        anchor_generator = AnchorGenerator(
            sizes=cfg.anchor_sizes,
            aspect_ratios=cfg.anchor_ratios,
        )

        # RPN head
        rpn_head = RPNHead(
            in_channels=cfg.C,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )

        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=cfg.rpn_fg_iou_thresh,
            bg_iou_thresh=cfg.rpn_bg_iou_thresh,
            batch_size_per_image=cfg.rpn_batch_size_per_image,
            positive_fraction=cfg.rpn_positive_fraction,
            pre_nms_top_n={
                "training": cfg.rpn_pre_nms_top_n_train,
                "testing": cfg.rpn_pre_nms_top_n_test,
            },
            post_nms_top_n={
                "training": cfg.rpn_post_nms_top_n_train,
                "testing": cfg.rpn_post_nms_top_n_test,
            },
            nms_thresh=cfg.rpn_nms_thresh,
        )

        # RoI Align — operates on FPN levels
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=cfg.roi_output_size,
            sampling_ratio=2,
        )

    def forward(self, images: torch.Tensor, targets=None):
        """
        Args:
            images: (N, 1, H, W) grayscale images
            targets: list[dict] with 'boxes' and 'labels' keys (training only)

        Returns:
            features: OrderedDict of FPN feature maps
            proposals: list[Tensor] — S proposals per image, each (S, 4)
            rpn_losses: dict of RPN losses (empty dict during eval)
            roi_features: Tensor (N*S, C, roi_size, roi_size)
        """
        N = images.shape[0]
        # Grayscale → 3ch
        x = self.channel_adapter(images)

        # Backbone + FPN
        features = self.backbone(x)

        # RPN needs an ImageList
        image_sizes = [(images.shape[2], images.shape[3])] * N
        image_list = ImageList(images, image_sizes)

        # Build RPN targets if training
        rpn_targets = None
        if targets is not None:
            rpn_targets = targets

        proposals, rpn_losses = self.rpn(image_list, features, rpn_targets)

        # Ensure exactly S proposals per image (pad or truncate)
        processed_proposals = []
        for p in proposals:
            if p.shape[0] > self.cfg.S:
                p = p[: self.cfg.S]
            elif p.shape[0] < self.cfg.S:
                pad = p[-1:].expand(self.cfg.S - p.shape[0], -1)
                p = torch.cat([p, pad], dim=0)
            processed_proposals.append(p)

        # RoI Align for all proposals
        roi_features = self.roi_align(features, processed_proposals, image_sizes)
        # roi_features: (N*S, C, roi_size, roi_size)

        return features, processed_proposals, rpn_losses, roi_features
