"""Faster R-CNN with ViT-Small (VasoMIM) backbone + Simple Feature Pyramid.

Architecture:
    Grayscale → 3ch repeat → ViT-Small(384d, 12 blocks, 6 heads)
    → reshape to spatial → SimpleFPN (4 levels, 256ch)
    → RPN → RoIAlign → TwoMLPHead → FastRCNNPredictor
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    TwoMLPHead,
)
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign


# ---------------------------------------------------------------------------
# ViT-Small backbone
# ---------------------------------------------------------------------------

class ViTSmallBackbone(nn.Module):
    """ViT-Small encoder that outputs a 2D spatial feature map."""

    def __init__(self, img_size: int = 512, pretrained_path: str = None):
        super().__init__()
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_chans=3,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            num_classes=0,       # no classification head
            global_pool="",      # no pooling — keep all tokens
        )
        self.embed_dim = 384
        self.grid_size = img_size // 16   # 32 for 512×512

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=True)
        msg = self.vit.load_state_dict(state, strict=False)
        print(f"[ViTSmallBackbone] loaded {path}")
        print(f"  missing : {msg.missing_keys}")
        print(f"  unexpected: {msg.unexpected_keys}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            (B, 384, grid_h, grid_w) — single-scale feature map at stride 16
        """
        # timm VisionTransformer with num_classes=0, global_pool="" returns (B, N+1, D)
        tokens = self.vit.forward_features(x)       # (B, 1+G*G, 384)
        tokens = tokens[:, 1:, :]                    # drop cls token → (B, G*G, 384)
        B, N, D = tokens.shape
        G = self.grid_size
        return tokens.transpose(1, 2).reshape(B, D, G, G)


# ---------------------------------------------------------------------------
# Simple Feature Pyramid (ViTDet-style)
# ---------------------------------------------------------------------------

class SimpleFPN(nn.Module):
    """Converts a single stride-16 feature map into a 4-level feature pyramid.

    Produces levels at strides 4, 8, 16, 32, each with `out_channels` channels.
    """

    def __init__(self, in_dim: int = 384, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # stride 4: 2× upsample
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(1, in_dim // 2),   # equivalent to LayerNorm for (B,C,H,W)
            nn.GELU(),
            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
        )
        self.lat0 = nn.Sequential(
            nn.Conv2d(in_dim // 4, out_channels, 1), nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # stride 8: 1× upsample
        self.up1 = nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)
        self.lat1 = nn.Sequential(
            nn.Conv2d(in_dim // 2, out_channels, 1), nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # stride 16: identity
        self.lat2 = nn.Sequential(
            nn.Conv2d(in_dim, out_channels, 1), nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        # stride 32: downsample
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lat3 = nn.Sequential(
            nn.Conv2d(in_dim, out_channels, 1), nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        """
        Args:
            x: (B, in_dim, G, G) — stride-16 features from ViT
        Returns:
            OrderedDict with keys "0".."3" at strides 4, 8, 16, 32
        """
        return OrderedDict([
            ("0", self.lat0(self.up2(x))),          # stride 4
            ("1", self.lat1(self.up1(x))),          # stride 8
            ("2", self.lat2(x)),                     # stride 16
            ("3", self.lat3(self.down(x))),          # stride 32
        ])


# ---------------------------------------------------------------------------
# Combined backbone + FPN
# ---------------------------------------------------------------------------

class ViTDetBackbone(nn.Module):
    """ViT-Small + SimpleFPN producing multi-scale features for detection."""

    def __init__(self, img_size: int = 512, out_channels: int = 256,
                 pretrained_path: str = None):
        super().__init__()
        self.vit = ViTSmallBackbone(img_size=img_size, pretrained_path=pretrained_path)
        self.fpn = SimpleFPN(in_dim=self.vit.embed_dim, out_channels=out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict:
        feat = self.vit(x)
        return self.fpn(feat)


# ---------------------------------------------------------------------------
# Faster R-CNN assembly
# ---------------------------------------------------------------------------

def build_faster_rcnn(
    num_classes: int = 2,
    pretrained_path: str = None,
    img_size: int = 512,
    out_channels: int = 256,
    roi_output_size: int = 7,
    representation_size: int = 1024,
):
    """Build Faster R-CNN with ViT-Small + SimpleFPN backbone.

    Args:
        num_classes: number of classes INCLUDING background (2 = bg + stenosis)
        pretrained_path: path to VasoMIM encoder weights
        img_size: input image size
        out_channels: FPN output channels
        roi_output_size: RoI pooling output size
        representation_size: hidden dim for detection head MLP

    Returns:
        GeneralizedRCNN model
    """
    # Backbone
    backbone = ViTDetBackbone(
        img_size=img_size,
        out_channels=out_channels,
        pretrained_path=pretrained_path,
    )

    # Anchor generator — 4 levels (no extra pool level)
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    # RPN
    rpn_head = RPNHead(
        in_channels=out_channels,
        num_anchors=anchor_generator.num_anchors_per_location()[0],
    )
    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=rpn_head,
        fg_iou_thresh=0.7,
        bg_iou_thresh=0.3,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n={"training": 2000, "testing": 1000},
        post_nms_top_n={"training": 1000, "testing": 300},
        nms_thresh=0.7,
    )

    # RoI heads
    roi_align = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=roi_output_size,
        sampling_ratio=2,
    )

    box_head = TwoMLPHead(
        in_channels=out_channels * roi_output_size ** 2,
        representation_size=representation_size,
    )

    box_predictor = FastRCNNPredictor(
        in_channels=representation_size,
        num_classes=num_classes,
    )

    roi_heads = RoIHeads(
        box_roi_pool=roi_align,
        box_head=box_head,
        box_predictor=box_predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.5,
        batch_size_per_image=128,
        positive_fraction=0.25,
        bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=100,
    )

    # Transform — normalizes & batches images
    transform = GeneralizedRCNNTransform(
        min_size=img_size,
        max_size=img_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    model = GeneralizedRCNN(backbone, rpn, roi_heads, transform)
    return model
