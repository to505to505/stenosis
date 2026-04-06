"""Feature Adaption module for Guided Anchoring.

Ported from mmdetection's FeatureAdaption — uses deformable convolutions
to adapt feature maps based on predicted anchor shapes.
"""

import torch
import torch.nn as nn

try:
    from mmcv.ops import DeformConv2d
except ImportError:
    # Fallback: use torchvision's deform_conv2d wrapped in an nn.Module
    from torchvision.ops import deform_conv2d

    class DeformConv2d(nn.Module):
        """Minimal DeformConv2d using torchvision ops."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            deform_groups: int = 1,
            bias: bool = False,
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.deform_groups = deform_groups

            self.weight = nn.Parameter(
                torch.empty(out_channels, in_channels // groups,
                            kernel_size, kernel_size)
            )
            nn.init.kaiming_uniform_(self.weight)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_channels))
            else:
                self.bias = None

        def forward(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
            return deform_conv2d(
                x, offset, self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )


class FeatureAdaption(nn.Module):
    """Feature adaption module from Guided Anchoring.

    Uses deformable convolution with offsets predicted from anchor shape
    to adapt features according to the predicted anchor geometry.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for deformable convolution.
        deform_groups: Number of deformable groups.
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        kernel_size: int = 3,
        deform_groups: int = 4,
    ):
        super().__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            2, deform_groups * offset_channels, 1, bias=False
        )
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.normal_(self.conv_adaption.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (N, C, H, W).
            shape: Predicted anchor shape (N, 2, H, W) — (w_delta, h_delta).

        Returns:
            Adapted feature map (N, C, H, W).
        """
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.conv_adaption(x, offset))
        return x
