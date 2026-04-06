"""Classification and Regression heads for Stenosis-DetNet.

Two fully connected layers for binary classification (stenosis vs non-stenosis)
and bounding box regression (x, y, h, w).
"""

import torch
import torch.nn as nn

from ..config import Config


class DetectionHeads(nn.Module):
    """Binary classification + bounding box regression heads.

    Input: RoI features (N, C, roi_size, roi_size)
    Output: class logits (N, 2), box deltas (N, 4)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.C  # 256 after adaptive avg pool

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.fc_hidden, cfg.num_classes),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(in_dim, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.fc_hidden, 4),
        )

    def forward(self, roi_features: torch.Tensor):
        """
        Args:
            roi_features: (N, C, roi_size, roi_size)

        Returns:
            cls_logits: (N, num_classes)
            box_deltas: (N, 4) — (dx, dy, dw, dh)
        """
        x = self.pool(roi_features).flatten(1)  # (N, C)
        cls_logits = self.cls_head(x)
        box_deltas = self.reg_head(x)
        return cls_logits, box_deltas
