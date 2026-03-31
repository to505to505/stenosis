"""Multi-Task Outputs (MTO) module.

Classification and bounding-box regression heads operating on
aggregated RoI features from PSTFA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class MTO(nn.Module):
    """Multi-task classification + bounding-box regression head.

    Input: aggregated RoI features  (N, C, roi_size, roi_size)
    Output: class logits (N, num_classes), box deltas (N, 4)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.C  # 256 after adaptive avg pool

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.fc_hidden, cfg.num_classes),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(in_dim, cfg.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.fc_hidden, 4),
        )

    def forward(self, roi_features: torch.Tensor):
        """
        Args:
            roi_features: (N, C, roi_size, roi_size)

        Returns:
            cls_logits: (N, num_classes)
            box_deltas: (N, 4)  — (dx, dy, dw, dh) offsets
        """
        x = self.pool(roi_features).flatten(1)  # (N, C)
        cls_logits = self.cls_head(x)
        box_deltas = self.reg_head(x)
        return cls_logits, box_deltas
