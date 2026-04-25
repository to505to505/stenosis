"""Stenosis Detection Decoder (DiffusionDet-style).

Iterative decoder that takes FPN features + noisy proposals from SQNB,
extracts RoI features, and refines box predictions through self-attention
decoder layers. Each layer outputs refined boxes that are used for
RoI Align in the next layer.

Architecture per decoder layer:
  1. RoI Align from FPN using current proposals → (P, C, roi, roi)
  2. Self-attention over proposal features
  3. FFN + classification/regression heads
  4. Updated proposals = predicted boxes (fed to next layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from ..config import Config


class DecoderLayer(nn.Module):
    """Single decoder layer: self-attention + FFN with cls/reg heads.

    Args:
        d_model: Hidden dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward network hidden dimension.
        dropout: Dropout probability.
        num_classes: Number of foreground classes.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.d_model = d_model

        # Self-attention over proposals
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Per-layer prediction heads
        self.cls_head = nn.Linear(d_model, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, P, d_model) proposal features.

        Returns:
            features: (B, P, d_model) refined features.
            cls_logits: (B, P, num_classes) classification logits.
            box_deltas: (B, P, 4) box regression deltas.
        """
        # Self-attention (fp32 to prevent QK^T overflow in fp16)
        residual = features
        with torch.amp.autocast("cuda", enabled=False):
            attn_out, _ = self.self_attn(
                features.float(), features.float(), features.float()
            )
        features = self.norm1(residual + self.dropout1(attn_out))

        # FFN
        residual = features
        ffn_out = self.ffn(features)
        features = self.norm2(residual + self.dropout2(ffn_out))

        # Predictions
        cls_logits = self.cls_head(features)
        box_deltas = self.reg_head(features)

        return features, cls_logits, box_deltas


class StenosisDecoder(nn.Module):
    """DiffusionDet-style iterative decoder with multi-scale RoI Align.

    Process:
      1. Start with SQNB proposals.
      2. For each decoder layer:
         a. RoI Align from FPN using current proposals.
         b. Pool + project to d_model.
         c. Self-attention + FFN + predict cls/reg.
         d. Refine proposals using predicted deltas.
      3. Return predictions from all layers (for auxiliary losses).

    Args:
        cfg: Config with decoder_layers, decoder_dim, decoder_heads,
             decoder_ffn_dim, decoder_dropout, C, roi_output_size, num_classes.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.decoder_dim
        self.num_layers = cfg.decoder_layers
        self.roi_size = cfg.roi_output_size

        # Multi-scale RoI Align over FPN levels
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=cfg.roi_output_size,
            sampling_ratio=2,
        )

        # Project pooled RoI features to d_model
        roi_feat_dim = cfg.C * cfg.roi_output_size * cfg.roi_output_size
        self.input_proj = nn.Sequential(
            nn.Linear(roi_feat_dim, cfg.decoder_dim),
            nn.LayerNorm(cfg.decoder_dim),
            nn.ReLU(inplace=True),
        )

        # Stacked decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=cfg.decoder_dim,
                num_heads=cfg.decoder_heads,
                ffn_dim=cfg.decoder_ffn_dim,
                dropout=cfg.decoder_dropout,
                num_classes=cfg.num_classes,
            )
            for _ in range(cfg.decoder_layers)
        ])

        # Time embedding for diffusion step conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(1, cfg.decoder_dim),
            nn.SiLU(),
            nn.Linear(cfg.decoder_dim, cfg.decoder_dim),
        )

    def forward(
        self,
        fpn_features: dict,
        proposals: torch.Tensor,
        image_sizes: list[tuple[int, int]],
        timesteps: torch.Tensor | None = None,
    ) -> list[dict]:
        """
        Args:
            fpn_features: Dict of FPN feature maps, each (sum_N, C, H_i, W_i)
                where sum_N is the total frames across batch.
            proposals: (sum_N, P, 4) bounding box proposals in xyxy absolute pixels.
            image_sizes: List of (H, W) for each element in the batch
                (one per frame for RoI Align).
            timesteps: (sum_N,) diffusion timesteps (optional, for conditioning).

        Returns:
            outputs: List of dicts (one per decoder layer), each containing:
                "cls_logits": (sum_N, P, num_classes)
                "box_pred": (sum_N, P, 4) predicted boxes in xyxy absolute
        """
        sum_N = proposals.shape[0]
        P = proposals.shape[1]

        current_boxes = proposals.clone()
        all_outputs = []

        # Time embedding (optional conditioning)
        if timesteps is not None:
            t_embed = self.time_mlp(
                timesteps.float().unsqueeze(-1)  # (sum_N, 1)
            )  # (sum_N, d_model)
        else:
            t_embed = None

        for layer_idx, layer in enumerate(self.layers):
            # RoI Align: need boxes as list of (P, 4) per image
            roi_boxes_list = [current_boxes[i] for i in range(sum_N)]
            roi_features = self.roi_align(
                fpn_features, roi_boxes_list, image_sizes
            )  # (sum_N * P, C, roi, roi)

            # Project to d_model
            roi_flat = roi_features.flatten(1)  # (sum_N * P, C*roi*roi)
            roi_proj = self.input_proj(roi_flat)  # (sum_N * P, d_model)

            # Reshape to (sum_N, P, d_model) for batched attention
            features = roi_proj.reshape(sum_N, P, self.d_model)

            # Add time embedding if available
            if t_embed is not None:
                features = features + t_embed.unsqueeze(1)  # broadcast over P

            # Decoder layer: self-attention + FFN + predict
            features, cls_logits, box_deltas = layer(features)

            # Apply box deltas to refine proposals
            # box_deltas are (dx, dy, dw, dh) relative offsets
            pred_boxes = self._apply_deltas(current_boxes, box_deltas)

            all_outputs.append({
                "cls_logits": cls_logits,   # (sum_N, P, num_classes)
                "box_pred": pred_boxes,     # (sum_N, P, 4) xyxy absolute
            })

            # Update proposals for next layer
            current_boxes = pred_boxes.detach()

        return all_outputs

    def _apply_deltas(
        self, boxes: torch.Tensor, deltas: torch.Tensor
    ) -> torch.Tensor:
        """Apply regression deltas to boxes.

        Args:
            boxes: (sum_N, P, 4) current boxes in xyxy format.
            deltas: (sum_N, P, 4) predicted deltas (dx, dy, dw, dh).

        Returns:
            refined: (sum_N, P, 4) refined boxes in xyxy format.
        """
        # Convert to cxcywh
        cx = (boxes[..., 0] + boxes[..., 2]) / 2
        cy = (boxes[..., 1] + boxes[..., 3]) / 2
        w = (boxes[..., 2] - boxes[..., 0]).clamp(min=1.0)
        h = (boxes[..., 3] - boxes[..., 1]).clamp(min=1.0)

        # Apply deltas (clamp to prevent numerical blowup)
        dx, dy, dw, dh = deltas.unbind(-1)
        dx = dx.clamp(-4.0, 4.0)
        dy = dy.clamp(-4.0, 4.0)
        new_cx = cx + dx * w
        new_cy = cy + dy * h
        new_w = w * torch.exp(dw.clamp(-4.0, 4.0))
        new_h = h * torch.exp(dh.clamp(-4.0, 4.0))

        # Convert back to xyxy
        x1 = new_cx - new_w / 2
        y1 = new_cy - new_h / 2
        x2 = new_cx + new_w / 2
        y2 = new_cy + new_h / 2

        refined = torch.stack([x1, y1, x2, y2], dim=-1)

        # Clamp to image bounds
        refined[..., 0::2].clamp_(min=0, max=self.cfg.img_w)
        refined[..., 1::2].clamp_(min=0, max=self.cfg.img_h)

        return refined

    def forward_single_stage(
        self,
        fpn_features: dict,
        roi_features: torch.Tensor,
        proposals: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> list[dict]:
        """Run decoder on pre-extracted RoI features (for STFS second stage).

        Args:
            fpn_features: Not used directly (features already extracted).
            roi_features: (sum_N, P, C*roi*roi) pre-extracted RoI features.
            proposals: (sum_N, P, 4) current box proposals.
            timesteps: Optional diffusion timesteps.

        Returns:
            outputs: List of dicts from each decoder layer.
        """
        sum_N = roi_features.shape[0]
        P = roi_features.shape[1]

        # Project to d_model
        roi_proj = self.input_proj(roi_features)  # (sum_N, P, d_model)
        features = roi_proj

        if timesteps is not None:
            t_embed = self.time_mlp(timesteps.float().unsqueeze(-1))
            features = features + t_embed.unsqueeze(1)

        current_boxes = proposals.clone()
        all_outputs = []

        for layer in self.layers:
            features, cls_logits, box_deltas = layer(features)
            pred_boxes = self._apply_deltas(current_boxes, box_deltas)
            all_outputs.append({
                "cls_logits": cls_logits,
                "box_pred": pred_boxes,
            })
            current_boxes = pred_boxes.detach()

        return all_outputs
