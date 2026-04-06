"""STQD-Det: End-to-end Spatio-Temporal Quantum Diffusion Detector.

Full pipeline:
  Training:
    images (B,T,1,H,W) → Backbone (per-frame) → FPN
    → GFE on top layer → Enhanced FPN
    → SQNB (GT + Poisson noise) → noisy proposals
    → Decoder (stage 1) → candidate detections
    → STFS (Hungarian matching + voting + aggregation)
    → Decoder (stage 2, on corrected RoIs) → refined detections
    → Losses (Focal + L1 + GIoU + Consistency)

  Inference:
    Same pipeline, SQNB from image center, NMS, confidence > 0.5.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.ops import nms

from ..config import Config
from .backbone import Backbone
from .gfe import GFEModule
from .sqnb import SQNBGenerator
from .decoder import StenosisDecoder
from .stfs import STFSModule, hungarian_match_across_frames, vote_groups
from .losses import STQDDetCriterion


class STQDDet(nn.Module):
    """STQD-Det: Spatio-Temporal Quantum Diffusion Detector.

    Args:
        cfg: Model configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Step 1: Backbone + FPN
        self.backbone = Backbone(cfg)

        # Step 2: Global Feature Enhancement
        self.gfe = GFEModule(cfg)

        # Step 3: Sequential Quantum Noise Box generator
        self.sqnb = SQNBGenerator(cfg)

        # Step 4: Stenosis Detection Decoder (used for both stages)
        self.decoder = StenosisDecoder(cfg)

        # Step 5: Spatio-Temporal Feature Sharing
        self.stfs = STFSModule(cfg)

        # Step 6: Loss criterion
        self.criterion = STQDDetCriterion(cfg)

    def forward(
        self,
        images: torch.Tensor,
        targets: list[list[dict]] | None = None,
    ) -> dict | list[list[dict]]:
        """
        Args:
            images: (B, T, 1, H, W) batch of T-frame grayscale sequences.
            targets: Training only. List of B elements, each a list of T dicts
                with "boxes" (M, 4) xyxy and "labels" (M,) per frame.

        Returns:
            Training: dict of losses.
            Inference: list of B elements, each a list of T dicts with
                "boxes", "scores", "labels" per frame.
        """
        B, T, _, H, W = images.shape
        device = images.device

        # ── Step 1: Backbone ──────────────────────────────────────────
        # Process each frame through backbone separately to save VRAM.
        # With gradient checkpointing, intermediate ResNet activations are
        # freed after each frame and recomputed during backward.
        flat_images = images.reshape(B * T, 1, H, W)
        fpn_list = []
        for i in range(B * T):
            frame = flat_images[i : i + 1]  # (1, 1, H, W)
            if self.training and self.cfg.gradient_checkpointing:
                fpn_i = checkpoint(self.backbone, frame, use_reentrant=False)
            else:
                fpn_i = self.backbone(frame)
            fpn_list.append(fpn_i)
        # Merge per-frame FPN outputs: each key → (B*T, 256, H_i, W_i)
        fpn_features = {}
        for key in fpn_list[0]:
            fpn_features[key] = torch.cat([f[key] for f in fpn_list], dim=0)

        # ── Step 2: GFE on top FPN layer ─────────────────────────────
        fpn_features = self.gfe(fpn_features, num_frames=T)
        # fpn_features["3"] is now enhanced

        # Image sizes for RoI Align (all same size)
        image_sizes = [(H, W)] * (B * T)

        if self.training:
            return self._forward_train(
                fpn_features, targets, image_sizes, B, T, device
            )
        else:
            return self._forward_inference(
                fpn_features, image_sizes, B, T, device
            )

    def _forward_train(
        self,
        fpn_features: dict,
        targets: list[list[dict]],
        image_sizes: list[tuple[int, int]],
        B: int,
        T: int,
        device: torch.device,
    ) -> dict:
        """Training forward pass."""
        all_losses = {
            "loss_cls": torch.tensor(0.0, device=device),
            "loss_l1": torch.tensor(0.0, device=device),
            "loss_giou": torch.tensor(0.0, device=device),
            "loss_consistency": torch.tensor(0.0, device=device),
            "total_loss": torch.tensor(0.0, device=device),
        }

        for b in range(B):
            # Extract GT boxes per frame for this batch element
            gt_boxes = [targets[b][t]["boxes"].to(device) for t in range(T)]
            gt_labels = [targets[b][t]["labels"].to(device) for t in range(T)]

            # ── Step 3: SQNB (Poisson noise around GT) ────────────────
            noisy_proposals, timesteps, noise = self.sqnb.forward_diffusion(
                gt_boxes, num_frames=T
            )
            # noisy_proposals: (T, P, 4) xyxy
            # timesteps: (T,) per-frame timesteps

            # ── Step 4: Decoder Stage 1 ───────────────────────────────
            # Extract FPN features for this batch element
            batch_fpn = {}
            for key, feat in fpn_features.items():
                batch_fpn[key] = feat[b * T : (b + 1) * T]

            batch_image_sizes = image_sizes[b * T : (b + 1) * T]

            stage1_outputs = self.decoder(
                batch_fpn, noisy_proposals, batch_image_sizes, timesteps
            )

            # ── Step 5: STFS ──────────────────────────────────────────
            # Get last layer predictions for STFS matching (detached —
            # STFS matching is non-differentiable, no need to keep grad)
            last_output = stage1_outputs[-1]
            frame_predictions = []
            for t in range(T):
                cls_probs = last_output["cls_logits"][t].detach().sigmoid()
                scores, labels = cls_probs.max(dim=-1)
                frame_predictions.append({
                    "boxes": last_output["box_pred"][t].detach(),
                    "scores": scores,
                    "labels": labels,
                })

            # Run STFS (no grad needed — only produces voted count)
            with torch.no_grad():
                h_tp, h_fn, h_fp, corrected_rois = self.stfs(
                    batch_fpn, frame_predictions, batch_image_sizes, T
                )

            # Compute voted count for consistency loss
            voted_count = len(h_tp) + len(h_fn)  # expected number of real objects

            # Stage 1 + consistency loss
            stage1_with_consistency = self.criterion(
                stage1_outputs, gt_boxes, gt_labels,
                voted_count=float(voted_count),
            )

            # Accumulate losses
            for key in all_losses:
                all_losses[key] = all_losses[key] + stage1_with_consistency[key]

        # Average over batch
        for key in all_losses:
            all_losses[key] = all_losses[key] / B

        return all_losses

    @torch.no_grad()
    def _forward_inference(
        self,
        fpn_features: dict,
        image_sizes: list[tuple[int, int]],
        B: int,
        T: int,
        device: torch.device,
    ) -> list[list[dict]]:
        """Inference forward pass with NMS and confidence filtering."""
        all_results = []

        for b in range(B):
            # ── Step 3: SQNB (Poisson noise from center) ──────────────
            proposals = self.sqnb.generate_proposals(T, device)
            # proposals: (T, P, 4) xyxy

            # Extract batch FPN features
            batch_fpn = {}
            for key, feat in fpn_features.items():
                batch_fpn[key] = feat[b * T : (b + 1) * T]

            batch_image_sizes = image_sizes[b * T : (b + 1) * T]

            # ── Step 4: Decoder Stage 1 ───────────────────────────────
            stage1_outputs = self.decoder(
                batch_fpn, proposals, batch_image_sizes
            )

            # Get final layer predictions
            last_output = stage1_outputs[-1]
            frame_predictions = []
            for t in range(T):
                cls_probs = last_output["cls_logits"][t].sigmoid()
                scores, labels = cls_probs.max(dim=-1)
                frame_predictions.append({
                    "boxes": last_output["box_pred"][t],
                    "scores": scores,
                    "labels": labels,
                })

            # ── Step 5: STFS ──────────────────────────────────────────
            h_tp, h_fn, h_fp, corrected_rois = self.stfs(
                batch_fpn, frame_predictions, batch_image_sizes, T
            )

            # ── Step 8: Post-process per frame ────────────────────────
            batch_results = []
            for t in range(T):
                boxes = last_output["box_pred"][t]      # (P, 4)
                cls_probs = last_output["cls_logits"][t].sigmoid()  # (P, C)
                scores, labels = cls_probs.max(dim=-1)

                # Confidence filter
                keep = scores > self.cfg.score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # NMS
                if len(boxes) > 0:
                    nms_keep = nms(boxes, scores, self.cfg.nms_thresh)
                    boxes = boxes[nms_keep]
                    scores = scores[nms_keep]
                    labels = labels[nms_keep]

                batch_results.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })

            all_results.append(batch_results)

        return all_results

    def init_weights(self):
        """Initialize custom layers (backbone is already ImageNet-pretrained)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
