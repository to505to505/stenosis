"""PSSTT (Proposal-Shifted Spatio-Temporal Transformer) detector.

Implementation of:
    Han et al., "Coronary artery stenosis detection via proposal-shifted
    spatial-temporal transformer in X-ray angiography",
    Computers in Biology and Medicine 153 (2023) 106546.

We build on torchvision's `fasterrcnn_resnet50_fpn_v2` (COCO pretrained) for
the backbone + RPN, and replace the box head with a custom
:class:`PSSTTBoxHead` that:

  1. takes per-frame proposals,
  2. for each proposal builds K+1=5 shifted RoIs (zero + up/down/left/right),
  3. pools those same RoIs from every one of the T frames' FPN maps,
  4. runs a TFA Transformer encoder on the T*(K+1) tokens,
  5. fuses them with a 1x1 Conv1d to a single per-proposal feature,
  6. feeds that to a Fast R-CNN cls + bbox-reg head.

For training we mirror torchvision's RoIHeads sampling/box-coder/loss logic
(reused directly via the official helpers) so the box head matches the
classical Faster R-CNN training recipe.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection._utils import (
    BalancedPositiveNegativeSampler,
    BoxCoder,
    Matcher,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from .config import Config


# ────────────────────────── shift vectors ──────────────────────────────
# Zero-shift first (index 0), then K=4 directions in the order
# (up, down, left, right). αk = (αk,1 in x, αk,2 in y) per Eq. (1).
SHIFT_VECTORS_DEFAULT: Tuple[Tuple[int, int], ...] = (
    (0, 0),     # zero-shift
    (0, -1),    # up
    (0, 1),     # down
    (-1, 0),    # left
    (1, 0),     # right
)


# ─────────────────────────── TFA encoder ───────────────────────────────
class TFALayer(nn.Module):
    """Pre-norm Transformer encoder layer (LN → MSA → +x → LN → MLP → +x)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TFAEncoder(nn.Module):
    """L-layer Transformer encoder over a fixed-length T*(K+1) token sequence."""

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([
            TFALayer(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, num_tokens, D)
        x = x + self.pos
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ────────────────────────── PSSTT box head ─────────────────────────────
class PSSTTBoxHead(nn.Module):
    """Proposal-shifted spatio-temporal tokenization + TFA + feature fusion.

    Operates on a per-(b, t0) basis: for each reference frame's proposals,
    pools shifted RoIs against all T frames of that batch element and
    aggregates them with TFA.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.T = cfg.T
        self.K = cfg.num_shifts
        self.C = cfg.backbone_out_channels
        self.wr = cfg.roi_size
        self.hr = cfg.roi_size
        self.D = cfg.token_dim
        num_tokens = self.T * (self.K + 1)
        wrhrC = self.wr * self.hr * self.C

        self.tokenize = nn.Linear(wrhrC, self.D)
        self.encoder = TFAEncoder(
            num_tokens=num_tokens,
            dim=self.D,
            depth=cfg.tfa_depth,
            heads=cfg.tfa_heads,
            mlp_ratio=cfg.tfa_mlp_ratio,
            dropout=cfg.tfa_dropout,
        )
        # Fusion conv: along the token axis (T*(K+1) → 1), preserving D.
        self.fusion = nn.Conv1d(num_tokens, 1, kernel_size=1)
        self.proj_back = nn.Linear(self.D, wrhrC)

        # Multi-scale RoI align — same scales as torchvision FRCNN default.
        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=cfg.roi_size,
            sampling_ratio=2,
        )

        self.register_buffer(
            "shift_vectors",
            torch.tensor(SHIFT_VECTORS_DEFAULT, dtype=torch.float32),
            persistent=False,
        )

    @property
    def out_features(self) -> int:
        return self.wr * self.hr * self.C

    def build_shifted_boxes(
        self,
        proposals_xyxy: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Return (S, K+1, 4) shifted boxes per proposal, clipped to image."""
        if proposals_xyxy.numel() == 0:
            return proposals_xyxy.new_zeros((0, self.K + 1, 4))
        S = proposals_xyxy.shape[0]
        x1, y1, x2, y2 = proposals_xyxy.unbind(-1)
        w = (x2 - x1).clamp(min=1.0)
        h = (y2 - y1).clamp(min=1.0)
        # (K+1, 2) shift vectors in (dx, dy)
        shifts = self.shift_vectors.to(proposals_xyxy.device)
        dx = shifts[:, 0]  # (K+1,)
        dy = shifts[:, 1]  # (K+1,)
        # (S, K+1)
        x1s = x1.unsqueeze(-1) + dx.unsqueeze(0) * w.unsqueeze(-1)
        y1s = y1.unsqueeze(-1) + dy.unsqueeze(0) * h.unsqueeze(-1)
        x2s = x2.unsqueeze(-1) + dx.unsqueeze(0) * w.unsqueeze(-1)
        y2s = y2.unsqueeze(-1) + dy.unsqueeze(0) * h.unsqueeze(-1)
        H, W = image_size
        x1s = x1s.clamp(0, W - 1)
        x2s = x2s.clamp(0, W - 1)
        y1s = y1s.clamp(0, H - 1)
        y2s = y2s.clamp(0, H - 1)
        # Ensure each box has positive area (degenerate clip → 1px box).
        x2s = torch.maximum(x2s, x1s + 1.0)
        y2s = torch.maximum(y2s, y1s + 1.0)
        return torch.stack([x1s, y1s, x2s, y2s], dim=-1)  # (S, K+1, 4)

    def forward(
        self,
        per_frame_features: List[Dict[str, torch.Tensor]],
        proposals_per_bt0: List[torch.Tensor],
        ref_frame_per_bt0: List[int],
        batch_idx_per_bt0: List[int],
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Aggregate spatio-temporal features for each (b, t0) reference frame.

        Args:
            per_frame_features: list of length T; each entry is a dict mapping
                FPN level name to a (B, C, h_l, w_l) tensor for frame t.
            proposals_per_bt0: list of (S_i, 4) proposals (xyxy, pixel coords)
                indexed by ``i`` over all (b, t0) pairs we want to process.
            ref_frame_per_bt0: list of t0 indices (one per entry above).
            batch_idx_per_bt0: list of b indices (one per entry above).
            image_size: (H, W) of the input frames.

        Returns:
            (sum_i S_i, out_features) flat feature tensor in (b, t0) order.
        """
        T = self.T
        K1 = self.K + 1

        # Concatenate everything across (b, t0) into a single big roi_pool
        # call per support frame for efficiency.
        # Per (b, t0): build shifted boxes -> (S_i, K+1, 4)
        shifted_per_bt0: List[torch.Tensor] = []
        sizes = []
        for proposals in proposals_per_bt0:
            shifted = self.build_shifted_boxes(proposals, image_size)
            shifted_per_bt0.append(shifted)
            sizes.append(shifted.shape[0])  # S_i

        # All shifted boxes flattened to (S_total * (K+1), 4) but grouped by support-frame batch.
        # For RoIAlign we need: per support frame t, a List[Tensor] of length
        # equal to the number of unique batch indices (B), each tensor being
        # the boxes belonging to that batch element. We construct that mapping
        # by reorganising shifted_per_bt0 by batch index.

        # First, get B from features
        any_level = next(iter(per_frame_features[0].values()))
        B = any_level.shape[0]

        # Per batch element b, collect the concatenation of all shifted boxes
        # from every (b, t0) pair where this b appears, along with an index
        # back into the original (b, t0) order.
        per_b_boxes: List[List[torch.Tensor]] = [[] for _ in range(B)]
        per_b_to_bt0: List[List[int]] = [[] for _ in range(B)]
        per_b_counts: List[List[int]] = [[] for _ in range(B)]
        for i, (b, shifted) in enumerate(zip(batch_idx_per_bt0, shifted_per_bt0)):
            per_b_boxes[b].append(shifted.reshape(-1, 4))  # (S_i * (K+1), 4)
            per_b_to_bt0[b].append(i)
            per_b_counts[b].append(shifted.shape[0])  # S_i

        # Per-batch boxes for RoIAlign.
        boxes_per_b = [
            (torch.cat(lst, dim=0) if lst else
             shifted_per_bt0[0].new_zeros((0, 4)))
            for lst in per_b_boxes
        ]
        image_shapes = [image_size] * B

        # Pool per support frame: produces (sum_b N_b, C, wr, hr).
        # Stack across T: (sum_b N_b, T, C, wr, hr).
        pooled_per_t: List[torch.Tensor] = []
        for t in range(T):
            features_t = per_frame_features[t]
            pooled_t = self.roi_pool(features_t, boxes_per_b, image_shapes)
            pooled_per_t.append(pooled_t)
        pooled = torch.stack(pooled_per_t, dim=1)  # (N_total, T, C, wr, hr)

        # Slice back per (b, t0) using per_b_counts/per_b_to_bt0.
        # First flatten to (N_total, T*C*wr*hr) so we can split.
        N_total = pooled.shape[0]
        # We need to map flat indices in pooled back to (b, position-in-b).
        # boxes_per_b was concatenated in the order per_b_boxes[b] = [block_for_bt0_idx_j ...]
        # so for batch b, pooled[offset : offset+sum(per_b_counts[b]*(K+1))] is its block.
        cursor = 0
        out_features = self.out_features

        # We will fill an output of shape (sum_i S_i, out_features) in original
        # (b, t0) order. Compute offsets in the output.
        out_sizes = sizes  # S_i per (b, t0)
        out_offsets = [0]
        for s in out_sizes:
            out_offsets.append(out_offsets[-1] + s)
        S_total = out_offsets[-1]
        device = pooled.device
        dtype = pooled.dtype
        out = pooled.new_zeros((S_total, out_features), dtype=dtype)

        for b in range(B):
            counts = per_b_counts[b]      # list of S_i for this b
            bt0_idxs = per_b_to_bt0[b]    # original (b, t0) indices
            for S_i, bt0_idx in zip(counts, bt0_idxs):
                if S_i == 0:
                    continue
                block_len = S_i * K1
                block = pooled[cursor : cursor + block_len]  # (S_i*K1, T, C, wr, hr)
                cursor += block_len
                # → (S_i, K1, T, C, wr, hr) → permute → (S_i, T, K1, C, wr, hr)
                block = block.reshape(S_i, K1, T, self.C, self.wr, self.hr)
                block = block.permute(0, 2, 1, 3, 4, 5).contiguous()
                # → (S_i, T*K1, C*wr*hr)
                tokens = block.reshape(S_i, T * K1, self.C * self.wr * self.hr)
                # tokenise → (S_i, T*K1, D)
                tokens = self.tokenize(tokens)
                # TFA encoder → (S_i, T*K1, D)
                tokens = self.encoder(tokens)
                # fusion: (S_i, T*K1, D) → conv1d over token axis → (S_i, 1, D)
                fused = self.fusion(tokens).squeeze(1)  # (S_i, D)
                # project back to (S_i, C*wr*hr)
                aggregated = self.proj_back(fused)
                # write into output in (b, t0) order
                lo = out_offsets[bt0_idx]
                hi = out_offsets[bt0_idx + 1]
                out[lo:hi] = aggregated
        return out  # (sum_i S_i, C*wr*hr)


# ───────────────────────── Video Faster R-CNN ──────────────────────────
class VideoFasterRCNN(nn.Module):
    """End-to-end PSSTT detector.

    Wraps torchvision's Faster R-CNN parts (backbone, RPN, box-coder /
    matcher / sampler) and provides a video-aware forward that:

      • shares backbone + RPN across frames,
      • runs the PSSTT head per reference frame.

    Forward signature mirrors :class:`rfdetr_video.model.VideoRFDETR` for
    convenience but accepts only one mode (no distillation, no general
    queries).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.num_classes_with_bg = cfg.num_classes + 1  # +bg

        weights = (
            FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if cfg.pretrained_coco
            else None
        )
        # Build a fresh FasterRCNN (we will replace the box head). We use
        # num_classes=91 to match the COCO checkpoint shape during weight
        # loading, then overwrite head / predictor afterwards.
        frcnn = fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,  # weights includes the backbone already
            rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
            rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
            rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
            box_score_thresh=cfg.box_score_thresh,
            box_nms_thresh=cfg.box_nms_thresh,
            box_detections_per_img=cfg.box_detections_per_img,
            box_fg_iou_thresh=cfg.box_fg_iou_thresh,
            box_bg_iou_thresh=cfg.box_bg_iou_thresh,
            box_batch_size_per_image=cfg.box_batch_size_per_image,
            box_positive_fraction=cfg.box_positive_fraction,
        )
        self.backbone = frcnn.backbone
        self.rpn = frcnn.rpn
        # The torchvision transform normalises pixels and resizes. We feed
        # already-normalised, fixed-size tensors from the dataset, so the
        # transform is a no-op for us; we still keep it around so RPN /
        # MultiScaleRoIAlign get image_sizes via ImageList.
        self.tv_transform = frcnn.transform

        # Borrow proposal-assignment + fg/bg sampling + box-coder from RoIHeads.
        rh = frcnn.roi_heads
        self.box_coder: BoxCoder = rh.box_coder
        self.proposal_matcher: Matcher = rh.proposal_matcher
        self.fg_bg_sampler: BalancedPositiveNegativeSampler = rh.fg_bg_sampler
        self.box_score_thresh = float(rh.score_thresh)
        self.box_nms_thresh = float(rh.nms_thresh)
        self.box_detections_per_img = int(rh.detections_per_img)

        # PSSTT head + predictor.
        self.psstt_head = PSSTTBoxHead(cfg)
        self.box_predictor = FastRCNNPredictor(
            in_channels=self.psstt_head.out_features,
            num_classes=self.num_classes_with_bg,
        )

    # ───────── parameter groups (lower lr on pretrained backbone) ─────
    def get_param_groups(self):
        backbone_params, other_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                other_params.append(p)
        return [
            {"params": backbone_params, "lr": self.cfg.lr * self.cfg.lr_backbone_mult},
            {"params": other_params, "lr": self.cfg.lr},
        ]

    # ──────── torchvision-style training-sample helpers (per frame) ───
    def _assign_targets_to_proposals(
        self,
        proposals: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        matched_idxs, labels = [], []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels
        ):
            if gt_boxes_in_image.numel() == 0:
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device,
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device,
                )
            else:
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_image, proposals_in_image,
                )
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1
            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def _subsample(self, labels: List[torch.Tensor]) -> List[torch.Tensor]:
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds: List[torch.Tensor] = []
        for img_sampled_pos_inds, img_sampled_neg_inds in zip(
            sampled_pos_inds, sampled_neg_inds
        ):
            img_sampled_inds = torch.where(img_sampled_pos_inds | img_sampled_neg_inds)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(
        self,
        proposals: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ):
        """Mirror torchvision's RoIHeads.select_training_samples for one frame.

        Returns
        -------
        proposals     : list[B] of (M_i, 4) sampled proposals (with GT boxes
                        prepended, matching torchvision behaviour)
        labels        : list[B] of (M_i,) cls labels in [0..num_classes]
        reg_targets   : list[B] of (M_i, 4) box-coder regression targets
        """
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        # Append GT boxes to proposals (torchvision standard trick).
        proposals = [
            torch.cat([p, gt_b]) if gt_b.numel() > 0 else p
            for p, gt_b in zip(proposals, gt_boxes)
        ]
        matched_idxs, labels = self._assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels,
        )
        sampled_inds = self._subsample(labels)
        matched_gt_boxes = []
        out_proposals = []
        out_labels = []
        for i, idxs in enumerate(sampled_inds):
            out_proposals.append(proposals[i][idxs])
            out_labels.append(labels[i][idxs])
            if gt_boxes[i].numel() == 0:
                matched_gt_boxes.append(
                    torch.zeros((idxs.shape[0], 4), device=proposals[i].device,
                                dtype=proposals[i].dtype)
                )
            else:
                matched_gt_boxes.append(gt_boxes[i][matched_idxs[i][idxs]])
        reg_targets = self.box_coder.encode(matched_gt_boxes, out_proposals)
        return out_proposals, out_labels, reg_targets

    # ───────── feature extraction ─────────────────────────────────────
    def _backbone_per_frame(
        self,
        images_btchw: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run the backbone+FPN once on the flattened (B*T, 3, H, W) tensor
        and split outputs into a per-frame list of dicts of (B, C, h, w)."""
        B, T = images_btchw.shape[:2]
        x = images_btchw.reshape(B * T, *images_btchw.shape[2:])
        feats = self.backbone(x)  # Dict[str, (B*T, C, h_l, w_l)]
        per_frame: List[Dict[str, torch.Tensor]] = []
        for t in range(T):
            d: Dict[str, torch.Tensor] = {}
            for k, v in feats.items():
                vv = v.reshape(B, T, *v.shape[1:])
                d[k] = vv[:, t].contiguous()
            per_frame.append(d)
        return per_frame, feats

    def _image_list(self, images_btchw: torch.Tensor) -> ImageList:
        """Build a single ImageList covering all B*T frames."""
        B, T, _, H, W = images_btchw.shape
        flat = images_btchw.reshape(B * T, 3, H, W)
        sizes = [(H, W)] * (B * T)
        return ImageList(flat, sizes)

    def _postprocess_detections_for_frame(
        self,
        class_logits: torch.Tensor,    # (sum_i S_i, num_classes+1)
        box_regression: torch.Tensor,  # (sum_i S_i, (num_classes+1)*4)
        proposals: List[torch.Tensor], # list[B] of (S_i, 4)
        image_shapes: List[Tuple[int, int]],
    ):
        """Postprocess centre/ref-frame detections — mirror torchvision."""
        device = class_logits.device
        num_classes = self.num_classes_with_bg

        boxes_per_image = [p.shape[0] for p in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes, all_scores, all_labels = [], [], []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes,
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # Remove background; keep per-class boxes.
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            inds = torch.where(scores > self.box_score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = box_ops.batched_nms(boxes, scores, labels, self.box_nms_thresh)
            keep = keep[: self.box_detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels

    # ─────────────────────────── forward ──────────────────────────────
    def forward(
        self,
        images_btchw: torch.Tensor,
        targets: Optional[List[List[Dict[str, torch.Tensor]]]] = None,
    ):
        """Video forward.

        Args:
            images_btchw: (B, T, 3, H, W) tensor of normalised frames.
            targets: list[B] of list[T] of {"boxes": (N,4) xyxy px,
                "labels": (N,) int64 with bg=0}. Required for training.

        Returns:
            train: dict[str, Tensor] of scalar losses
                ('loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg')
            eval: dict with key 'centre' -> list[B] of {'boxes', 'scores', 'labels'}
                  in pixel coordinates of the input frame (centre = T//2). If
                  ``cfg.supervise_all_frames`` is True, also returns
                  'all_frames' -> list[B] of list[T] of the same dict, evaluated
                  on every reference frame.
        """
        cfg = self.cfg
        is_train = self.training and targets is not None
        B, T = images_btchw.shape[:2]
        H, W = images_btchw.shape[3], images_btchw.shape[4]
        device = images_btchw.device

        per_frame_features, feats_flat = self._backbone_per_frame(images_btchw)
        image_list = self._image_list(images_btchw)

        # ── RPN ──
        # torchvision's RPN expects targets flattened to per-image (B*T) too.
        if is_train:
            assert targets is not None
            rpn_targets: List[Dict[str, torch.Tensor]] = []
            for b in range(B):
                for t in range(T):
                    td = targets[b][t]
                    rpn_targets.append({
                        "boxes": td["boxes"].to(device),
                        "labels": td["labels"].to(device),
                    })
        else:
            rpn_targets = None
        proposals_flat, rpn_losses = self.rpn(image_list, feats_flat, rpn_targets)
        # proposals_flat: list of length B*T; each (S, 4) in image-pixel xyxy.

        # ── Decide which reference frames to process ──
        if is_train:
            ref_frames = list(range(T)) if cfg.supervise_all_frames else [T // 2]
        else:
            # Eval: only need centre-frame predictions for the metrics we report.
            # Running PSSTT on all T frames at eval would multiply cost by T
            # for no measurement gain.
            ref_frames = [T // 2]

        # Build (b, t0) work list.
        bt0_proposals: List[torch.Tensor] = []
        bt0_ref: List[int] = []
        bt0_b: List[int] = []
        bt0_targets: List[Dict[str, torch.Tensor]] = []
        for b in range(B):
            for t0 in ref_frames:
                flat_idx = b * T + t0
                bt0_proposals.append(proposals_flat[flat_idx])
                bt0_ref.append(t0)
                bt0_b.append(b)
                if is_train:
                    bt0_targets.append({
                        "boxes": targets[b][t0]["boxes"].to(device),
                        "labels": targets[b][t0]["labels"].to(device),
                    })

        # ── Training-sample selection on the ref-frame proposals only ──
        if is_train:
            sampled_props, sampled_labels, reg_targets = self.select_training_samples(
                bt0_proposals, bt0_targets,
            )
            psstt_proposals = sampled_props
        else:
            psstt_proposals = bt0_proposals
            sampled_labels = None
            reg_targets = None

        # ── Run PSSTT head per (b, t0) ──
        # We feed per-frame features for the *batch* of that (b, t0); since the
        # head batches across all (b, t0) pairs internally, we pass everything
        # together using per_frame_features[t] which is (B, C, h, w) — the head
        # picks the right b via batch_idx_per_bt0.
        psstt_feats = self.psstt_head(
            per_frame_features=per_frame_features,
            proposals_per_bt0=psstt_proposals,
            ref_frame_per_bt0=bt0_ref,
            batch_idx_per_bt0=bt0_b,
            image_size=(H, W),
        )  # (sum_i S_i, C*wr*hr)

        class_logits, box_regression = self.box_predictor(psstt_feats)

        if is_train:
            loss_cls, loss_box = fastrcnn_loss(
                class_logits, box_regression, sampled_labels, reg_targets,
            )
            losses = dict(rpn_losses)
            losses["loss_classifier"] = loss_cls
            losses["loss_box_reg"] = loss_box
            return losses

        # ── Eval: split predictions back per (b, t0) and postprocess ──
        boxes_per = [p.shape[0] for p in psstt_proposals]
        image_shapes = [(H, W)] * len(psstt_proposals)
        all_boxes, all_scores, all_labels = self._postprocess_detections_for_frame(
            class_logits, box_regression, psstt_proposals, image_shapes,
        )

        # Pack back into (b, t0) layout.
        per_bt0: List[Dict[str, torch.Tensor]] = []
        for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
            per_bt0.append({"boxes": boxes, "scores": scores, "labels": labels})

        out: Dict[str, object] = {}
        # By-frame layout: list[B] of list[T_ref] of dict
        per_b_per_t: List[List[Dict[str, torch.Tensor]]] = [
            [{} for _ in ref_frames] for _ in range(B)
        ]
        for i, (b, t0) in enumerate(zip(bt0_b, bt0_ref)):
            ref_idx = ref_frames.index(t0)
            per_b_per_t[b][ref_idx] = per_bt0[i]
        if self.training and cfg.supervise_all_frames and len(ref_frames) == T:
            out["all_frames"] = per_b_per_t
        # Always provide centre frame.
        centre = T // 2
        if centre in ref_frames:
            ci = ref_frames.index(centre)
            out["centre"] = [per_b_per_t[b][ci] for b in range(B)]
        else:
            out["centre"] = [per_b_per_t[b][0] for b in range(B)]
        return out
