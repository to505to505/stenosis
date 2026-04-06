"""Spatio-Temporal Feature Sharing (STFS) Module.

Feedback loop to correct False Positives (FP) and False Negatives (FN)
across N frames using:
  1. Hungarian matching of candidate boxes across frames.
  2. Voting system: H-TP / H-FN / H-FP classification.
  3. RoI padding & feature extraction for error frames.
  4. RoI Aggregator: MHA → LN → DynConv → LN → Linear → Residual.
  5. Second-stage decoder pass for refined predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, roi_align
import numpy as np

from ..config import Config
from .gfe import DynamicConv


class RoIAggregator(nn.Module):
    """Aggregates wrong RoI features using reference (right) RoI features.

    Architecture:
      MHA(Q=wrong, K=right, V=right) → LayerNorm → DynamicConv → LayerNorm
      → FC2(ReLU(FC1(...))) → Residual Add

    Args:
        feat_dim: Feature dimension (C * roi * roi flattened or C for spatial).
        channels: Spatial feature channels (C).
        spatial_size: RoI spatial size (roi_output_size).
        num_heads: MHA heads.
        ffn_dim: Linear block hidden dimension.
    """

    def __init__(
        self,
        channels: int,
        spatial_size: int,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        feat_dim = channels * spatial_size * spatial_size

        # MHA: wrong features attend to right features
        self.mha = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feat_dim)

        # Dynamic Convolution (operates in spatial domain)
        self.dynamic_conv = DynamicConv(
            channels=channels, kernel_size=3, groups=4
        )
        self.norm2 = nn.LayerNorm([channels, spatial_size, spatial_size])

        # Linear block: FC2(ReLU(FC1(...)))
        self.linear_block = nn.Sequential(
            nn.Linear(feat_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, feat_dim),
        )
        self.norm3 = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        wrong_features: torch.Tensor,
        right_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            wrong_features: (K, C, roi, roi) features from H-FN/H-FP frames.
            right_features: (K, C, roi, roi) reference features from positive frames.

        Returns:
            aggregated: (K, C, roi, roi) corrected features.
        """
        K, C, rh, rw = wrong_features.shape
        feat_dim = C * rh * rw

        # Flatten for MHA
        wrong_flat = wrong_features.reshape(K, 1, feat_dim)   # (K, 1, feat_dim)
        right_flat = right_features.reshape(K, 1, feat_dim)   # (K, 1, feat_dim)

        # MHA: Q=wrong, K=right, V=right
        attn_out, _ = self.mha(wrong_flat, right_flat, right_flat)
        attn_out = self.norm1(attn_out + wrong_flat)  # residual
        attn_out = attn_out.squeeze(1)  # (K, feat_dim)

        # Reshape to spatial for Dynamic Conv
        spatial = attn_out.reshape(K, C, rh, rw)
        dc_out = self.dynamic_conv(spatial)
        dc_out = self.norm2(dc_out + spatial)  # residual

        # Linear block
        dc_flat = dc_out.reshape(K, feat_dim)
        linear_out = self.linear_block(dc_flat)
        linear_out = self.dropout(linear_out)
        aggregated = self.norm3(linear_out + dc_flat)  # residual

        return aggregated.reshape(K, C, rh, rw)


def hungarian_match_across_frames(
    predictions: list[dict],
    iou_weight: float = 1.0,
    dist_weight: float = 0.5,
    score_thresh: float = 0.3,
) -> list[dict]:
    """Match predicted boxes across N frames using Hungarian algorithm.

    Args:
        predictions: List of N dicts, each with:
            "boxes": (M_n, 4) predicted boxes in xyxy.
            "scores": (M_n,) confidence scores.
            "labels": (M_n,) predicted class labels.
        iou_weight: Weight for IoU in cost matrix.
        dist_weight: Weight for Manhattan distance in cost matrix.
        score_thresh: Minimum score to consider a prediction.

    Returns:
        groups: List of dicts, each with:
            "frame_indices": list of frame indices where this group appears.
            "boxes": dict mapping frame_idx → box tensor (4,).
            "n_boxes": number of frames with this detection.
            "ref_box": representative box (from highest confidence frame).
    """
    from scipy.optimize import linear_sum_assignment

    N = len(predictions)

    # Filter by confidence threshold
    filtered = []
    for n, pred in enumerate(predictions):
        mask = pred["scores"] >= score_thresh
        filtered.append({
            "boxes": pred["boxes"][mask],
            "scores": pred["scores"][mask],
            "labels": pred["labels"][mask],
            "frame_idx": n,
        })

    # Sequential frame-to-frame matching
    # Start with frame 0's detections as initial groups
    groups = []
    if len(filtered[0]["boxes"]) > 0:
        for i in range(len(filtered[0]["boxes"])):
            groups.append({
                "frame_indices": [0],
                "boxes": {0: filtered[0]["boxes"][i]},
                "scores": {0: filtered[0]["scores"][i].item()},
                "n_boxes": 1,
            })

    # Match subsequent frames to existing groups
    for n in range(1, N):
        curr_boxes = filtered[n]["boxes"]
        curr_scores = filtered[n]["scores"]

        if len(curr_boxes) == 0 or len(groups) == 0:
            # Add unmatched current boxes as new groups
            for i in range(len(curr_boxes)):
                groups.append({
                    "frame_indices": [n],
                    "boxes": {n: curr_boxes[i]},
                    "scores": {n: curr_scores[i].item()},
                    "n_boxes": 1,
                })
            continue

        # Build cost matrix: groups × current detections
        # Use the most recent box from each group as reference
        group_ref_boxes = []
        for g in groups:
            # Use the latest frame's box as reference
            latest_frame = max(g["frame_indices"])
            group_ref_boxes.append(g["boxes"][latest_frame])

        ref_boxes = torch.stack(group_ref_boxes)  # (G, 4)

        # IoU cost (higher is better, so negate for Hungarian)
        iou_matrix = box_iou(ref_boxes, curr_boxes)  # (G, M)

        # Manhattan distance cost (lower is better)
        ref_centers = torch.stack([
            (ref_boxes[:, 0] + ref_boxes[:, 2]) / 2,
            (ref_boxes[:, 1] + ref_boxes[:, 3]) / 2,
        ], dim=-1)
        curr_centers = torch.stack([
            (curr_boxes[:, 0] + curr_boxes[:, 2]) / 2,
            (curr_boxes[:, 1] + curr_boxes[:, 3]) / 2,
        ], dim=-1)

        # Manhattan distance: (G, M)
        dist_matrix = torch.cdist(
            ref_centers.float(), curr_centers.float(), p=1
        )
        # Normalize distance to [0, 1] range
        max_dist = dist_matrix.max().clamp(min=1.0)
        dist_norm = dist_matrix / max_dist

        # Combined cost: negative IoU + distance (minimize)
        cost = -iou_weight * iou_matrix + dist_weight * dist_norm

        # Hungarian matching (replace NaN/Inf with large values)
        cost_np = cost.detach().cpu().numpy()
        cost_np = np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=-1e6)
        row_idx, col_idx = linear_sum_assignment(cost_np)

        matched_curr = set()
        for r, c in zip(row_idx, col_idx):
            # Only match if IoU > 0.1 (reject bad matches)
            if iou_matrix[r, c] > 0.1:
                groups[r]["frame_indices"].append(n)
                groups[r]["boxes"][n] = curr_boxes[c]
                groups[r]["scores"][n] = curr_scores[c].item()
                groups[r]["n_boxes"] += 1
                matched_curr.add(c)

        # Add unmatched current boxes as new groups
        for i in range(len(curr_boxes)):
            if i not in matched_curr:
                groups.append({
                    "frame_indices": [n],
                    "boxes": {n: curr_boxes[i]},
                    "scores": {n: curr_scores[i].item()},
                    "n_boxes": 1,
                })

    # Set reference box (highest confidence frame)
    for g in groups:
        best_frame = max(g["scores"], key=g["scores"].get)
        g["ref_box"] = g["boxes"][best_frame]
        g["ref_frame"] = best_frame

    return groups


def vote_groups(
    groups: list[dict], num_frames: int
) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify groups into H-TP, H-FN, H-FP by voting.

    Args:
        groups: Output from hungarian_match_across_frames.
        num_frames: N, total number of frames.

    Returns:
        h_tp: Groups with n_boxes == N (true positive).
        h_fn: Groups with N/2 <= n_boxes < N (false negative in some frames).
        h_fp: Groups with n_boxes < N/2 (false positive, spurious).
    """
    N = num_frames
    half_N = N / 2

    h_tp, h_fn, h_fp = [], [], []
    for g in groups:
        n = g["n_boxes"]
        if n >= N:
            h_tp.append(g)
        elif n >= half_N:
            h_fn.append(g)
        else:
            h_fp.append(g)

    return h_tp, h_fn, h_fp


class STFSModule(nn.Module):
    """Spatio-Temporal Feature Sharing module.

    Feedback loop: match detections across frames → vote → extract expanded
    RoIs for error frames → aggregate with reference features → re-decode.

    Args:
        cfg: Config with stfs_alpha, C, roi_output_size, stfs_num_heads,
             stfs_ffn_dim, stfs_iou_weight, stfs_dist_weight.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.alpha = cfg.stfs_alpha  # padding coefficient
        self.roi_size = cfg.roi_output_size

        self.aggregator = RoIAggregator(
            channels=cfg.C,
            spatial_size=cfg.roi_output_size,
            num_heads=cfg.stfs_num_heads,
            ffn_dim=cfg.stfs_ffn_dim,
        )

    def forward(
        self,
        fpn_features: dict,
        predictions: list[dict],
        image_sizes: list[tuple[int, int]],
        num_frames: int,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        """
        Args:
            fpn_features: Dict of FPN feature maps, each (sum_N, C, H_i, W_i).
            predictions: List of N dicts with "boxes", "scores", "labels"
                per frame after first-stage decoding.
            image_sizes: List of (H, W) per frame.
            num_frames: N.

        Returns:
            h_tp: True positive groups (kept as-is).
            h_fn: False negative groups with corrected features.
            h_fp: False positive groups with corrected features.
            corrected_rois: Dict mapping (group_idx, frame_idx) → (C, roi, roi)
                aggregated features for second-stage decoding.
        """
        N = num_frames
        device = predictions[0]["boxes"].device

        # Step 1: Hungarian matching + voting
        groups = hungarian_match_across_frames(
            predictions,
            iou_weight=self.cfg.stfs_iou_weight,
            dist_weight=self.cfg.stfs_dist_weight,
        )
        h_tp, h_fn, h_fp = vote_groups(groups, N)

        # Step 2: Extract expanded RoIs for error frames
        corrected_rois = {}

        # Process H-FN groups (missing detections in some frames)
        for g_idx, group in enumerate(h_fn):
            ref_box = group["ref_box"]  # (4,) xyxy from reference frame
            ref_frame = group["ref_frame"]
            present_frames = set(group["frame_indices"])
            missing_frames = [f for f in range(N) if f not in present_frames]

            if len(missing_frames) == 0:
                continue

            # Extract reference RoI features
            ref_roi = self._extract_roi(
                fpn_features, ref_box.unsqueeze(0), ref_frame, image_sizes
            )  # (1, C, roi, roi)

            # For each missing frame, extract expanded RoI
            for mf in missing_frames:
                expanded_box = self._expand_box(ref_box, self.alpha)
                wrong_roi = self._extract_roi(
                    fpn_features, expanded_box.unsqueeze(0), mf, image_sizes
                )  # (1, C, roi, roi)

                # Aggregate: wrong attends to right
                agg = self.aggregator(wrong_roi, ref_roi)  # (1, C, roi, roi)
                corrected_rois[("fn", g_idx, mf)] = agg.squeeze(0)

        # Process H-FP groups (spurious detections)
        for g_idx, group in enumerate(h_fp):
            ref_box = group["ref_box"]
            ref_frame = group["ref_frame"]

            # Extract reference (spurious) features
            ref_roi = self._extract_roi(
                fpn_features, ref_box.unsqueeze(0), ref_frame, image_sizes
            )

            # For frames where this spurious detection appears, extract expanded
            for f_idx in group["frame_indices"]:
                box = group["boxes"][f_idx]
                expanded_box = self._expand_box(box, self.alpha)
                wrong_roi = self._extract_roi(
                    fpn_features, expanded_box.unsqueeze(0), f_idx, image_sizes
                )

                # For FP: use surrounding context to suppress
                # The aggregator helps the model learn to reject these
                agg = self.aggregator(wrong_roi, ref_roi)
                corrected_rois[("fp", g_idx, f_idx)] = agg.squeeze(0)

        return h_tp, h_fn, h_fp, corrected_rois

    def _extract_roi(
        self,
        fpn_features: dict,
        boxes: torch.Tensor,
        frame_idx: int,
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Extract RoI features for specific frame.

        Args:
            fpn_features: FPN feature dict.
            boxes: (K, 4) boxes in xyxy absolute pixels.
            frame_idx: Which frame to extract from.
            image_sizes: List of (H, W).

        Returns:
            roi_features: (K, C, roi, roi) extracted features.
        """
        # Use the highest resolution FPN level for simplicity
        # (stride 4 → "0" key in FPN)
        feat_key = "0"
        feat = fpn_features[feat_key]  # (sum_N, C, H/4, W/4)

        # Extract single frame's feature map
        frame_feat = feat[frame_idx].unsqueeze(0)  # (1, C, H/4, W/4)

        # Scale boxes to feature map resolution
        spatial_scale = frame_feat.shape[-1] / image_sizes[frame_idx][1]

        # roi_align expects list of boxes with batch index
        roi_features = roi_align(
            frame_feat,
            [boxes],
            output_size=self.roi_size,
            spatial_scale=spatial_scale,
            sampling_ratio=2,
        )  # (K, C, roi, roi)

        return roi_features

    def _expand_box(self, box: torch.Tensor, alpha: float) -> torch.Tensor:
        """Expand box by padding coefficient α.

        Args:
            box: (4,) xyxy box.
            alpha: Expansion factor (2.0 means box becomes 2x wider/taller).

        Returns:
            expanded: (4,) expanded xyxy box, clamped to image bounds.
        """
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]

        new_w = w * alpha
        new_h = h * alpha

        expanded = torch.stack([
            cx - new_w / 2,
            cy - new_h / 2,
            cx + new_w / 2,
            cy + new_h / 2,
        ])

        expanded[0::2].clamp_(min=0, max=self.cfg.img_w)
        expanded[1::2].clamp_(min=0, max=self.cfg.img_h)

        return expanded
