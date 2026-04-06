"""Sequence Consistency Alignment (SCA) post-processing.

Four-step algorithm to filter false positives and interpolate
missing detections across a temporal sequence:

1. IoU clustering between adjacent frames
2. Distance + SSIM fallback clustering for fast-moving arteries
3. Temporal filtering (keep lesions appearing in >= T_frame frames)
4. Linear interpolation of missing frames + centerline snapping
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

from .centerline import extract_centerline, snap_bbox_center_to_centerline
from .config import Config


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-8)


def box_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Euclidean distance between box centers."""
    cx_a = (box_a[0] + box_a[2]) / 2
    cy_a = (box_a[1] + box_a[3]) / 2
    cx_b = (box_b[0] + box_b[2]) / 2
    cy_b = (box_b[1] + box_b[3]) / 2
    return np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


def extract_patch(image: np.ndarray, box: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Extract and resize an image patch within a bounding box.

    Args:
        image: (H, W) grayscale image.
        box: [x1, y1, x2, y2] absolute coordinates.
        target_size: Resize patch to this square size for comparison.

    Returns:
        patch: (target_size, target_size) normalized float patch.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_size, target_size), dtype=np.float32)

    patch = image[y1:y2, x1:x2]
    patch = cv2.resize(patch, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    if patch.dtype == np.uint8:
        patch = patch.astype(np.float32) / 255.0
    return patch


def compute_ssim(patch_a: np.ndarray, patch_b: np.ndarray) -> float:
    """Compute SSIM between two patches."""
    if ssim is None:
        raise ImportError("scikit-image is required for SSIM. "
                          "Install with: pip install scikit-image")
    val = ssim(patch_a, patch_b, data_range=1.0)
    return float(val)


# ── Union-Find for clustering ──────────────────────────────────────────


class UnionFind:
    """Simple Union-Find (disjoint set) data structure."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> Dict[int, List[int]]:
        clusters: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            clusters.setdefault(root, []).append(i)
        return clusters


# ── SCA Algorithm ───────────────────────────────────────────────────────


def sca_postprocess(
    detections: List[Dict[str, np.ndarray]],
    images: Optional[List[np.ndarray]] = None,
    cfg: Optional[Config] = None,
    skip_centerline: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Apply Sequence Consistency Alignment to a sequence of detections.

    Args:
        detections: List of T dicts, each with:
            'boxes': (N_t, 4) x1y1x2y2
            'scores': (N_t,)
        images: List of T grayscale images (H, W). Needed for SSIM
                and centerline snapping. If None, SSIM step is skipped.
        cfg: Config with SCA thresholds. Uses defaults if None.
        skip_centerline: If True, skip centerline snapping in step 4.

    Returns:
        filtered: List of T dicts with filtered/interpolated boxes+scores.
    """
    if cfg is None:
        cfg = Config()

    T_iou = cfg.sca_t_iou
    T_distance = cfg.sca_t_distance
    T_sim = cfg.sca_t_sim
    T_frame = cfg.sca_t_frame
    T = len(detections)

    # Build a flat index: (frame_idx, box_idx) → flat_idx
    flat_boxes = []     # (total_dets, 4)
    flat_scores = []    # (total_dets,)
    flat_frame_idx = [] # which frame
    flat_local_idx = [] # which box within that frame

    for t, det in enumerate(detections):
        n = det["boxes"].shape[0]
        for j in range(n):
            flat_boxes.append(det["boxes"][j])
            flat_scores.append(det["scores"][j])
            flat_frame_idx.append(t)
            flat_local_idx.append(j)

    total = len(flat_boxes)
    if total == 0:
        return detections

    flat_boxes = np.array(flat_boxes)
    flat_scores = np.array(flat_scores)
    flat_frame_idx = np.array(flat_frame_idx)

    # ── Step 1 + 2: Clustering ──────────────────────────────────────
    uf = UnionFind(total)

    for t in range(T - 1):
        # Get indices of detections in frame t and t+1
        idx_t = np.where(flat_frame_idx == t)[0]
        idx_t1 = np.where(flat_frame_idx == t + 1)[0]

        for i in idx_t:
            for j in idx_t1:
                box_i = flat_boxes[i]
                box_j = flat_boxes[j]

                # Step 1: IoU clustering
                iou = compute_iou(box_i, box_j)
                if iou > T_iou:
                    uf.union(i, j)
                    continue

                # Step 2: Distance + SSIM fallback
                dist = box_center_distance(box_i, box_j)
                if dist < T_distance and images is not None:
                    patch_i = extract_patch(images[t], box_i)
                    patch_j = extract_patch(images[t + 1], box_j)
                    similarity = compute_ssim(patch_i, patch_j)
                    if similarity > T_sim:
                        uf.union(i, j)

    # ── Step 3: Temporal filtering ──────────────────────────────────
    clusters = uf.groups()
    valid_indices = set()

    for root, members in clusters.items():
        frames_covered = set(flat_frame_idx[m] for m in members)
        if len(frames_covered) >= T_frame:
            valid_indices.update(members)

    # ── Step 4: Interpolation ───────────────────────────────────────
    # Build output per frame — start with valid detections
    output = [{"boxes": np.zeros((0, 4), dtype=np.float32),
               "scores": np.zeros(0, dtype=np.float32)}
              for _ in range(T)]

    # Group valid detections by cluster
    valid_clusters: Dict[int, List[int]] = {}
    for idx in valid_indices:
        root = uf.find(idx)
        valid_clusters.setdefault(root, []).append(idx)

    for root, members in valid_clusters.items():
        # Collect frame → box mapping
        frame_to_box: Dict[int, np.ndarray] = {}
        frame_to_score: Dict[int, float] = {}
        for m in members:
            t = flat_frame_idx[m]
            # Keep highest scoring detection per frame
            if t not in frame_to_score or flat_scores[m] > frame_to_score[t]:
                frame_to_box[t] = flat_boxes[m]
                frame_to_score[t] = flat_scores[m]

        # Find all frames this cluster spans
        all_frames = sorted(frame_to_box.keys())
        if len(all_frames) < 2:
            # Single frame — just include as-is
            for f in all_frames:
                _append_detection(output[f], frame_to_box[f], frame_to_score[f])
            continue

        min_f, max_f = all_frames[0], all_frames[-1]

        # Linear interpolation for missing frames
        # Fit linear regression on x1, y1, x2, y2 over known frames
        known_frames = np.array(all_frames, dtype=np.float64)
        known_boxes = np.array([frame_to_box[f] for f in all_frames])

        for t in range(min_f, max_f + 1):
            if t in frame_to_box:
                box = frame_to_box[t]
                score = frame_to_score[t]
            else:
                # Interpolate
                box = np.zeros(4, dtype=np.float64)
                for coord in range(4):
                    coeffs = np.polyfit(known_frames, known_boxes[:, coord], deg=1)
                    box[coord] = np.polyval(coeffs, t)
                box = box.astype(np.float32)
                score = np.mean(list(frame_to_score.values()))

            # Centerline snapping (if images available)
            if not skip_centerline and images is not None:
                try:
                    skeleton = extract_centerline(
                        images[t],
                        scale_range=cfg.frangi_scale_range,
                        scale_step=cfg.frangi_scale_step,
                        threshold=cfg.frangi_threshold,
                    )
                    box = snap_bbox_center_to_centerline(box, skeleton)
                except Exception:
                    pass  # Skip snapping on failure

            _append_detection(output[t], box, score)

    return output


def _append_detection(
    det: Dict[str, np.ndarray], box: np.ndarray, score: float
):
    """Append a single detection to a frame's detection dict."""
    box = np.atleast_2d(box).astype(np.float32)
    score_arr = np.array([score], dtype=np.float32)
    if det["boxes"].shape[0] == 0:
        det["boxes"] = box
        det["scores"] = score_arr
    else:
        det["boxes"] = np.vstack([det["boxes"], box])
        det["scores"] = np.concatenate([det["scores"], score_arr])
