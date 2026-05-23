"""Sequence Consistency Alignment (paper §2.4).

Eval-time post-processing that consumes per-frame detections from a
T-frame window and:

  1. Clusters detections across frames by IoU > ``T_iou`` (Eq. before
     Fig. 5). For pairs whose IoU falls below ``T_iou`` but whose centre
     distance is below ``T_distance`` we fall back to a structural-
     similarity (SSIM) check on the image patches; pairs with SSIM > T_sim
     are merged into the same cluster.
  2. Drops every cluster present in fewer than ``T_frame`` frames as a
     false positive.
  3. For surviving clusters, linearly interpolates the top-left and
     bottom-right corners across missing frames (paper's "interpolation
     completion"). We omit the centerline-projection step because reliable
     coronary-centerline extraction from raw frames is out of scope; linear
     interpolation of the corners is the documented procedure for boxes
     near the structure already represented in adjacent frames.

The module operates on numpy / python data — there is no learnable
parameter in SCA. It is invoked once per sliding-window-aware test pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────── geometry helpers ──────────────────────────
def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _center(box: np.ndarray) -> Tuple[float, float]:
    return float(0.5 * (box[0] + box[2])), float(0.5 * (box[1] + box[3]))


def _center_distance(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return float(np.hypot(ax - bx, ay - by))


# ─────────────────────────── SSIM (luminance) ──────────────────────────
def _crop_patch(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Crop an xyxy box from a (H, W) or (H, W, C) image; clip to bounds."""
    H, W = img.shape[:2]
    x1 = int(np.floor(max(0.0, box[0])))
    y1 = int(np.floor(max(0.0, box[1])))
    x2 = int(np.ceil(min(W - 1.0, box[2])))
    y2 = int(np.ceil(min(H - 1.0, box[3])))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return img[y1:y2, x1:x2]


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0].astype(np.float64)
    # Standard ITU-R BT.601 luminance.
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(
        np.float64,
    )


def _resize_to(arr: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize (no external deps). Avoids the SciPy/OpenCV dep just for SSIM."""
    H, W = arr.shape[:2]
    th, tw = hw
    if H == th and W == tw:
        return arr
    if H == 0 or W == 0 or th == 0 or tw == 0:
        return np.zeros((th, tw), dtype=arr.dtype)
    # Use coordinates in source space.
    ys = np.linspace(0, H - 1, th)
    xs = np.linspace(0, W - 1, tw)
    y0 = np.floor(ys).astype(np.int64); y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(xs).astype(np.int64); x1 = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0).reshape(-1, 1)
    wx = (xs - x0).reshape(1, -1)
    Ia = arr[np.ix_(y0, x0)]
    Ib = arr[np.ix_(y0, x1)]
    Ic = arr[np.ix_(y1, x0)]
    Id = arr[np.ix_(y1, x1)]
    return (Ia * (1 - wy) * (1 - wx) + Ib * (1 - wy) * wx
            + Ic * wy * (1 - wx) + Id * wy * wx)


def patch_ssim(img_a: np.ndarray, box_a: np.ndarray,
               img_b: np.ndarray, box_b: np.ndarray) -> float:
    """SSIM between two image patches (Wang et al. 2004, single-window).

    We use a luminance-only SSIM on equal-sized resized patches, which is
    sufficient for the cross-frame consistency check the paper describes.
    """
    pa = _to_gray(_crop_patch(img_a, box_a))
    pb = _to_gray(_crop_patch(img_b, box_b))
    h = max(min(pa.shape[0], pb.shape[0]), 4)
    w = max(min(pa.shape[1], pb.shape[1]), 4)
    pa = _resize_to(pa, (h, w))
    pb = _resize_to(pb, (h, w))
    # Normalise to [0, 1] if patches are in 0..255.
    if pa.max() > 1.5 or pb.max() > 1.5:
        pa = pa / 255.0
        pb = pb / 255.0
    mu_a = pa.mean(); mu_b = pb.mean()
    va = pa.var(); vb = pb.var()
    cov = ((pa - mu_a) * (pb - mu_b)).mean()
    L = 1.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, -1.0, 1.0))


# ─────────────────────────── SCA datatypes ─────────────────────────────
@dataclass
class FrameDetections:
    """A frame's detections from a single sliding-window output."""

    boxes: np.ndarray         # (N, 4) xyxy pixel
    scores: np.ndarray        # (N,)
    image: Optional[np.ndarray] = None    # (H, W) or (H, W, 3); needed for SSIM fallback


@dataclass
class SCAConfig:
    t_iou: float = 0.2
    t_frame: int = 3
    t_distance: float = 50.0
    t_sim: float = 0.5
    interpolate_missing: bool = True


# ─────────────────────────── core algorithm ────────────────────────────
def _pair_same_cluster(
    a_box: np.ndarray,
    b_box: np.ndarray,
    a_img: Optional[np.ndarray],
    b_img: Optional[np.ndarray],
    cfg: SCAConfig,
) -> bool:
    """Return True iff two detections in adjacent frames should be merged."""
    iou = _iou_xyxy(a_box, b_box)
    if iou > cfg.t_iou:
        return True
    if a_img is None or b_img is None:
        return False
    if _center_distance(a_box, b_box) >= cfg.t_distance:
        return False
    sim = patch_ssim(a_img, a_box, b_img, b_box)
    return sim > cfg.t_sim


def apply_sca(
    per_frame: List[FrameDetections],
    cfg: SCAConfig,
) -> List[FrameDetections]:
    """Apply Sequence Consistency Alignment to a single window's detections.

    Args:
        per_frame: list of length T — one ``FrameDetections`` per frame.
        cfg: SCA thresholds.

    Returns:
        New list of length T with filtered + interpolation-completed
        detections per frame.
    """
    T = len(per_frame)
    if T == 0:
        return per_frame

    # Build node list: (frame_idx, det_idx).
    nodes: List[Tuple[int, int]] = []
    for t, fd in enumerate(per_frame):
        for j in range(int(fd.boxes.shape[0])):
            nodes.append((t, j))
    n_nodes = len(nodes)
    if n_nodes == 0:
        return [FrameDetections(boxes=np.zeros((0, 4), dtype=np.float32),
                                scores=np.zeros((0,), dtype=np.float32),
                                image=fd.image) for fd in per_frame]

    # Union-Find over nodes. We only attempt to merge detections in
    # **adjacent** frames (matching the paper's "two boxes in adjacent
    # frame image" criterion). Two non-adjacent detections can still end
    # up in the same cluster via transitive merging.
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Index nodes per frame for fast adjacency walks.
    nodes_in_frame: List[List[int]] = [[] for _ in range(T)]
    for node_idx, (t, j) in enumerate(nodes):
        nodes_in_frame[t].append(node_idx)

    for t in range(T - 1):
        for ai in nodes_in_frame[t]:
            ta, ja = nodes[ai]
            a_box = per_frame[ta].boxes[ja]
            for bi in nodes_in_frame[t + 1]:
                tb, jb = nodes[bi]
                b_box = per_frame[tb].boxes[jb]
                if _pair_same_cluster(
                    a_box, b_box,
                    per_frame[ta].image, per_frame[tb].image, cfg,
                ):
                    union(ai, bi)

    # Group node indices by cluster root.
    clusters: Dict[int, List[int]] = {}
    for node_idx in range(n_nodes):
        r = find(node_idx)
        clusters.setdefault(r, []).append(node_idx)

    # Filter clusters by T_frame.
    kept_clusters: List[List[int]] = []
    for cluster_nodes in clusters.values():
        frames_present = {nodes[ni][0] for ni in cluster_nodes}
        if len(frames_present) >= cfg.t_frame:
            kept_clusters.append(cluster_nodes)

    # Build per-frame output: original boxes that survive, plus
    # interpolated boxes for missing frames inside kept clusters.
    out_boxes: List[List[np.ndarray]] = [[] for _ in range(T)]
    out_scores: List[List[float]] = [[] for _ in range(T)]
    out_origin: List[List[str]] = [[] for _ in range(T)]   # "kept" | "interp"

    for cluster_nodes in kept_clusters:
        # Sort by frame index.
        per_t_members: Dict[int, List[int]] = {}
        for ni in cluster_nodes:
            t, j = nodes[ni]
            per_t_members.setdefault(t, []).append(j)

        # Aggregate to one representative box per frame (mean of corners,
        # max score) for interpolation purposes. Surviving detections in
        # the cluster are emitted as-is.
        cluster_frame_box: Dict[int, np.ndarray] = {}
        for t, jlist in per_t_members.items():
            boxes_t = per_frame[t].boxes[jlist]
            scores_t = per_frame[t].scores[jlist]
            cluster_frame_box[t] = boxes_t.mean(axis=0)
            for j_idx, j in enumerate(jlist):
                out_boxes[t].append(per_frame[t].boxes[j])
                out_scores[t].append(float(scores_t[j_idx]))
                out_origin[t].append("kept")

        if not cfg.interpolate_missing:
            continue
        present_ts = sorted(cluster_frame_box.keys())
        if len(present_ts) < 2:
            continue
        # Build piecewise-linear interpolation across the present frames.
        ts_arr = np.array(present_ts, dtype=np.float32)
        corners = np.stack([cluster_frame_box[t] for t in present_ts], axis=0)  # (P, 4)
        for t_missing in range(present_ts[0], present_ts[-1] + 1):
            if t_missing in per_t_members:
                continue
            interp = np.empty(4, dtype=np.float32)
            for k in range(4):
                interp[k] = np.interp(float(t_missing), ts_arr, corners[:, k])
            # Interpolated boxes inherit the median cluster score as a
            # conservative confidence estimate.
            interp_score = float(np.median([np.max(per_frame[t].scores[jlist])
                                            for t, jlist in per_t_members.items()]))
            out_boxes[t_missing].append(interp)
            out_scores[t_missing].append(interp_score)
            out_origin[t_missing].append("interp")

    result: List[FrameDetections] = []
    for t in range(T):
        if len(out_boxes[t]) == 0:
            result.append(FrameDetections(
                boxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                image=per_frame[t].image,
            ))
        else:
            result.append(FrameDetections(
                boxes=np.stack(out_boxes[t], axis=0).astype(np.float32),
                scores=np.array(out_scores[t], dtype=np.float32),
                image=per_frame[t].image,
            ))
    return result
