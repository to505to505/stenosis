"""Coronary artery centerline extraction using Frangi vesselness + thinning.

Extracts vessel centerlines from X-ray angiography images and provides
point-snapping utilities for the SCA post-processing step.
"""

import numpy as np

try:
    from skimage.filters import frangi
    from skimage.morphology import skeletonize
except ImportError:
    frangi = None
    skeletonize = None

import cv2


def extract_vesselness(
    image: np.ndarray,
    scale_range: tuple = (1, 8),
    scale_step: int = 2,
) -> np.ndarray:
    """Compute Frangi vesselness response.

    Args:
        image: Grayscale image (H, W), uint8 or float.
        scale_range: (min_sigma, max_sigma) for multi-scale analysis.
        scale_step: Step between sigma values.

    Returns:
        vesselness: (H, W) float array in [0, 1].
    """
    if frangi is None:
        raise ImportError("scikit-image is required for centerline extraction. "
                          "Install with: pip install scikit-image")

    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    elif image.dtype != np.float64:
        image = image.astype(np.float64)

    return frangi(image, scale_range=scale_range, scale_step=scale_step)


def extract_centerline(
    image: np.ndarray,
    scale_range: tuple = (1, 8),
    scale_step: int = 2,
    threshold: float = 0.02,
) -> np.ndarray:
    """Extract vessel centerline skeleton from a grayscale image.

    Pipeline:
        1. Frangi vesselness filter (multi-scale)
        2. Threshold to binary vessel mask
        3. Morphological cleanup (close + open)
        4. Skeletonize to 1-pixel wide centerline

    Args:
        image: Grayscale image (H, W).
        scale_range: Frangi sigma range.
        scale_step: Frangi sigma step.
        threshold: Vesselness threshold for binarization.

    Returns:
        skeleton: (H, W) boolean array where True = centerline pixel.
    """
    if skeletonize is None:
        raise ImportError("scikit-image is required for centerline extraction.")

    vesselness = extract_vesselness(image, scale_range, scale_step)

    # Threshold to binary
    binary = (vesselness > threshold).astype(np.uint8)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Skeletonize to centerline
    skeleton = skeletonize(binary.astype(bool))
    return skeleton


def snap_to_centerline(
    point_yx: tuple,
    skeleton: np.ndarray,
    max_distance: float = float("inf"),
) -> tuple:
    """Snap a point (y, x) to the nearest skeleton pixel.

    Args:
        point_yx: (y, x) coordinates of the point to snap.
        skeleton: (H, W) boolean skeleton array.
        max_distance: Maximum allowed snapping distance. If the nearest
                      skeleton pixel is farther, the original point is returned.

    Returns:
        snapped_yx: (y, x) of the nearest skeleton pixel, or original if
                    no skeleton pixel is within max_distance.
    """
    skeleton_points = np.argwhere(skeleton)  # (N, 2) — (row, col) = (y, x)

    if skeleton_points.shape[0] == 0:
        return point_yx

    y, x = point_yx
    distances = np.sqrt(
        (skeleton_points[:, 0] - y) ** 2 +
        (skeleton_points[:, 1] - x) ** 2
    )

    nearest_idx = np.argmin(distances)
    if distances[nearest_idx] <= max_distance:
        return tuple(skeleton_points[nearest_idx])
    return point_yx


def snap_bbox_center_to_centerline(
    bbox: np.ndarray,
    skeleton: np.ndarray,
    max_distance: float = float("inf"),
) -> np.ndarray:
    """Snap a bounding box center to the nearest centerline point.

    Translates the entire bounding box so its center lies on the skeleton.

    Args:
        bbox: (4,) array [x1, y1, x2, y2].
        skeleton: (H, W) boolean skeleton.
        max_distance: Max snapping distance.

    Returns:
        snapped_bbox: (4,) translated bbox.
    """
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2

    snapped_y, snapped_x = snap_to_centerline((cy, cx), skeleton, max_distance)

    dx = snapped_x - cx
    dy = snapped_y - cy

    return np.array([
        bbox[0] + dx, bbox[1] + dy,
        bbox[2] + dx, bbox[3] + dy,
    ], dtype=bbox.dtype)
