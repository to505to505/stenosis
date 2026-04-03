import os
import logging
import warnings
import numpy as np
import cv2
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
from cupyx.scipy.ndimage import label as gpu_label
from PIL import Image


warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.ERROR)


def load_image(path: str) -> np.ndarray:
    """
    Load an image as a grayscale numpy array.
    """
    img = Image.open(path).convert('L')
    return np.array(img)


def save_image(img: np.ndarray, path: str) -> None:
    """
    Save a numpy array as an image.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def sato_filter(image: np.ndarray, sigmas: list = [1, 2, 3, 4], border: int = 3) -> np.ndarray:
    """
    Apply Sato Hessian-based vesselness filter on GPU and zero out borders.
    Reimplements skimage.filters.sato(black_ridges=True, mode='reflect') via CuPy.
    """
    img_gpu = cp.asarray(image.astype(np.float64))
    filtered_max = cp.zeros_like(img_gpu)

    for sigma in sigmas:
        # Hessian matrix elements (scale-normalized)
        Hxx = gpu_gaussian_filter(img_gpu, sigma=sigma, order=[2, 0], mode='reflect') * (sigma ** 2)
        Hxy = gpu_gaussian_filter(img_gpu, sigma=sigma, order=[1, 1], mode='reflect') * (sigma ** 2)
        Hyy = gpu_gaussian_filter(img_gpu, sigma=sigma, order=[0, 2], mode='reflect') * (sigma ** 2)

        # Eigenvalues of 2x2 symmetric matrix [[Hxx, Hxy],[Hxy, Hyy]]
        tmp = cp.sqrt(((Hxx - Hyy) / 2.0) ** 2 + Hxy ** 2)
        mean_val = (Hxx + Hyy) / 2.0
        l1 = mean_val - tmp
        l2 = mean_val + tmp

        # Pick eigenvalue with largest absolute value
        e_large = cp.where(cp.abs(l1) > cp.abs(l2), l1, l2)

        # black_ridges=True: negate
        e_large = -e_large

        # Sato tubeness: max(0, largest eigenvalue)
        filtered = cp.maximum(e_large, 0)
        filtered_max = cp.maximum(filtered_max, filtered)

    result = cp.asnumpy(filtered_max).astype(np.uint8)
    # mask borders
    h, w = result.shape
    result[:border, :] = 0
    result[-border:, :] = 0
    result[:, :border] = 0
    result[:, -border:] = 0
    return result


def threshold_image(image: np.ndarray, percentile: float = 92.0) -> np.ndarray:
    """
    Threshold image by percentile on GPU, zeroing values below threshold.
    """
    img_gpu = cp.asarray(image)
    thresh_val = cp.percentile(img_gpu, percentile)
    mask = cp.where(img_gpu >= thresh_val, img_gpu, cp.zeros_like(img_gpu))
    return cp.asnumpy(mask)


def region_grow(img: np.ndarray, seed: tuple = None) -> np.ndarray:
    """
    Find the connected component of nonzero pixels containing the seed (or
    the brightest pixel if seed is None) using GPU-accelerated connected
    components labeling.  Equivalent to the original flood-fill region grow.
    """
    img_gpu = cp.asarray(img)
    binary = (img_gpu > 0).astype(cp.int32)
    labeled, num_features = gpu_label(binary)

    if num_features == 0:
        return np.zeros_like(img, dtype=np.uint8)

    if seed is not None:
        seed_label = int(labeled[seed[0], seed[1]])
    else:
        if cp.any(img_gpu):
            max_idx = int(cp.argmax(img_gpu))
            seed_label = int(labeled.ravel()[max_idx])
        else:
            return np.zeros_like(img, dtype=np.uint8)

    if seed_label == 0:
        return np.zeros_like(img, dtype=np.uint8)

    mask = (labeled == seed_label).astype(cp.uint8) * 255
    return cp.asnumpy(mask)


def segment_image(input_path: str,
                  output_path: str,
                  sato_sigmas: list = [1,2,3,4],
                  border: int = 3,
                  threshold_pct: float = 92.0) -> None:
    """
    Complete pipeline: load image, apply Sato filter, threshold, region grow, and save mask.
    """
    img = load_image(input_path)
    vesselness = sato_filter(img, sigmas=sato_sigmas, border=border)
    thresh = threshold_image(vesselness, percentile=threshold_pct)
    mask = region_grow(thresh)
    save_image(mask, output_path)


if __name__ == "__main__":
    border = 20
    percentile = 92.0 # default
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    for dataset in ["syntax", "cadica", "xcad", "coronarydominance"]:
        image_folder = os.path.join(base_dir, dataset)
        mask_folder = os.path.join(base_dir, f"{dataset}_frangi")

        os.makedirs(mask_folder, exist_ok=True)

        image_items = [f for f in os.listdir(image_folder) if f.endswith('.png')]

        from tqdm import tqdm
        iterator = tqdm(image_items, desc=f"Processing {dataset}")

        for item in iterator:
            input_path = os.path.join(image_folder, item)
            output_path = os.path.join(mask_folder, item)
            
            segment_image(input_path, output_path, border=border, threshold_pct=percentile)