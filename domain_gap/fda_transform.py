"""Fourier Domain Adaptation (FDA) — Yang & Soatto, CVPR 2020.

Aligns the low-frequency amplitude spectrum of source images (Dataset A,
``stenosis_arcade``) to that of randomly-sampled target images (Dataset B,
``cadica_50plus_new``). Produces an FDA-aligned mirror of Dataset A on disk:

    data/stenosis_arcade_fda_B/
        train/images/*.png   (FDA-adapted source frames at 512x512)
        train/labels/*.txt   (copied verbatim from the original split)
        val/...
        test/...

CRITICAL: Dataset C (``dataset2_split``) is NEVER read by this script —
it must remain unseen during the adaptation step.

References
----------
Yang, Y., & Soatto, S. "FDA: Fourier Domain Adaptation for Semantic
Segmentation", CVPR 2020.  https://arxiv.org/abs/2004.05498
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path('/home/dsa/stenosis')
SRC_ROOT = ROOT / 'data' / 'stenosis_arcade'
TRG_ROOT = ROOT / 'data' / 'cadica_50plus_new'   # internal video (B)
DEFAULT_OUT = ROOT / 'data' / 'stenosis_arcade_fda_B'

IMG_SIZE = 512
SRC_SPLITS = ('train', 'val', 'test')


# ---------------------------------------------------------------------------
# Core FDA op
# ---------------------------------------------------------------------------
def _low_freq_mask(h: int, w: int, beta: float) -> tuple[slice, slice]:
    """Centered square mask covering the lowest 2*b x 2*b coefficients of
    the *fft-shifted* spectrum, where b = floor(min(H,W) * beta)."""
    b = int(np.floor(min(h, w) * beta))
    cy, cx = h // 2, w // 2
    return slice(cy - b, cy + b + 1), slice(cx - b, cx + b + 1)


def fourier_domain_adaptation(
    src_img: np.ndarray,
    trg_img: np.ndarray,
    beta: float = 0.01,
) -> np.ndarray:
    """Replace the low-frequency amplitude of ``src_img`` with that of
    ``trg_img`` while keeping the source phase.

    Parameters
    ----------
    src_img, trg_img : np.ndarray
        HxWxC uint8 images. Both are assumed to share the same H, W, C
        (resize before calling).
    beta : float
        Fraction of the spectrum (per-axis half-width) to swap. Typical
        values 0.005..0.09; the paper recommends ~0.01 for natural images.

    Returns
    -------
    np.ndarray
        HxWxC uint8 FDA-adapted source image.
    """
    if src_img.shape != trg_img.shape:
        raise ValueError(f'shape mismatch: {src_img.shape} vs {trg_img.shape}')
    if src_img.dtype != np.uint8 or trg_img.dtype != np.uint8:
        raise ValueError('expected uint8 inputs')

    src = src_img.astype(np.float32)
    trg = trg_img.astype(np.float32)

    # Per-channel FFT (channels-last -> move to front for vectorised fft2).
    src = np.transpose(src, (2, 0, 1))   # (C, H, W)
    trg = np.transpose(trg, (2, 0, 1))

    fft_src = np.fft.fft2(src, axes=(-2, -1))
    fft_trg = np.fft.fft2(trg, axes=(-2, -1))

    amp_src = np.abs(fft_src)
    pha_src = np.angle(fft_src)
    amp_trg = np.abs(fft_trg)

    amp_src_shift = np.fft.fftshift(amp_src, axes=(-2, -1))
    amp_trg_shift = np.fft.fftshift(amp_trg, axes=(-2, -1))

    h, w = src.shape[-2:]
    sy, sx = _low_freq_mask(h, w, beta)
    amp_src_shift[..., sy, sx] = amp_trg_shift[..., sy, sx]

    amp_mix = np.fft.ifftshift(amp_src_shift, axes=(-2, -1))
    fft_mix = amp_mix * np.exp(1j * pha_src)
    out = np.fft.ifft2(fft_mix, axes=(-2, -1)).real

    out = np.transpose(out, (1, 2, 0))   # back to HxWxC
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------
def _list_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})


def _load_resized(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _copy_labels(src_lbl_dir: Path, dst_lbl_dir: Path) -> None:
    if not src_lbl_dir.exists():
        return
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for p in src_lbl_dir.iterdir():
        if p.suffix.lower() == '.txt':
            shutil.copy2(p, dst_lbl_dir / p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--beta', type=float, default=0.01,
                    help='FDA low-frequency window fraction (default 0.01)')
    ap.add_argument('--img-size', type=int, default=IMG_SIZE)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-targets', type=int, default=0,
                    help='If >0, cap the random target pool to this size')
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Build target pool from Dataset B only.
    trg_pool = _list_images(TRG_ROOT / 'images')
    if not trg_pool:
        raise RuntimeError(f'no target images found under {TRG_ROOT / "images"}')
    if args.max_targets > 0 and len(trg_pool) > args.max_targets:
        trg_pool = rng.sample(trg_pool, args.max_targets)
    print(f'target pool (B): {len(trg_pool)} images')

    # Cache target images lazily on first access (memory-light).
    trg_cache: dict[Path, np.ndarray] = {}

    def _get_trg() -> np.ndarray:
        p = rng.choice(trg_pool)
        arr = trg_cache.get(p)
        if arr is None:
            arr = _load_resized(p, args.img_size)
            trg_cache[p] = arr
        return arr

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in SRC_SPLITS:
        src_imgs = _list_images(SRC_ROOT / split / 'images')
        if not src_imgs:
            continue
        out_img_dir = args.out / split / 'images'
        out_img_dir.mkdir(parents=True, exist_ok=True)
        _copy_labels(SRC_ROOT / split / 'labels', args.out / split / 'labels')

        for sp in tqdm(src_imgs, desc=f'FDA {split}'):
            src_arr = _load_resized(sp, args.img_size)
            trg_arr = _get_trg()
            adapted = fourier_domain_adaptation(src_arr, trg_arr, beta=args.beta)
            Image.fromarray(adapted).save(out_img_dir / f'{sp.stem}.png',
                                          optimize=False)
            total += 1

    print(f'wrote {total} FDA-adapted images under {args.out} '
          f'(beta={args.beta}, size={args.img_size})')


if __name__ == '__main__':
    main()
