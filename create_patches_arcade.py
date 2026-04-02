#!/usr/bin/env python3
"""
Extract 64x64 patches from stenosis_arcade dataset and organise them into
three discrete classes for downstream classification / GAN training.

Output layout
─────────────
data/stenosis_arcade/patches/
    trainA/        ← healthy vessels   (class 0)  — from train/labels_healthy
    trainB_NS/     ← non-severe stenos (class 1)  — from train+val+test labels, YOLO class 0
    trainB_S/      ← severe stenosis   (class 2)  — from train+val+test labels, YOLO class 1

Each patch is:
  • cropped from the bounding-box annotation (YOLO xywh → pixel xyxy)
  • resized to 64×64
  • normalised to [-1, 1]  (saved as float32 .npy for lossless usage,
    AND as uint8 .png for quick visual inspection)
"""

import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image


# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent / "data" / "stenosis_arcade"
OUT  = BASE / "patches"

SPLITS_STENOSIS = ["train", "val", "test"]   # stenosis boxes: all splits
SPLITS_HEALTHY  = ["train"]                  # healthy boxes: train only

PATCH_SIZE = 64
IMG_SIZE   = 512  # all images are 512×512


# ── helpers ──────────────────────────────────────────────────────────────────
def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalised (cx, cy, w, h) → pixel (x1, y1, x2, y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    # clamp
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))
    return x1, y1, x2, y2


def extract_patch(img: Image.Image, box_xyxy, size=PATCH_SIZE):
    """Crop + resize → PIL Image (RGB, uint8)."""
    crop = img.crop(box_xyxy)          # (x1, y1, x2, y2)
    crop = crop.resize((size, size), Image.BILINEAR)
    return crop


def normalise(img_arr: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float32 [-1,1]."""
    return img_arr.astype(np.float32) / 127.5 - 1.0


def parse_label_file(label_path):
    """Yield (class_id, cx, cy, w, h) from a YOLO label file."""
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            yield cls, cx, cy, w, h


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    # create output dirs
    dir_healthy = OUT / "trainA"
    dir_ns      = OUT / "trainB_NS"
    dir_s       = OUT / "trainB_S"
    for d in (dir_healthy, dir_ns, dir_s):
        d.mkdir(parents=True, exist_ok=True)

    counters = {"healthy": 0, "non_severe": 0, "severe": 0}

    # ── 1. Healthy patches (train/labels_healthy) ───────────────────────────
    for split in SPLITS_HEALTHY:
        img_dir   = BASE / split / "images"
        label_dir = BASE / split / "labels_healthy"
        if not label_dir.exists():
            print(f"[WARN] {label_dir} not found, skipping.")
            continue

        label_files = sorted(glob.glob(str(label_dir / "*.txt")))
        for lf in label_files:
            stem = Path(lf).stem
            img_path = img_dir / f"{stem}.png"
            if not img_path.exists():
                # try jpg
                img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size

            for _, cx, cy, w, h in parse_label_file(lf):
                box = yolo_to_xyxy(cx, cy, w, h, iw, ih)
                patch = extract_patch(img, box)
                arr   = np.array(patch)
                norm  = normalise(arr)

                idx = counters["healthy"]
                # save png (visual) and npy (normalised float32)
                patch.save(dir_healthy / f"{idx:06d}.png")
                np.save(dir_healthy / f"{idx:06d}.npy", norm)
                counters["healthy"] += 1

    print(f"Healthy patches:      {counters['healthy']}")

    # ── 2. Stenosis patches (all splits, labels/) ───────────────────────────
    for split in SPLITS_STENOSIS:
        img_dir   = BASE / split / "images"
        label_dir = BASE / split / "labels"
        if not label_dir.exists():
            print(f"[WARN] {label_dir} not found, skipping.")
            continue

        label_files = sorted(glob.glob(str(label_dir / "*.txt")))
        for lf in label_files:
            stem = Path(lf).stem
            img_path = img_dir / f"{stem}.png"
            if not img_path.exists():
                img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                continue

            # skip empty files
            if os.path.getsize(lf) == 0:
                continue

            img = Image.open(img_path).convert("RGB")
            iw, ih = img.size

            for cls, cx, cy, w, h in parse_label_file(lf):
                box   = yolo_to_xyxy(cx, cy, w, h, iw, ih)
                patch = extract_patch(img, box)
                arr   = np.array(patch)
                norm  = normalise(arr)

                if cls == 0:  # non-severe
                    idx = counters["non_severe"]
                    patch.save(dir_ns / f"{idx:06d}.png")
                    np.save(dir_ns / f"{idx:06d}.npy", norm)
                    counters["non_severe"] += 1
                elif cls == 1:  # severe
                    idx = counters["severe"]
                    patch.save(dir_s / f"{idx:06d}.png")
                    np.save(dir_s / f"{idx:06d}.npy", norm)
                    counters["severe"] += 1
                else:
                    print(f"[WARN] Unknown class {cls} in {lf}")

    print(f"Non-severe patches:   {counters['non_severe']}")
    print(f"Severe patches:       {counters['severe']}")
    print(f"Total:                {sum(counters.values())}")
    print(f"\nPatches saved to {OUT}")


if __name__ == "__main__":
    main()
