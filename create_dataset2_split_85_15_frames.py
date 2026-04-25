"""
Create an 85/15 train/valid frame-level split of dataset2.
Frames are shuffled randomly (no sequence grouping).
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

DATASET_DIR = Path("/home/dsa/stenosis/data/dataset2_base")
OUTPUT_ROOT = Path("/home/dsa/stenosis/data/dataset2_split_85_15_frames")
TRAIN_RATIO = 0.85
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels"

    # Collect all (image, label) pairs
    samples = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"Missing label: {lbl}")
        samples.append((img, lbl))

    total = len(samples)
    print(f"Total images: {total}")

    # Shuffle and split
    rng = random.Random(SEED)
    rng.shuffle(samples)

    split_idx = int(total * TRAIN_RATIO)
    train_samples = samples[:split_idx]
    valid_samples = samples[split_idx:]

    print(f"Train: {len(train_samples)} images ({len(train_samples)/total:.1%})")
    print(f"Valid: {len(valid_samples)} images ({len(valid_samples)/total:.1%})")

    # Create output dirs
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split in ("train", "valid"):
        (OUTPUT_ROOT / split / "images").mkdir(parents=True)
        (OUTPUT_ROOT / split / "labels").mkdir(parents=True)

    # Copy files
    for img, lbl in train_samples:
        shutil.copy2(img, OUTPUT_ROOT / "train" / "images" / img.name)
        shutil.copy2(lbl, OUTPUT_ROOT / "train" / "labels" / lbl.name)
    for img, lbl in valid_samples:
        shutil.copy2(img, OUTPUT_ROOT / "valid" / "images" / img.name)
        shutil.copy2(lbl, OUTPUT_ROOT / "valid" / "labels" / lbl.name)

    # Write YAML
    yaml_text = (
        f"path: {OUTPUT_ROOT}\n"
        f"train: train/images\n"
        f"val: valid/images\n\n"
        f"nc: 1\n"
        f"names: ['Stenosis']\n"
    )
    (OUTPUT_ROOT / "data.yaml").write_text(yaml_text)

    # Write manifest
    manifest = {
        "ratios": {"train": TRAIN_RATIO, "valid": 1 - TRAIN_RATIO},
        "seed": SEED,
        "split_by": "frames",
        "splits": {
            "train": {"image_count": len(train_samples)},
            "valid": {"image_count": len(valid_samples)},
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(f"Done. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
