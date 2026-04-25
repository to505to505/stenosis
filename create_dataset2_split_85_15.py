"""
Create an 85/15 train/valid sequence-level split of dataset2.
No test set. Frames from the same sequence never appear in both subsets.
Sequence ID = first 3 underscore-separated tokens of the filename
(e.g. "14_092_1" from "14_092_1_0060_bmp_jpg.rf.HASH.jpg").
"""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

DATASET_DIR = Path("/home/dsa/stenosis/data/dataset2_base")
OUTPUT_ROOT = Path("/home/dsa/stenosis/data/dataset2_split_85_15")
TRAIN_RATIO = 0.85
SEED = 42
SEQ_TOKENS = 3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_sequence_id(filename: str) -> str:
    parts = filename.split("_")
    return "_".join(parts[:SEQ_TOKENS])


def main():
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels"

    # Collect samples grouped by sequence
    seq_samples: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img in sorted(images_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"Missing label: {lbl}")
        sid = extract_sequence_id(img.name)
        seq_samples[sid].append((img, lbl))

    total_images = sum(len(s) for s in seq_samples.values())
    print(f"Total sequences: {len(seq_samples)}, total images: {total_images}")

    # Greedy assignment: shuffle then sort by size desc, assign to minimize error
    rng = random.Random(SEED)
    sequences = list(seq_samples.items())
    rng.shuffle(sequences)
    sequences.sort(key=lambda x: len(x[1]), reverse=True)

    target_train = total_images * TRAIN_RATIO
    target_valid = total_images * (1 - TRAIN_RATIO)

    train_seqs, valid_seqs = [], []
    train_samples, valid_samples = [], []
    train_count, valid_count = 0, 0

    for sid, samples in sequences:
        n = len(samples)
        # Pick whichever split minimizes squared error from target
        cost_train = ((train_count + n - target_train) ** 2 / max(target_train, 1)
                      + (valid_count - target_valid) ** 2 / max(target_valid, 1))
        cost_valid = ((train_count - target_train) ** 2 / max(target_train, 1)
                      + (valid_count + n - target_valid) ** 2 / max(target_valid, 1))
        if cost_train <= cost_valid:
            train_seqs.append(sid)
            train_samples.extend(samples)
            train_count += n
        else:
            valid_seqs.append(sid)
            valid_samples.extend(samples)
            valid_count += n

    print(f"Train: {len(train_seqs)} sequences, {train_count} images ({train_count/total_images:.1%})")
    print(f"Valid: {len(valid_seqs)} sequences, {valid_count} images ({valid_count/total_images:.1%})")

    # Verify no overlap
    assert set(train_seqs).isdisjoint(set(valid_seqs)), "Sequence overlap!"

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
        "splits": {
            "train": {
                "sequences": sorted(train_seqs),
                "sequence_count": len(train_seqs),
                "image_count": train_count,
            },
            "valid": {
                "sequences": sorted(valid_seqs),
                "sequence_count": len(valid_seqs),
                "image_count": valid_count,
            },
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(f"Done. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
