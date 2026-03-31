#!/usr/bin/env python3
"""
Create a combined dataset: ARCADE + dataset2_split.

ARCADE classes (unchanged):
  0: stenosis_0  (50-70%)
  1: stenosis_1  (>70%)

dataset2 has 1 class (0: Stenosis) -> remap ALL to class 1 in combined dataset.

Split mapping:
  ARCADE: train -> train, val -> valid, test -> test
  dataset2_split: train -> train, valid -> valid, test -> test
"""

import os
import shutil
from pathlib import Path

BASE = Path("/home/dsa/stenosis/data")

ARCADE_SRC = BASE / "stenosis_arcade"
DATASET2_SRC = BASE / "dataset2_split"

COMBINED = BASE / "combined_arcade_dataset2"

# dataset2: all class 0 -> class 1
DATASET2_REMAP = {0: 1}


def remap_label_file(src_path: Path, dst_path: Path, remap: dict) -> bool:
    """Remap class IDs in a YOLO label file. Returns True if file has any valid lines."""
    lines = src_path.read_text().strip().split("\n") if src_path.stat().st_size > 0 else []
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(parts[0])
        new_cls = remap.get(old_cls)
        if new_cls is None:
            continue  # skip this annotation
        parts[0] = str(new_cls)
        new_lines.append(" ".join(parts))
    if new_lines:
        dst_path.write_text("\n".join(new_lines) + "\n")
        return True
    return False


def copy_with_labels(src_img_dir: Path, src_lbl_dir: Path,
                     dst_img_dir: Path, dst_lbl_dir: Path,
                     remap: dict = None, prefix: str = ""):
    """Copy images + labels. If remap is given, remap class IDs; otherwise copy as-is."""
    copied = 0
    for lbl_file in sorted(src_lbl_dir.glob("*.txt")):
        stem = lbl_file.stem
        # Find corresponding image
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = src_img_dir / (stem + ext)
            if candidate.exists():
                img_file = candidate
                break
        if img_file is None:
            continue

        dst_stem = prefix + stem if prefix else stem
        dst_lbl = dst_lbl_dir / (dst_stem + ".txt")
        dst_img = dst_img_dir / (dst_stem + img_file.suffix)

        if remap is not None:
            ok = remap_label_file(lbl_file, dst_lbl, remap)
            if not ok:
                continue
        else:
            shutil.copy2(lbl_file, dst_lbl)

        shutil.copy2(img_file, dst_img)
        copied += 1
    return copied


def main():
    # Clean output
    if COMBINED.exists():
        shutil.rmtree(COMBINED)

    split_map = {
        "train": "train",
        "valid": "valid",
        "test": "test",
    }

    # Create directories
    for split in split_map.values():
        (COMBINED / split / "images").mkdir(parents=True, exist_ok=True)
        (COMBINED / split / "labels").mkdir(parents=True, exist_ok=True)

    # --- Copy ARCADE (no remap needed, classes 0 and 1 stay) ---
    arcade_split_map = {"train": "train", "val": "valid", "test": "test"}
    print("=== Copying ARCADE data ===")
    for arc_split, comb_split in arcade_split_map.items():
        src_img = ARCADE_SRC / arc_split / "images"
        src_lbl = ARCADE_SRC / arc_split / "labels"
        dst_img = COMBINED / comb_split / "images"
        dst_lbl = COMBINED / comb_split / "labels"
        n = copy_with_labels(src_img, src_lbl, dst_img, dst_lbl, remap=None, prefix="arcade_")
        print(f"  {arc_split} -> {comb_split}: {n} images")

    # --- Copy dataset2_split (remap class 0 -> 1) ---
    ds2_split_map = {"train": "train", "valid": "valid", "test": "test"}
    print("\n=== Copying dataset2 data (class 0 -> 1) ===")
    for ds2_split, comb_split in ds2_split_map.items():
        src_img = DATASET2_SRC / ds2_split / "images"
        src_lbl = DATASET2_SRC / ds2_split / "labels"
        dst_img = COMBINED / comb_split / "images"
        dst_lbl = COMBINED / comb_split / "labels"
        n = copy_with_labels(src_img, src_lbl, dst_img, dst_lbl, remap=DATASET2_REMAP, prefix="ds2_")
        print(f"  {ds2_split} -> {comb_split}: {n} images")

    # --- Print totals ---
    print("\n=== Combined dataset totals ===")
    for split in ["train", "valid", "test"]:
        imgs = len(list((COMBINED / split / "images").glob("*")))
        lbls = len(list((COMBINED / split / "labels").glob("*.txt")))
        print(f"  {split}: {imgs} images, {lbls} labels")

    # --- Write data.yaml ---
    yaml_content = f"""train: {COMBINED / 'train' / 'images'}
val: {COMBINED / 'valid' / 'images'}
test: {COMBINED / 'test' / 'images'}

nc: 2
names: ['stenosis_0', 'stenosis_1']
"""
    (COMBINED / "data.yaml").write_text(yaml_content)
    print(f"\ndata.yaml written to {COMBINED / 'data.yaml'}")
    print("Done!")


if __name__ == "__main__":
    main()
