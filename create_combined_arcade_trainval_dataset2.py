#!/usr/bin/env python3
"""
Create a combined dataset: ARCADE train+val (NO test) as training
+ dataset2_split train/valid/test.
All classes are converted to a single class (0: stenosis).

Training split: arcade train + arcade val + dataset2_split train
Valid split: dataset2_split valid
Test split: dataset2_split test

ARCADE originally has 2 classes, dataset2 has 1 class.
All are remapped to class 0 (stenosis).
"""

import shutil
from pathlib import Path

BASE = Path("/home/dsa/stenosis/data")

ARCADE_SRC = BASE / "stenosis_arcade"
DATASET2_SRC = BASE / "dataset2_split"

COMBINED = BASE / "combined_arcade_trainval_dataset2"

# Remap everything to single class 0
ARCADE_REMAP = {0: 0, 1: 0}
DATASET2_REMAP = {0: 0}


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
            continue
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
    if COMBINED.exists():
        shutil.rmtree(COMBINED)

    # Create directories
    for split in ["train", "valid", "test"]:
        (COMBINED / split / "images").mkdir(parents=True, exist_ok=True)
        (COMBINED / split / "labels").mkdir(parents=True, exist_ok=True)

    # === Copy ARCADE train+val into train (NO test) ===
    print("=== Copying ARCADE train+val data into train ===")
    total_arcade = 0
    for arc_split in ["train", "val"]:
        src_img = ARCADE_SRC / arc_split / "images"
        src_lbl = ARCADE_SRC / arc_split / "labels"
        dst_img = COMBINED / "train" / "images"
        dst_lbl = COMBINED / "train" / "labels"
        n = copy_with_labels(src_img, src_lbl, dst_img, dst_lbl, remap=ARCADE_REMAP, prefix="arcade_")
        print(f"  arcade/{arc_split} -> train: {n} images")
        total_arcade += n
    print(f"  Total ARCADE in train: {total_arcade}")

    # === Copy dataset2_split train into train ===
    print("\n=== Copying dataset2_split train into train ===")
    n = copy_with_labels(
        DATASET2_SRC / "train" / "images",
        DATASET2_SRC / "train" / "labels",
        COMBINED / "train" / "images",
        COMBINED / "train" / "labels",
        remap=DATASET2_REMAP,
        prefix="ds2_",
    )
    print(f"  dataset2/train -> train: {n} images")

    # === Copy dataset2_split valid into valid ===
    print("\n=== Copying dataset2_split valid into valid ===")
    n = copy_with_labels(
        DATASET2_SRC / "valid" / "images",
        DATASET2_SRC / "valid" / "labels",
        COMBINED / "valid" / "images",
        COMBINED / "valid" / "labels",
        remap=DATASET2_REMAP,
        prefix="ds2_",
    )
    print(f"  dataset2/valid -> valid: {n} images")

    # === Copy dataset2_split test into test ===
    print("\n=== Copying dataset2_split test into test ===")
    n = copy_with_labels(
        DATASET2_SRC / "test" / "images",
        DATASET2_SRC / "test" / "labels",
        COMBINED / "test" / "images",
        COMBINED / "test" / "labels",
        remap=DATASET2_REMAP,
        prefix="ds2_",
    )
    print(f"  dataset2/test -> test: {n} images")

    # === Print totals ===
    print("\n=== Combined dataset totals ===")
    for split in ["train", "valid", "test"]:
        imgs = len(list((COMBINED / split / "images").glob("*")))
        lbls = len(list((COMBINED / split / "labels").glob("*.txt")))
        print(f"  {split}: {imgs} images, {lbls} labels")

    # === Write data.yaml ===
    yaml_content = f"""train: {COMBINED / 'train' / 'images'}
val: {COMBINED / 'valid' / 'images'}
test: {COMBINED / 'test' / 'images'}

nc: 1
names: ['stenosis']
"""
    (COMBINED / "data.yaml").write_text(yaml_content)
    print(f"\ndata.yaml written to {COMBINED / 'data.yaml'}")
    print("Done!")


if __name__ == "__main__":
    main()
