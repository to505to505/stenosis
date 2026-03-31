#!/usr/bin/env python3
"""
Create cadica_split_50+ dataset (only stenosis >= 50%) and
a combined dataset (cadica_50+ train + arcade train, arcade val/test only).

CADICA original classes:
  0: stenosis_0_20   -> REMOVE
  1: stenosis_20_50  -> REMOVE
  2: stenosis_50_70  -> new class 0
  3: stenosis_70_90  -> new class 1
  4: stenosis_90_98  -> new class 1
  5: stenosis_99     -> new class 1
  6: stenosis_100    -> new class 1

ARCADE classes (unchanged):
  0: stenosis_0  (50-70%)
  1: stenosis_1  (>70%)
"""

import os
import shutil
from pathlib import Path

BASE = Path("/home/dsa/stenosis/data")

# --- Sources ---
CADICA_SRC = BASE / "cadica_split"
ARCADE_SRC = BASE / "stenosis_arcade"

# --- Destinations ---
CADICA_50 = BASE / "cadica_split_50plus"
COMBINED = BASE / "combined_cadica_arcade"

# Class remapping for CADICA: old_id -> new_id (None means skip)
CADICA_REMAP = {
    0: None,  # stenosis_0_20 -> remove
    1: None,  # stenosis_20_50 -> remove
    2: 0,     # stenosis_50_70 -> class 0
    3: 1,     # stenosis_70_90 -> class 1
    4: 1,     # stenosis_90_98 -> class 1
    5: 1,     # stenosis_99 -> class 1
    6: 1,     # stenosis_100 -> class 1
}


def remap_label_file(src_path: Path, dst_path: Path, remap: dict) -> bool:
    """Remap class IDs in a YOLO label file. Returns True if file has any valid lines."""
    lines = src_path.read_text().strip().split("\n") if src_path.stat().st_size > 0 else []
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        old_class = int(parts[0])
        new_class = remap.get(old_class)
        if new_class is not None:
            parts[0] = str(new_class)
            new_lines.append(" ".join(parts))
    # Always write the file (even if empty, YOLO treats empty files as background)
    dst_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return True


def create_cadica_50plus():
    """Create cadica_split_50+ with remapped labels and symlinked images."""
    print("=" * 60)
    print("Creating cadica_split_50plus...")
    print("=" * 60)

    for split in ["train", "valid", "test"]:
        src_img_dir = CADICA_SRC / split / "images"
        src_lbl_dir = CADICA_SRC / split / "labels"
        dst_img_dir = CADICA_50 / split / "images"
        dst_lbl_dir = CADICA_50 / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        skipped = 0
        for lbl_file in sorted(src_lbl_dir.glob("*.txt")):
            stem = lbl_file.stem
            # Check if any annotation in this file has class >= 2
            lines = lbl_file.read_text().strip().split("\n") if lbl_file.stat().st_size > 0 else []
            has_valid = False
            for line in lines:
                if not line.strip():
                    continue
                old_class = int(line.strip().split()[0])
                if CADICA_REMAP.get(old_class) is not None:
                    has_valid = True
                    break

            # We keep all images (even those without 50+ annotations -> negative examples)
            # But if you only want images that have at least one 50+ annotation, uncomment:
            # if not has_valid:
            #     skipped += 1
            #     continue

            # Remap label
            remap_label_file(lbl_file, dst_lbl_dir / lbl_file.name, CADICA_REMAP)

            # Symlink image
            img_name = stem + ".png"
            src_img = src_img_dir / img_name
            dst_img = dst_img_dir / img_name
            if src_img.exists() and not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)
            kept += 1

        print(f"  {split}: {kept} images kept")

    # Write YAML
    yaml_content = f"""train: {CADICA_50}/train/images
val: {CADICA_50}/valid/images
test: {CADICA_50}/test/images

nc: 2
names: ['stenosis_0', 'stenosis_1']
"""
    (CADICA_50 / "data.yaml").write_text(yaml_content)
    print(f"  Wrote {CADICA_50 / 'data.yaml'}")


def create_combined_dataset():
    """Create combined dataset: train from cadica_50+ and arcade, val/test from arcade only."""
    print()
    print("=" * 60)
    print("Creating combined_cadica_arcade...")
    print("=" * 60)

    # --- TRAIN: merge cadica_50+ train + arcade train ---
    dst_train_img = COMBINED / "train" / "images"
    dst_train_lbl = COMBINED / "train" / "labels"
    dst_train_img.mkdir(parents=True, exist_ok=True)
    dst_train_lbl.mkdir(parents=True, exist_ok=True)

    # Copy/symlink CADICA 50+ train
    cadica_train_img = CADICA_50 / "train" / "images"
    cadica_train_lbl = CADICA_50 / "train" / "labels"
    cadica_count = 0
    for img in sorted(cadica_train_img.glob("*.png")):
        dst = dst_train_img / img.name
        if not dst.exists():
            # Resolve in case it's already a symlink
            os.symlink(img.resolve(), dst)
        lbl = cadica_train_lbl / (img.stem + ".txt")
        dst_lbl = dst_train_lbl / lbl.name
        if lbl.exists() and not dst_lbl.exists():
            os.symlink(lbl.resolve(), dst_lbl)
        cadica_count += 1
    print(f"  Train: {cadica_count} images from CADICA 50+")

    # Copy/symlink ARCADE train (prefix "arcade_" to avoid name conflicts)
    arcade_train_img = ARCADE_SRC / "train" / "images"
    arcade_train_lbl = ARCADE_SRC / "train" / "labels"
    arcade_count = 0
    for img in sorted(arcade_train_img.glob("*.png")):
        prefixed_name = "arcade_" + img.name
        dst = dst_train_img / prefixed_name
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        lbl = arcade_train_lbl / (img.stem + ".txt")
        dst_lbl = dst_train_lbl / ("arcade_" + lbl.name)
        if lbl.exists() and not dst_lbl.exists():
            os.symlink(lbl.resolve(), dst_lbl)
        arcade_count += 1
    print(f"  Train: {arcade_count} images from ARCADE")
    print(f"  Train total: {cadica_count + arcade_count}")

    # --- VAL: arcade only ---
    dst_val_img = COMBINED / "val" / "images"
    dst_val_lbl = COMBINED / "val" / "labels"
    dst_val_img.mkdir(parents=True, exist_ok=True)
    dst_val_lbl.mkdir(parents=True, exist_ok=True)

    val_count = 0
    arcade_val_img = ARCADE_SRC / "val" / "images"
    arcade_val_lbl = ARCADE_SRC / "val" / "labels"
    for img in sorted(arcade_val_img.glob("*.png")):
        dst = dst_val_img / img.name
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        lbl = arcade_val_lbl / (img.stem + ".txt")
        dst_lbl = dst_val_lbl / lbl.name
        if lbl.exists() and not dst_lbl.exists():
            os.symlink(lbl.resolve(), dst_lbl)
        val_count += 1
    print(f"  Val: {val_count} images from ARCADE")

    # --- TEST: arcade only ---
    dst_test_img = COMBINED / "test" / "images"
    dst_test_lbl = COMBINED / "test" / "labels"
    dst_test_img.mkdir(parents=True, exist_ok=True)
    dst_test_lbl.mkdir(parents=True, exist_ok=True)

    test_count = 0
    arcade_test_img = ARCADE_SRC / "test" / "images"
    arcade_test_lbl = ARCADE_SRC / "test" / "labels"
    for img in sorted(arcade_test_img.glob("*.png")):
        dst = dst_test_img / img.name
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        lbl = arcade_test_lbl / (img.stem + ".txt")
        dst_lbl = dst_test_lbl / lbl.name
        if lbl.exists() and not dst_lbl.exists():
            os.symlink(lbl.resolve(), dst_lbl)
        test_count += 1
    print(f"  Test: {test_count} images from ARCADE")

    # Write YAML
    yaml_content = f"""train: {COMBINED}/train/images
val: {COMBINED}/val/images
test: {COMBINED}/test/images

nc: 2
names: ['stenosis_0', 'stenosis_1']
"""
    (COMBINED / "data.yaml").write_text(yaml_content)
    print(f"  Wrote {COMBINED / 'data.yaml'}")


def verify():
    """Print class distribution in new datasets."""
    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)

    for name, base in [("cadica_split_50plus", CADICA_50), ("combined_cadica_arcade", COMBINED)]:
        print(f"\n--- {name} ---")
        for split_dir in ["train", "valid", "val", "test"]:
            lbl_dir = base / split_dir / "labels"
            if not lbl_dir.exists():
                continue
            class_counts = {}
            total_files = 0
            for lbl_file in lbl_dir.glob("*.txt"):
                total_files += 1
                content = lbl_file.read_text().strip()
                if not content:
                    continue
                for line in content.split("\n"):
                    cls = int(line.strip().split()[0])
                    class_counts[cls] = class_counts.get(cls, 0) + 1
            print(f"  {split_dir}: {total_files} files, classes: {dict(sorted(class_counts.items()))}")


if __name__ == "__main__":
    # Clean up if re-running
    for d in [CADICA_50, COMBINED]:
        if d.exists():
            shutil.rmtree(d)
            print(f"Removed existing {d}")

    create_cadica_50plus()
    create_combined_dataset()
    verify()
    print("\nDone!")
