#!/usr/bin/env python3
"""
Create a combined dataset from CADICA 50plus (3 random frames per series)
and Stenosis ARCADE.

For CADICA, each "series" is a unique px_vy combination (e.g., p1_v9).
We randomly pick at most 3 frames from each series (across train/valid/test
splits separately) to reduce CADICA dominance.

Structure:
  train: cadica_50plus train (3 per series) + arcade train
  val:   arcade val only
  test:  arcade test only

Classes (2):
  0: stenosis_0  (50-70%)
  1: stenosis_1  (>70%)
"""

import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 42
FRAMES_PER_SERIES = 3

BASE = Path("/home/dsa/stenosis/data")

# --- Sources ---
CADICA_50 = BASE / "cadica_split_50plus"
ARCADE_SRC = BASE / "stenosis_arcade"

# --- Destination ---
COMBINED = BASE / "combined_cadica3_arcade"

# Regex to extract series (px_vy) from filenames like p10_v1_00016
SERIES_RE = re.compile(r"^(p\d+_v\d+)_\d+$")


def get_series(stem: str) -> str | None:
    """Extract series id (e.g., 'p10_v1') from a filename stem."""
    m = SERIES_RE.match(stem)
    return m.group(1) if m else None


def sample_cadica_frames(lbl_dir: Path, max_per_series: int, rng: random.Random) -> list[Path]:
    """Group label files by series, then sample up to max_per_series from each."""
    series_map: dict[str, list[Path]] = defaultdict(list)

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        series = get_series(lbl_file.stem)
        if series:
            series_map[series].append(lbl_file)
        else:
            # Keep files that don't match the pattern as-is
            series_map[f"__unknown_{lbl_file.stem}__"] = [lbl_file]

    sampled = []
    for series in sorted(series_map):
        frames = series_map[series]
        if len(frames) <= max_per_series:
            sampled.extend(frames)
        else:
            sampled.extend(rng.sample(frames, max_per_series))

    return sorted(sampled)


def symlink_pair(src_img: Path, src_lbl: Path, dst_img_dir: Path, dst_lbl_dir: Path,
                 prefix: str = "") -> bool:
    """Symlink an image+label pair to destination. Returns True if successful."""
    img_name = prefix + src_img.name
    lbl_name = prefix + src_lbl.name
    dst_img = dst_img_dir / img_name
    dst_lbl = dst_lbl_dir / lbl_name

    if src_img.exists() and not dst_img.exists():
        os.symlink(src_img.resolve(), dst_img)
    if src_lbl.exists() and not dst_lbl.exists():
        os.symlink(src_lbl.resolve(), dst_lbl)
    return True


def create_combined():
    rng = random.Random(SEED)

    print("=" * 60)
    print(f"Creating combined_cadica3_arcade (max {FRAMES_PER_SERIES} frames/series)")
    print("=" * 60)

    # === TRAIN: CADICA 50+ (sampled) + ARCADE ===
    dst_train_img = COMBINED / "train" / "images"
    dst_train_lbl = COMBINED / "train" / "labels"
    dst_train_img.mkdir(parents=True, exist_ok=True)
    dst_train_lbl.mkdir(parents=True, exist_ok=True)

    # CADICA 50+ train — sample 3 per series
    cadica_train_img = CADICA_50 / "train" / "images"
    cadica_train_lbl = CADICA_50 / "train" / "labels"
    sampled_labels = sample_cadica_frames(cadica_train_lbl, FRAMES_PER_SERIES, rng)

    cadica_count = 0
    for lbl in sampled_labels:
        img = cadica_train_img / (lbl.stem + ".png")
        symlink_pair(img, lbl, dst_train_img, dst_train_lbl)
        cadica_count += 1

    # Print series stats
    total_series = len(set(get_series(l.stem) for l in cadica_train_lbl.glob("*.txt")))
    print(f"  Train CADICA: {cadica_count} frames sampled from {total_series} series "
          f"(was {len(list(cadica_train_lbl.glob('*.txt')))} frames)")

    # ARCADE train
    arcade_train_img = ARCADE_SRC / "train" / "images"
    arcade_train_lbl = ARCADE_SRC / "train" / "labels"
    arcade_count = 0
    for img in sorted(arcade_train_img.glob("*.png")):
        lbl = arcade_train_lbl / (img.stem + ".txt")
        symlink_pair(img, lbl, dst_train_img, dst_train_lbl, prefix="arcade_")
        arcade_count += 1
    print(f"  Train ARCADE: {arcade_count} images")
    print(f"  Train total:  {cadica_count + arcade_count}")

    # === VAL: ARCADE only ===
    dst_val_img = COMBINED / "val" / "images"
    dst_val_lbl = COMBINED / "val" / "labels"
    dst_val_img.mkdir(parents=True, exist_ok=True)
    dst_val_lbl.mkdir(parents=True, exist_ok=True)

    val_count = 0
    arcade_val_img = ARCADE_SRC / "val" / "images"
    arcade_val_lbl = ARCADE_SRC / "val" / "labels"
    for img in sorted(arcade_val_img.glob("*.png")):
        lbl = arcade_val_lbl / (img.stem + ".txt")
        symlink_pair(img, lbl, dst_val_img, dst_val_lbl)
        val_count += 1
    print(f"  Val:  {val_count} images (ARCADE)")

    # === TEST: ARCADE only ===
    dst_test_img = COMBINED / "test" / "images"
    dst_test_lbl = COMBINED / "test" / "labels"
    dst_test_img.mkdir(parents=True, exist_ok=True)
    dst_test_lbl.mkdir(parents=True, exist_ok=True)

    test_count = 0
    arcade_test_img = ARCADE_SRC / "test" / "images"
    arcade_test_lbl = ARCADE_SRC / "test" / "labels"
    for img in sorted(arcade_test_img.glob("*.png")):
        lbl = arcade_test_lbl / (img.stem + ".txt")
        symlink_pair(img, lbl, dst_test_img, dst_test_lbl)
        test_count += 1
    print(f"  Test: {test_count} images (ARCADE)")

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
    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)

    for split_dir in ["train", "val", "test"]:
        lbl_dir = COMBINED / split_dir / "labels"
        img_dir = COMBINED / split_dir / "images"
        if not lbl_dir.exists():
            continue
        class_counts: dict[int, int] = {}
        total_files = 0
        empty_files = 0
        for lbl_file in lbl_dir.glob("*.txt"):
            total_files += 1
            content = lbl_file.read_text().strip()
            if not content:
                empty_files += 1
                continue
            for line in content.split("\n"):
                cls = int(line.strip().split()[0])
                class_counts[cls] = class_counts.get(cls, 0) + 1
        img_count = len(list(img_dir.glob("*.png")))
        print(f"  {split_dir}: {img_count} images, {total_files} labels "
              f"({empty_files} empty), classes: {dict(sorted(class_counts.items()))}")


if __name__ == "__main__":
    if COMBINED.exists():
        shutil.rmtree(COMBINED)
        print(f"Removed existing {COMBINED}")

    create_combined()
    verify()
    print("\nDone!")
