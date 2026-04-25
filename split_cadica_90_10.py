"""Split data/cadica_base into train/val (90/10) by patient.

Constraints:
  1. All frames of one patient stay in the same subset.
  2. Validation set must contain patients covering ALL stenosis categories
     (p0_20, p20_50, p50_70, p70_90, p90_98, p99, p100).
  3. Approximate 90/10 ratio.
"""

import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

SEED = 42
SRC = Path("data/cadica_base")
DST = Path("data/cadica_split_90_10")
VAL_RATIO = 0.10
IMG_W, IMG_H = 512, 512  # CADICA image dimensions

ALL_CATEGORIES = {"p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_patient(fname: str) -> str:
    """p3_v8_00018.png -> p3"""
    return fname.split("_")[0]


def get_patient_categories(labels_dir: Path, patient: str) -> set[str]:
    """Return set of stenosis categories present for a patient."""
    cats = set()
    for lbl in labels_dir.glob(f"{patient}_v*.txt"):
        for line in lbl.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts and parts[0] in ALL_CATEGORIES:
                cats.add(parts[0])
    return cats


def convert_label_to_yolo(src: Path, dst: Path) -> None:
    """Convert CADICA label (class_name x y w h absolute) to YOLO (0 cx cy w h normalized).

    All severity classes are mapped to single class 0 (stenosis).
    """
    text = src.read_text().strip()
    if not text:
        dst.write_text("")
        return
    lines = []
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) != 5 or parts[0] not in ALL_CATEGORIES:
            continue
        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        cx = (x + w / 2.0) / IMG_W
        cy = (y + h / 2.0) / IMG_H
        wn = w / IMG_W
        hn = h / IMG_H
        lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    random.seed(SEED)

    images_dir = SRC / "images"
    labels_dir = SRC / "labels"

    # Collect per-patient frame counts and categories
    patient_frames: dict[str, list[str]] = defaultdict(list)
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() in IMAGE_SUFFIXES:
            patient_frames[extract_patient(img.name)].append(img.name)

    patient_cats: dict[str, set[str]] = {}
    for p in patient_frames:
        patient_cats[p] = get_patient_categories(labels_dir, p)

    patients = sorted(patient_frames.keys(), key=lambda p: int(p[1:]))
    total_frames = sum(len(patient_frames[p]) for p in patients)
    target_val = total_frames * VAL_RATIO

    print(f"Total patients: {len(patients)}")
    print(f"Total frames: {total_frames}")
    print(f"Target val frames: ~{target_val:.0f}")

    # Step 1: Greedily pick val patients to cover all categories, preferring
    #          patients with rare categories first (p99, p100).
    val_patients = set()
    val_frames = 0
    covered_cats = set()

    # Rank categories by rarity (fewest patients first)
    cat_to_patients: dict[str, list[str]] = defaultdict(list)
    for p in patients:
        for c in patient_cats[p]:
            cat_to_patients[c].append(p)

    cats_by_rarity = sorted(ALL_CATEGORIES, key=lambda c: len(cat_to_patients[c]))

    for cat in cats_by_rarity:
        if cat in covered_cats:
            continue
        # Pick the patient with fewest frames that covers this (and ideally other) uncovered cats
        candidates = [p for p in cat_to_patients[cat] if p not in val_patients]
        if not candidates:
            print(f"WARNING: category {cat} cannot be added to val (already assigned)")
            continue
        # Prefer patient covering more uncovered categories; break ties by fewer frames
        candidates.sort(
            key=lambda p: (-len(patient_cats[p] - covered_cats), len(patient_frames[p]))
        )
        chosen = candidates[0]
        val_patients.add(chosen)
        val_frames += len(patient_frames[chosen])
        covered_cats |= patient_cats[chosen]
        print(f"  Val seed: {chosen} ({len(patient_frames[chosen])} frames, cats={patient_cats[chosen]})")

    print(f"After seeding: val has {len(val_patients)} patients, {val_frames} frames, cats={covered_cats}")

    # Step 2: If val is under target, add more patients (smallest first to not overshoot)
    remaining = [p for p in patients if p not in val_patients]
    random.shuffle(remaining)
    remaining.sort(key=lambda p: len(patient_frames[p]))

    for p in remaining:
        if val_frames >= target_val:
            break
        val_patients.add(p)
        val_frames += len(patient_frames[p])

    # Step 3: If val is over target, check if we can remove a non-critical patient
    # (one whose categories are still covered by other val patients)
    val_list = sorted(val_patients, key=lambda p: -len(patient_frames[p]))
    for p in val_list:
        if val_frames <= target_val:
            break
        other_cats = set()
        for q in val_patients:
            if q != p:
                other_cats |= patient_cats[q]
        if ALL_CATEGORIES <= other_cats:
            val_patients.remove(p)
            val_frames -= len(patient_frames[p])

    train_patients = set(patients) - val_patients

    train_frames = sum(len(patient_frames[p]) for p in train_patients)
    val_frames = sum(len(patient_frames[p]) for p in val_patients)

    print(f"\n=== Final split ===")
    print(f"Train: {len(train_patients)} patients, {train_frames} frames ({train_frames/total_frames*100:.1f}%)")
    print(f"Val:   {len(val_patients)} patients, {val_frames} frames ({val_frames/total_frames*100:.1f}%)")

    # Verify all categories in val
    val_cats = set()
    for p in val_patients:
        val_cats |= patient_cats[p]
    missing = ALL_CATEGORIES - val_cats
    if missing:
        print(f"ERROR: val is missing categories: {missing}")
        return
    print(f"Val categories: {sorted(val_cats)}")
    print(f"Val patients: {sorted(val_patients, key=lambda p: int(p[1:]))}")
    print(f"Train patients: {sorted(train_patients, key=lambda p: int(p[1:]))}")

    # Create output directories (use "valid" to match rfdetr_temporal convention)
    for split in ("train", "valid"):
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy files and convert labels to YOLO format (single-class: all → class 0)
    for split, pset in [("train", train_patients), ("valid", val_patients)]:
        count = 0
        for p in sorted(pset, key=lambda p: int(p[1:])):
            for fname in patient_frames[p]:
                stem = Path(fname).stem
                src_img = images_dir / fname
                src_lbl = labels_dir / f"{stem}.txt"
                shutil.copy2(src_img, DST / split / "images" / fname)
                if src_lbl.exists():
                    convert_label_to_yolo(src_lbl, DST / split / "labels" / f"{stem}.txt")
                count += 1
        print(f"Copied {count} samples to {DST / split}")

    # Write data.yaml
    yaml_content = f"""\
train: {os.path.abspath(DST / 'train' / 'images')}
val: {os.path.abspath(DST / 'valid' / 'images')}

nc: 1
names:
  0: stenosis
"""
    (DST / "data.yaml").write_text(yaml_content)
    print(f"Wrote {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()
