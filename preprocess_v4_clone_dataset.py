"""
Clone combined_arcade2x_trainval_dataset2 and apply preprocess_contrast_fast_v4 to all images.
Labels are copied as-is. data.yaml is updated to point to the new location.
"""

from pathlib import Path
import numpy as np
import cv2
import shutil
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

SRC = Path("/home/dsa/stenosis/data/combined_arcade2x_trainval_dataset2")
DST = Path("/home/dsa/stenosis/data/combined_arcade2x_trainval_dataset2_prepv4")


def preprocess_contrast_fast_v4(img):
    gray = img[:, :, 0]
    se = np.ones((50, 50), np.uint8)
    img_not = cv2.bitwise_not(gray)
    eroded = ndimage.grey_erosion(img_not, footprint=se, mode='reflect')
    opened = ndimage.grey_dilation(eroded, footprint=se, mode='reflect')
    wth = img_not.astype(np.int32) - opened.astype(np.int32)
    wth = np.clip(wth, 0, 255).astype(np.uint8)
    raw_minus_topwhite = gray.astype(np.int32) - wth.astype(np.int32)
    raw_minus_topwhite = ((raw_minus_topwhite > 0) * raw_minus_topwhite).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res = clahe.apply(raw_minus_topwhite)
    return cv2.merge([res, res, res])


def process_image(src_path: Path, dst_path: Path):
    img = cv2.imread(str(src_path))
    if img is None:
        print(f"  WARNING: could not read {src_path}")
        return
    processed = preprocess_contrast_fast_v4(img)
    cv2.imwrite(str(dst_path), processed)


def clone_and_preprocess():
    if DST.exists():
        print(f"Destination {DST} already exists. Remove it first if you want to re-run.")
        return

    splits = ["train", "valid", "test"]

    # Gather all image tasks
    tasks = []
    for split in splits:
        src_img_dir = SRC / split / "images"
        src_lbl_dir = SRC / split / "labels"
        dst_img_dir = DST / split / "images"
        dst_lbl_dir = DST / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Copy labels as-is
        if src_lbl_dir.exists():
            for lbl in src_lbl_dir.iterdir():
                shutil.copy2(lbl, dst_lbl_dir / lbl.name)

        # Collect image tasks
        if src_img_dir.exists():
            for img_path in src_img_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    tasks.append((img_path, dst_img_dir / img_path.name))

    print(f"Processing {len(tasks)} images with 8 workers...")
    done = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_image, s, d): (s, d) for s, d in tasks}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)}")

    # Write updated data.yaml
    yaml_content = (
        f"train: {DST}/train/images\n"
        f"val: {DST}/valid/images\n"
        f"test: {DST}/test/images\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['stenosis']\n"
    )
    (DST / "data.yaml").write_text(yaml_content)

    print(f"\nDone! Cloned and preprocessed dataset at:\n  {DST}")


if __name__ == "__main__":
    clone_and_preprocess()
