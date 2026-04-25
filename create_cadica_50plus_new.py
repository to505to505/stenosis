#!/usr/bin/env python3
"""
Create cadica_50plus_new from cadica_base.

1. Find all video sequences (p{patient}_v{video}) that have at least one frame
   with stenosis >= 50% (p50_70, p70_90, p90_98, p99, p100).
2. Keep ALL frames from those sequences.
3. Remap all 50+ stenosis classes to label 0; drop annotations with class < 50%.

Label format: <class_name> <x> <y> <w> <h>  ->  0 <x> <y> <w> <h>
"""

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

BASE = Path("/home/dsa/stenosis/data")
SRC = BASE / "cadica_base"
DST = BASE / "cadica_50plus_new"

IMG_W, IMG_H = 512, 512  # CADICA image dimensions

# Classes considered 50+
CLASSES_50PLUS = {"p50_70", "p70_90", "p90_98", "p99", "p100"}


def get_sequence(filename: str) -> str:
    """Extract sequence prefix from filename like p1_v3_00012.txt -> p1_v3"""
    match = re.match(r"^(p\d+_v\d+)_\d+", filename)
    return match.group(1) if match else None


def main():
    src_img = SRC / "images"
    src_lbl = SRC / "labels"

    # Group label files by sequence
    seq_files = defaultdict(list)
    for lbl_file in sorted(src_lbl.glob("*.txt")):
        if "endpointdlp" in lbl_file.name:
            continue
        seq = get_sequence(lbl_file.stem)
        if seq:
            seq_files[seq].append(lbl_file)

    print(f"Total sequences: {len(seq_files)}")

    # Find sequences with at least one 50+ annotation
    seqs_with_50plus = set()
    for seq, files in seq_files.items():
        for lbl_file in files:
            text = lbl_file.read_text().strip()
            if not text:
                continue
            for line in text.split("\n"):
                cls_name = line.strip().split()[0]
                if cls_name in CLASSES_50PLUS:
                    seqs_with_50plus.add(seq)
                    break
            if seq in seqs_with_50plus:
                break

    print(f"Sequences with 50+ stenosis: {len(seqs_with_50plus)}")

    # Create destination
    dst_img = DST / "images"
    dst_lbl = DST / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    kept_images = 0
    kept_annotations = 0
    dropped_annotations = 0

    for seq in sorted(seqs_with_50plus):
        for lbl_file in seq_files[seq]:
            stem = lbl_file.stem

            # Process label: keep only 50+ annotations, remap to class 0
            text = lbl_file.read_text().strip()
            new_lines = []
            if text:
                for line in text.split("\n"):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_name = parts[0]
                    if cls_name in CLASSES_50PLUS:
                        # Convert pixel x,y,w,h to YOLO normalized cx,cy,w,h
                        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        cx = (x + w / 2.0) / IMG_W
                        cy = (y + h / 2.0) / IMG_H
                        wn = w / IMG_W
                        hn = h / IMG_H
                        new_lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
                        kept_annotations += 1
                    else:
                        dropped_annotations += 1

            # Write label (empty file = negative/background)
            (dst_lbl / lbl_file.name).write_text(
                "\n".join(new_lines) + ("\n" if new_lines else "")
            )

            # Symlink image
            img_name = stem + ".png"
            src_img_path = src_img / img_name
            dst_img_path = dst_img / img_name
            if src_img_path.exists() and not dst_img_path.exists():
                os.symlink(src_img_path.resolve(), dst_img_path)
            kept_images += 1

    print(f"\nResult:")
    print(f"  Images: {kept_images}")
    print(f"  Kept annotations (50+): {kept_annotations}")
    print(f"  Dropped annotations (<50): {dropped_annotations}")

    # Write data.yaml
    yaml_content = f"""train: {DST}/images
val: {DST}/images

nc: 1
names: ['stenosis']
"""
    (DST / "data.yaml").write_text(yaml_content)
    print(f"  Wrote {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()
