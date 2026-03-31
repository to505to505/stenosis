from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)

# CADICA stenosis severity classes (ordered by severity)
CADICA_CLASS_MAP: Dict[str, int] = {
    "p0_20": 0,
    "p20_50": 1,
    "p50_70": 2,
    "p70_90": 3,
    "p90_98": 4,
    "p99": 5,
    "p100": 6,
}
CADICA_CLASS_NAMES = [
    "stenosis_0_20",
    "stenosis_20_50",
    "stenosis_50_70",
    "stenosis_70_90",
    "stenosis_90_98",
    "stenosis_99",
    "stenosis_100",
]

Sample = Tuple[Path, Path]
PatientGroups = Dict[str, List[Sample]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the CADICA dataset by patient so that the same patient never "
            "appears in different train/valid/test splits.  Optionally converts "
            "labels from native CADICA format (class_name x y w h, absolute pixels) "
            "to YOLO format (class_id xc yc w h, normalised)."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/dsa/stenosis/data/cadica_base"),
        help="Path to the source CADICA directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory where train/, valid/ and test/ will be created. "
            "Default: <dataset-dir>/../cadica_split"
        ),
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALID", "TEST"),
        default=DEFAULT_RATIOS,
        help="Split ratios for train/valid/test. Default: 0.70 0.15 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before greedy assignment. Default: 42",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move image files instead of copying them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing split directories before writing new files.",
    )
    parser.add_argument(
        "--no-convert-labels",
        action="store_true",
        help=(
            "Skip YOLO label conversion: copy labels as-is in the original "
            "CADICA format (class_name x y w h, absolute pixels)."
        ),
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(512, 512),
        help="Image width and height in pixels for normalisation. Default: 512 512",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=None,
        help=(
            "Where to write a YOLO data yaml for the new split. "
            "Default: <output-root>/data_patient_split.yaml"
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help=(
            "Where to write JSON with patient-to-split mapping. "
            "Default: <output-root>/patient_split_manifest.json"
        ),
    )
    return parser.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────


def validate_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != 3:
        raise ValueError("Exactly three ratios are required: train valid test.")
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be positive.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0.")


def extract_patient_id(filename: str) -> str:
    """Return the patient token from a CADICA filename.

    CADICA files follow the pattern ``p<N>_v<M>_<frame>.ext``.
    The patient id is the first underscore-separated token (e.g. ``p1``).
    """
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Cannot extract patient id from '{filename}': "
            "expected at least 3 underscore-separated tokens (p<N>_v<M>_<frame>)."
        )
    return parts[0]


def collect_samples(dataset_dir: Path) -> PatientGroups:
    """Walk images/ and pair each image with its label file, grouped by patient."""
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    patient_to_samples: PatientGroups = defaultdict(list)
    missing_labels: List[Path] = []

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(label_path)
            continue

        patient_id = extract_patient_id(image_path.name)
        patient_to_samples[patient_id].append((image_path, label_path))

    if missing_labels:
        preview = "\n".join(str(p) for p in missing_labels[:10])
        raise FileNotFoundError(
            "Missing label files for some images. First missing labels:\n" + preview
        )

    if not patient_to_samples:
        raise RuntimeError(f"No samples found in {images_dir}")

    return dict(patient_to_samples)


# ── splitting ────────────────────────────────────────────────────────────────


def assign_patients_to_splits(
    patient_to_samples: PatientGroups,
    ratios: Sequence[float],
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Sample]]]:
    """Greedy assignment of patients to splits minimising chi-squared deviation."""
    rng = random.Random(seed)

    patients = list(patient_to_samples.items())
    rng.shuffle(patients)
    patients.sort(key=lambda item: len(item[1]), reverse=True)

    total_samples = sum(len(samples) for _, samples in patients)
    target_counts = {
        split: total_samples * ratio for split, ratio in zip(SPLITS, ratios)
    }

    split_patients: Dict[str, List[str]] = {s: [] for s in SPLITS}
    split_samples: Dict[str, List[Sample]] = {s: [] for s in SPLITS}
    current_counts = {s: 0 for s in SPLITS}

    for patient_id, samples in patients:
        n = len(samples)

        def cost(candidate: str) -> float:
            sim = current_counts.copy()
            sim[candidate] += n
            return sum(
                ((sim[s] - target_counts[s]) ** 2) / max(target_counts[s], 1.0)
                for s in SPLITS
            )

        best = min(SPLITS, key=cost)
        split_patients[best].append(patient_id)
        split_samples[best].extend(samples)
        current_counts[best] += n

    return split_patients, split_samples


# ── label conversion ─────────────────────────────────────────────────────────


def convert_cadica_label_to_yolo(
    src_label: Path,
    dst_label: Path,
    img_w: int,
    img_h: int,
) -> None:
    """Convert a single CADICA label file to YOLO format.

    CADICA format per line:  ``class_name  x  y  w  h``  (absolute pixels, top-left)
    YOLO  format per line:   ``class_id  xc  yc  w  h``  (normalised centre-based)
    """
    lines: List[str] = []
    text = src_label.read_text(encoding="utf-8").strip()
    if not text:
        # Empty label → negative frame (no annotations)
        dst_label.write_text("", encoding="utf-8")
        return

    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            # Skip malformed lines
            continue

        cls_name = parts[0]
        if cls_name not in CADICA_CLASS_MAP:
            raise ValueError(
                f"Unknown CADICA class '{cls_name}' in {src_label}. "
                f"Known classes: {list(CADICA_CLASS_MAP)}"
            )

        cls_id = CADICA_CLASS_MAP[cls_name]
        x, y, w, h = (float(v) for v in parts[1:])

        # Convert top-left absolute → centre normalised
        xc = (x + w / 2.0) / img_w
        yc = (y + h / 2.0) / img_h
        wn = w / img_w
        hn = h / img_h

        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ── I/O ──────────────────────────────────────────────────────────────────────


def prepare_output_dirs(output_root: Path, force: bool) -> Dict[str, Dict[str, Path]]:
    layout: Dict[str, Dict[str, Path]] = {}

    for split in SPLITS:
        split_dir = output_root / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        layout[split] = {"root": split_dir, "images": images_dir, "labels": labels_dir}

        if split_dir.exists():
            if force:
                shutil.rmtree(split_dir)
            else:
                raise FileExistsError(
                    f"Output directory already exists: {split_dir}. "
                    "Use --force to overwrite it."
                )

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    return layout


def transfer_samples(
    split_samples: Dict[str, List[Sample]],
    layout: Dict[str, Dict[str, Path]],
    move_images: bool,
    convert_labels: bool,
    img_w: int,
    img_h: int,
) -> None:
    img_fn = shutil.move if move_images else shutil.copy2

    for split, samples in split_samples.items():
        for image_path, label_path in samples:
            img_fn(image_path, layout[split]["images"] / image_path.name)

            dst_label = layout[split]["labels"] / label_path.name
            if convert_labels:
                convert_cadica_label_to_yolo(label_path, dst_label, img_w, img_h)
            else:
                shutil.copy2(label_path, dst_label)


def write_yaml(
    output_root: Path,
    yaml_path: Path,
    nc: int,
    names: List[str],
) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    train_rel = Path(
        os.path.relpath(output_root / "train" / "images", yaml_path.parent)
    ).as_posix()
    valid_rel = Path(
        os.path.relpath(output_root / "valid" / "images", yaml_path.parent)
    ).as_posix()
    test_rel = Path(
        os.path.relpath(output_root / "test" / "images", yaml_path.parent)
    ).as_posix()

    yaml_text = (
        f"train: {train_rel}\n"
        f"val: {valid_rel}\n"
        f"test: {test_rel}\n\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")


def write_manifest(
    manifest_path: Path,
    split_patients: Dict[str, List[str]],
    split_samples: Dict[str, List[Sample]],
    ratios: Sequence[float],
    seed: int,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ratios": {s: r for s, r in zip(SPLITS, ratios)},
        "seed": seed,
        "splits": {
            s: {
                "patients": split_patients[s],
                "patient_count": len(split_patients[s]),
                "image_count": len(split_samples[s]),
            }
            for s in SPLITS
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(
    split_patients: Dict[str, List[str]],
    split_samples: Dict[str, List[Sample]],
) -> None:
    total_patients = sum(len(v) for v in split_patients.values())
    total_images = sum(len(v) for v in split_samples.values())

    print(f"Total patients: {total_patients}")
    print(f"Total images:   {total_images}")

    for split in SPLITS:
        pc = len(split_patients[split])
        ic = len(split_samples[split])
        ps = pc / total_patients if total_patients else 0.0
        iss = ic / total_images if total_images else 0.0
        print(
            f"  {split:5s}: patients={pc:3d} ({ps:.1%}), "
            f"images={ic:4d} ({iss:.1%})  {split_patients[split]}"
        )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    validate_ratios(args.ratios)

    dataset_dir = args.dataset_dir.resolve()
    output_root = (
        args.output_root or (dataset_dir.parent / "cadica_split")
    ).resolve()
    yaml_path = (
        args.yaml_path or (output_root / "data_patient_split.yaml")
    ).resolve()
    manifest_path = (
        args.manifest_path or (output_root / "patient_split_manifest.json")
    ).resolve()

    if output_root == dataset_dir:
        raise ValueError(
            "--output-root must differ from --dataset-dir to avoid overwriting sources."
        )

    convert_labels = not args.no_convert_labels
    img_w, img_h = args.img_size

    # ── collect & split ──────────────────────────────────────────────────
    patient_to_samples = collect_samples(dataset_dir)
    split_patients, split_samples = assign_patients_to_splits(
        patient_to_samples=patient_to_samples,
        ratios=args.ratios,
        seed=args.seed,
    )

    # ── write files ──────────────────────────────────────────────────────
    layout = prepare_output_dirs(output_root, args.force)
    transfer_samples(
        split_samples, layout, args.move, convert_labels, img_w, img_h
    )

    write_yaml(output_root, yaml_path, nc=len(CADICA_CLASS_MAP), names=CADICA_CLASS_NAMES)
    write_manifest(manifest_path, split_patients, split_samples, args.ratios, args.seed)

    # ── report ───────────────────────────────────────────────────────────
    print_summary(split_patients, split_samples)
    label_fmt = "YOLO (normalised)" if convert_labels else "CADICA (raw)"
    print(f"Label format:        {label_fmt}")
    print(f"YOLO config written: {yaml_path}")
    print(f"Split manifest:      {manifest_path}")


if __name__ == "__main__":
    main()
