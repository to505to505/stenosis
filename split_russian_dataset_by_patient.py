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

Sample = Tuple[Path, Path]
PatientGroups = Dict[str, List[Sample]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a YOLO dataset by patient so that the same patient never appears "
            "in different train/valid/test splits."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/home/dsa/stenosis/data/dataset2_base"),
        help="Path to the source dataset directory containing images/ and labels/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory where train/, valid/ and test/ will be created. "
            "Default: parent directory of --dataset-dir."
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
        "--patient-tokens",
        type=int,
        default=2,
        help="Number of underscore-separated tokens to treat as patient id. Default: 2",
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
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing split directories before writing new files.",
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


def validate_ratios(ratios: Sequence[float]) -> None:
    if len(ratios) != 3:
        raise ValueError("Exactly three ratios are required: train valid test.")
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be positive.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0.")


def extract_patient_id(filename: str, patient_tokens: int) -> str:
    parts = filename.split("_")
    if len(parts) < patient_tokens:
        raise ValueError(
            f"Cannot extract patient id from '{filename}': expected at least {patient_tokens} underscore-separated tokens."
        )
    return "_".join(parts[:patient_tokens])


def collect_samples(dataset_dir: Path, patient_tokens: int) -> PatientGroups:
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

        patient_id = extract_patient_id(image_path.name, patient_tokens)
        patient_to_samples[patient_id].append((image_path, label_path))

    if missing_labels:
        preview = "\n".join(str(path) for path in missing_labels[:10])
        raise FileNotFoundError(
            "Missing label files for some images. First missing labels:\n" + preview
        )

    if not patient_to_samples:
        raise RuntimeError(f"No samples found in {images_dir}")

    return dict(patient_to_samples)


def assign_patients_to_splits(
    patient_to_samples: PatientGroups,
    ratios: Sequence[float],
    seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Sample]]]:
    rng = random.Random(seed)

    patients = list(patient_to_samples.items())
    rng.shuffle(patients)
    patients.sort(key=lambda item: len(item[1]), reverse=True)

    total_samples = sum(len(samples) for _, samples in patients)
    target_counts = {
        split: total_samples * ratio for split, ratio in zip(SPLITS, ratios)
    }

    split_patients: Dict[str, List[str]] = {split: [] for split in SPLITS}
    split_samples: Dict[str, List[Sample]] = {split: [] for split in SPLITS}
    current_counts = {split: 0 for split in SPLITS}

    for patient_id, samples in patients:
        patient_sample_count = len(samples)

        def cost(candidate_split: str) -> float:
            simulated_counts = current_counts.copy()
            simulated_counts[candidate_split] += patient_sample_count
            return sum(
                ((simulated_counts[split] - target_counts[split]) ** 2)
                / max(target_counts[split], 1.0)
                for split in SPLITS
            )

        best_split = min(SPLITS, key=cost)
        split_patients[best_split].append(patient_id)
        split_samples[best_split].extend(samples)
        current_counts[best_split] += patient_sample_count

    return split_patients, split_samples


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
                    f"Output directory already exists: {split_dir}. Use --force to overwrite it."
                )

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    return layout


def transfer_samples(
    split_samples: Dict[str, List[Sample]],
    layout: Dict[str, Dict[str, Path]],
    move_files: bool,
) -> None:
    copy_fn = shutil.move if move_files else shutil.copy2

    for split, samples in split_samples.items():
        for image_path, label_path in samples:
            copy_fn(image_path, layout[split]["images"] / image_path.name)
            copy_fn(label_path, layout[split]["labels"] / label_path.name)


def read_source_metadata(source_yaml_path: Path) -> Tuple[str, str]:
    default_nc = "1"
    default_names = "['Stenosis']"

    if not source_yaml_path.exists():
        return default_nc, default_names

    nc = default_nc
    names = default_names

    for line in source_yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("nc:"):
            nc = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("names:"):
            names = stripped.split(":", 1)[1].strip()

    return nc, names


def write_yaml(output_root: Path, yaml_path: Path, source_yaml_path: Path) -> None:
    nc, names = read_source_metadata(source_yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    train_rel = Path(os.path.relpath(output_root / "train" / "images", yaml_path.parent)).as_posix()
    valid_rel = Path(os.path.relpath(output_root / "valid" / "images", yaml_path.parent)).as_posix()
    test_rel = Path(os.path.relpath(output_root / "test" / "images", yaml_path.parent)).as_posix()

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
        "ratios": {
            split: ratio for split, ratio in zip(SPLITS, ratios)
        },
        "seed": seed,
        "splits": {
            split: {
                "patients": split_patients[split],
                "patient_count": len(split_patients[split]),
                "image_count": len(split_samples[split]),
            }
            for split in SPLITS
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
    total_patients = sum(len(patients) for patients in split_patients.values())
    total_images = sum(len(samples) for samples in split_samples.values())

    print(f"Total patients: {total_patients}")
    print(f"Total images: {total_images}")

    for split in SPLITS:
        patient_count = len(split_patients[split])
        image_count = len(split_samples[split])
        patient_share = patient_count / total_patients if total_patients else 0.0
        image_share = image_count / total_images if total_images else 0.0
        print(
            f"{split}: patients={patient_count} ({patient_share:.1%}), "
            f"images={image_count} ({image_share:.1%})"
        )


def main() -> None:
    args = parse_args()
    validate_ratios(args.ratios)

    dataset_dir = args.dataset_dir.resolve()
    output_root = (args.output_root or dataset_dir.parent).resolve()
    yaml_path = (args.yaml_path or (output_root / "data_patient_split.yaml")).resolve()
    manifest_path = (
        args.manifest_path or (output_root / "patient_split_manifest.json")
    ).resolve()

    if output_root == dataset_dir:
        raise ValueError(
            "--output-root must be different from --dataset-dir, otherwise source files may be overwritten."
        )

    patient_to_samples = collect_samples(dataset_dir, args.patient_tokens)
    split_patients, split_samples = assign_patients_to_splits(
        patient_to_samples=patient_to_samples,
        ratios=args.ratios,
        seed=args.seed,
    )

    layout = prepare_output_dirs(output_root, args.force)
    transfer_samples(split_samples, layout, args.move)
    write_yaml(output_root, yaml_path, dataset_dir / "data.yaml")
    write_manifest(manifest_path, split_patients, split_samples, args.ratios, args.seed)
    print_summary(split_patients, split_samples)
    print(f"YOLO config written to: {yaml_path}")
    print(f"Split manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
