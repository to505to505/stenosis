"""
Convert all stenosis classes in combined_arcade_full_dataset2 to a single class (0).

Modifies:
  - All .txt label files in train/labels, valid/labels, test/labels
  - data.yaml (nc: 1, names: ['stenosis'])
"""

import os
import glob
import yaml

DATASET_DIR = "/workspace/stenosis/data/combined_arcade_full_dataset2"


def convert_labels(dataset_dir: str):
    label_dirs = [
        os.path.join(dataset_dir, split, "labels")
        for split in ("train", "valid", "test")
    ]

    total_files = 0
    total_changed = 0

    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            print(f"Skipping (not found): {label_dir}")
            continue

        txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
        for txt_path in txt_files:
            total_files += 1
            with open(txt_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            changed = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] != "0":
                    parts[0] = "0"
                    changed = True
                new_lines.append(" ".join(parts))

            if changed:
                total_changed += 1
                with open(txt_path, "w") as f:
                    f.write("\n".join(new_lines))
                    if new_lines:
                        f.write("\n")

    print(f"Processed {total_files} label files, changed {total_changed}")


def update_data_yaml(dataset_dir: str):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    data["nc"] = 1
    data["names"] = ["stenosis"]

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Updated {yaml_path}: nc=1, names=['stenosis']")


if __name__ == "__main__":
    convert_labels(DATASET_DIR)
    update_data_yaml(DATASET_DIR)
    print("Done.")
