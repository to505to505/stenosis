#!/bin/bash
# Download dataset2_split from Hugging Face to data/dataset2_split

set -e

REPO_ID="to505to505/dataset2"
TARGET_DIR="data/dataset2_split"

if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR")" ]; then
    echo "$TARGET_DIR already exists and is not empty. Skipping download."
    exit 0
fi

mkdir -p "$TARGET_DIR"

echo "Downloading $REPO_ID to $TARGET_DIR ..."
huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir "$TARGET_DIR"

echo "Done! Dataset saved to $TARGET_DIR"
