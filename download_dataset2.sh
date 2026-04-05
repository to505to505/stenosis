#!/bin/bash
# Download dataset2_split from Backblaze B2 to data/dataset2_split

set -e

BUCKET="stenosis"
REMOTE_PATH="dataset2_split"
TARGET_DIR="data/dataset2_split"

if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR")" ]; then
    echo "$TARGET_DIR already exists and is not empty. Skipping download."
    exit 0
fi

mkdir -p "$TARGET_DIR"

echo "Downloading b2://$BUCKET/$REMOTE_PATH to $TARGET_DIR ..."
b2 sync --threads 16 "b2://$BUCKET/$REMOTE_PATH" "$TARGET_DIR"

echo "Done! Dataset saved to $TARGET_DIR"
