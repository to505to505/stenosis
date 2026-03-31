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

# Ensure huggingface_hub is installed
pip install -q huggingface_hub 2>/dev/null || true

echo "Downloading $REPO_ID to $TARGET_DIR ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$REPO_ID', repo_type='dataset', local_dir='$TARGET_DIR')
"

echo "Done! Dataset saved to $TARGET_DIR"
