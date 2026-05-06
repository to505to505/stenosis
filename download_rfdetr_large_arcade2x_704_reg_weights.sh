#!/bin/bash
# Download rfdetr_large_arcade2x_704_reg checkpoint from Backblaze B2

set -e

BUCKET="stenosis"
REMOTE_FILE="weights/rfdetr_large_arcade2x_704/checkpoint_best_total.pth"
TARGET_FILE="rfdetr_runs/rfdetr_large_arcade2x_704_reg/checkpoint_best_total.pth"

if [ -f "$TARGET_FILE" ]; then
    echo "$TARGET_FILE already exists. Skipping download."
    exit 0
fi

mkdir -p "$(dirname "$TARGET_FILE")"

echo "Downloading b2://$BUCKET/$REMOTE_FILE to $TARGET_FILE ..."
b2 download-file-by-name "$BUCKET" "$REMOTE_FILE" "$TARGET_FILE"

echo "Done! Checkpoint saved to $TARGET_FILE"
