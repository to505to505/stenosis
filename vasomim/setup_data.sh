#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
ZIP_URL="https://huggingface.co/datasets/waha2000huang/XA-170K/resolve/main/xa170k.zip"
ZIP_FILE="${DATA_DIR}/xa170k.zip"

echo "=== Step 1: Download XA-170K dataset ==="
mkdir -p "${DATA_DIR}"

if [ -f "${ZIP_FILE}" ]; then
    echo "Archive already exists, skipping download."
else
    echo "Downloading from ${ZIP_URL} ..."
    wget -O "${ZIP_FILE}" "${ZIP_URL}"
fi

echo "=== Step 2: Extract archive ==="
unzip -o "${ZIP_FILE}" -d "${DATA_DIR}"

# Move contents from dataset/ subfolder to data/ if needed
if [ -d "${DATA_DIR}/dataset" ]; then
    echo "Moving contents from dataset/ to data/ ..."
    for dir in "${DATA_DIR}/dataset"/*/; do
        name="$(basename "$dir")"
        if [ -d "${DATA_DIR}/${name}" ]; then
            echo "  ${name}/ already exists, skipping."
        else
            mv "$dir" "${DATA_DIR}/${name}"
        fi
    done
    rmdir "${DATA_DIR}/dataset" 2>/dev/null || true
fi

echo "=== Step 3: Clean up zip ==="
rm -f "${ZIP_FILE}"

echo "=== Step 4: Generate Frangi filter masks ==="
cd "${SCRIPT_DIR}"
python frangi_filter.py

echo "=== Done! ==="
echo "Data directory structure:"
ls -d "${DATA_DIR}"/*/
