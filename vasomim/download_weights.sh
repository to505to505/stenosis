#!/bin/bash
# Download VasoMIM pretrained ViT-Small encoder weights from Hugging Face

WEIGHTS_DIR="$(dirname "$0")/weights"
WEIGHTS_FILE="$WEIGHTS_DIR/vit_small_encoder_512.pth"
URL="https://huggingface.co/to505to505/vit_s_vasomim/resolve/main/vit_small_encoder_512.pth"

mkdir -p "$WEIGHTS_DIR"

if [ -f "$WEIGHTS_FILE" ]; then
    echo "Weights already exist at $WEIGHTS_FILE"
    exit 0
fi

echo "Downloading VasoMIM ViT-Small weights..."
if command -v wget &> /dev/null; then
    wget -O "$WEIGHTS_FILE" "$URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$WEIGHTS_FILE" "$URL"
else
    echo "Error: neither wget nor curl found" >&2
    exit 1
fi

echo "Saved to $WEIGHTS_FILE"
