#!/bin/bash
mkdir -p "$(dirname "$0")/weights"
wget -O "$(dirname "$0")/weights/ViTAE-S-GPU.pth" \
  "https://huggingface.co/to505to505/vitDetSmall/resolve/main/ViTAE-S-GPU.pth"
