#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_DIR="${SCRIPT_DIR}/checkpoints"
IMAGENET_DIR="${SCRIPT_DIR}/data"
NUM_GPUS=1
BATCH_SIZE_PER_GPU=512

echo "Starting 8-GPU training..."
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Total batch size: $((NUM_GPUS * BATCH_SIZE_PER_GPU))"
echo "Learning rate: 1.5e-4 (fixed)"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port 29501 \
    main_pretrain.py \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --model mae_vit_small_patch16 \
    --mask_ratio 0.50 \
    --norm_pix_loss \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path "${IMAGENET_DIR}" \
    --output_dir "${JOB_DIR}" \