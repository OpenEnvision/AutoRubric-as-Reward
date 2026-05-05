#!/usr/bin/env bash
set -euo pipefail

GPU_NUM="${GPU_NUM:-8}"
MASTER_PORT="${MASTER_PORT:-19013}"
MODEL_PATH="${MODEL_PATH:-data/qwenimage_edit}"
EDIT_JSONL="${EDIT_JSONL:-assets/edit_data.jsonl}"
BASE_IMAGE_DIR="${BASE_IMAGE_DIR:-data/edit_images}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl_embeddings/qwenimage_edit}"

mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node="${GPU_NUM}" --master_port "${MASTER_PORT}" \
    fastvideo/data_preprocess/preprocess_qwenimage_edit_embeddings.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_dir "${EDIT_JSONL}" \
    --base_image_dir "${BASE_IMAGE_DIR}" \
    --height 512 \
    --width 512 \
    --train_batch_size 1 \
    --dataloader_num_workers 1
