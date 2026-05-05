#!/usr/bin/env bash
set -euo pipefail

GPU_NUM="${GPU_NUM:-8}"
MASTER_PORT="${MASTER_PORT:-19012}"
MODEL_PATH="${MODEL_PATH:-data/flux}"
PROMPT_FILE="${PROMPT_FILE:-assets/prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-data/rl_embeddings/flux}"

mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node="${GPU_NUM}" --master_port "${MASTER_PORT}" \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --prompt_dir "${PROMPT_FILE}" \
    --train_batch_size 1 \
    --dataloader_num_workers 1
