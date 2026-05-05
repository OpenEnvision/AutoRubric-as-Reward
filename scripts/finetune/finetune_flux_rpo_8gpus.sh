#!/usr/bin/env bash
set -euo pipefail

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE="${WANDB_MODE:-online}"

GPU_NUM="${GPU_NUM:-8}"
MASTER_PORT="${MASTER_PORT:-19002}"
MODEL_PATH="${MODEL_PATH:-data/flux}"
DATA_JSON_PATH="${DATA_JSON_PATH:-data/rl_embeddings/flux/metadata.json}"
ARR_CONFIG_PATH="${ARR_CONFIG_PATH:-rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/rpo_flux_lora}"

mkdir -p images "${OUTPUT_DIR}"

torchrun --nproc_per_node="${GPU_NUM}" --master_port "${MASTER_PORT}" \
    fastvideo/train_rpo_flux.py \
    --seed 42 \
    --pretrained_model_name_or_path "${MODEL_PATH}" \
    --vae_model_path "${MODEL_PATH}" \
    --cache_dir data/.cache \
    --data_json_path "${DATA_JSON_PATH}" \
    --use_arr \
    --arr_config_path "${ARR_CONFIG_PATH}" \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 300 \
    --learning_rate 5e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 50 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir "${OUTPUT_DIR}" \
    --h 512 \
    --w 512 \
    --t 1 \
    --sampling_steps 8 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 2 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 1.0 \
    --clip_range 0.2 \
    --adv_clip_max 5.0 \
    --kl_beta 0.01 \
    --init_same_noise \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules "to_k,to_q,to_v,to_out.0"
