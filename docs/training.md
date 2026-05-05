# Training Guide

This page focuses on preprocessing and RPO launchers. Before a long training run,
generate or select a rubric source through the [Auto-Rubric Guide](auto_rubric/README.md).
For deterministic runs, prefer a saved `rubrics_file`; see [Rubric Reuse](auto_rubric/rubric_reuse.md).

## FLUX

1. Download FLUX.1-dev into `data/flux`.
2. Start or configure the Auto-Rubric VLM judge.
3. Preprocess prompt embeddings:

```bash
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
```

4. Launch 8-GPU RPO:

```bash
ARR_CONFIG_PATH=rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml \
bash scripts/finetune/finetune_flux_rpo_8gpus.sh
```

Important defaults:

- output directory: `data/outputs/rpo_flux_lora`
- metadata: `data/rl_embeddings/flux/metadata.json`
- resolution: `512x512`
- learning rate: `5e-5`
- sampling steps: `8`
- PPO clip range: `0.2`
- KL coefficient: `0.01`
- LoRA rank: `16`
- reward: pairwise Auto-Rubric rank reward

## Qwen-Image-Edit

1. Download Qwen-Image-Edit into `data/qwenimage_edit`.
2. Prepare an edit JSONL and a directory of source images.
3. Preprocess edit instruction and source-image embeddings:

```bash
bash scripts/preprocess/preprocess_qwen_image_edit_rl_embeddings.sh
```

4. Launch 8-GPU RPO:

```bash
ARR_CONFIG_PATH=rubric_pipeline/config/qwen3vl_8B_instruct_edit.yaml \
bash scripts/finetune/finetune_qwen_image_edit_rpo_8gpus.sh
```

Important defaults:

- output directory: `data/outputs/rpo_qwen_image_edit_lora`
- metadata: `data/rl_embeddings/qwenimage_edit/metadata.json`
- resolution: `512x512`
- learning rate: `1e-5`
- sampling steps: `10`
- PPO clip range: `0.2`
- KL coefficient: `0.02`
- LoRA rank: `32`
- reward: pairwise Auto-Rubric rank reward with source-image context

## Reusing Saved Rubrics

For large training runs, generate rubrics once and save them:

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml \
  --seed_dataset path/to/seed_pairs.json \
  --test_dataset path/to/dev_pairs.json \
  --base_url http://localhost:8000/v1
```

Put the final rubric text into `rubric_pipeline/rubrics/flux_t2i.txt`, then add this key to the YAML config:

```yaml
rubrics_file: "rubric_pipeline/rubrics/flux_t2i.txt"
```

This avoids regenerating rubrics inside every training launch.

## Debugging

- Set `WANDB_MODE=offline` for local dry runs.
- Reduce `max_train_steps` and `sampling_steps` in the launcher for smoke tests.
- Use `concurrency_limit` below the number of requests your VLM server can handle.
- For pairwise RPO, keep `num_generations=2`; changing it requires changing reward mapping.
