# Data Preprocessing

This repo keeps preprocessing only for FLUX text-to-image and Qwen-Image-Edit.

## FLUX Text-To-Image

Input prompts are read from `assets/prompts.txt` by default. Each line is one prompt.

```bash
GPU_NUM=8 \
MODEL_PATH=data/flux \
PROMPT_FILE=assets/prompts.txt \
OUTPUT_DIR=data/rl_embeddings/flux \
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
```

The script writes:

- `prompt_embed/*.pt`
- `pooled_prompt_embeds/*.pt`
- `text_ids/*.pt`
- `metadata.json`

`metadata.json` contains image prompt metadata for the FLUX training dataloader.

## Qwen-Image-Edit

The default input format is JSONL, matching `assets/edit_data.jsonl`:

```json
{"source_image": "images/0001_0.jpg", "instruction": "Add a red hat."}
```

Launch preprocessing:

```bash
GPU_NUM=8 \
MODEL_PATH=data/qwenimage_edit \
EDIT_JSONL=assets/edit_data.jsonl \
BASE_IMAGE_DIR=data/edit_images \
OUTPUT_DIR=data/rl_embeddings/qwenimage_edit \
bash scripts/preprocess/preprocess_qwen_image_edit_rl_embeddings.sh
```

The script writes:

- `prompt_embed/*.pt`
- `prompt_attention_mask/*.pt`
- `image_latents/*.pt`
- `metadata.json`

The metadata includes the edit instruction, source image path, original sequence length, and resize dimensions used by the Qwen-Image-Edit pipeline.
