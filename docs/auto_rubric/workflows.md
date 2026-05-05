# Auto-Rubric Workflows

This page collects common commands. Replace paths and endpoints with your local setup.

## 1. Local Judge Smoke Test

Start the local VLM:

```bash
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct TP_SIZE=1 PORT=8000 \
  bash rubric_pipeline/vllm_serve.sh
```

Set the key expected by the OpenAI client:

```bash
export OPENAI_API_KEY=EMPTY
```

Generate and test T2I rubrics:

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml \
  --seed_dataset examples/seed_t2i_pairwise.json \
  --test_dataset examples/test_t2i_pairwise.json \
  --base_url http://localhost:8000/v1 \
  --concurrency_limit 2
```

## 2. Generate Image-Edit Rubrics

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_edit.yaml \
  --seed_dataset examples/seed_edit_pairwise.json \
  --test_dataset examples/test_edit_pairwise.json \
  --base_url http://localhost:8000/v1 \
  --concurrency_limit 2
```

Image-edit samples should provide a `source_image` plus two edited outputs. The rank applies only to edited candidates.

## 3. Save And Reuse A Rubric File

Create the folder:

```bash
mkdir -p rubric_pipeline/rubrics
```

Save the generated rubric text as:

```text
rubric_pipeline/rubrics/flux_t2i_general_v1.txt
```

Use it in config:

```yaml
rubrics_file: "rubric_pipeline/rubrics/flux_t2i_general_v1.txt"
```

Evaluate the saved rubric:

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml \
  --rubrics_file rubric_pipeline/rubrics/flux_t2i_general_v1.txt \
  --test_dataset examples/test_t2i_pairwise.json \
  --base_url http://localhost:8000/v1 \
  --concurrency_limit 4
```

## 4. Use Hosted VLMs

Use a config like:

```yaml
model_name: "gpt-5"
base_url: "https://api.openai.com/v1"
api_key: "${OPENAI_API_KEY}"
```

Then run the same `judger.py` command. The code expands environment variables in `api_key`.

## 5. Prepare For RPO

For training, the judge must be initialized from one of:

- `rubrics` in YAML;
- `rubrics_file` in YAML or CLI;
- `seed_dataset` passed to `Judger`.

For long runs, prefer `rubrics_file`. It keeps training startup deterministic.

FLUX:

```bash
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
bash scripts/finetune/finetune_flux_rpo_8gpus.sh
```

Qwen-Image-Edit:

```bash
bash scripts/preprocess/preprocess_qwen_image_edit_rl_embeddings.sh
bash scripts/finetune/finetune_qwen_image_edit_rpo_8gpus.sh
```

## 6. Evaluate Position Bias

Create a swapped copy of your validation set where image order is reversed and `label_rank` is updated accordingly. Evaluate both files with the same saved rubric.

Good signs:

- forward and swapped accuracy are close;
- reasons reference image content rather than image position;
- failures are explainable and not always biased toward Image 1.

