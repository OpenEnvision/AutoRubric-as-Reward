# VLM Selection

Auto-Rubric uses an OpenAI-compatible chat API. The judge can be a local VLM served through vLLM or a hosted model exposed through an OpenAI-compatible endpoint. The important requirement is not the provider name; it is whether the model can reliably inspect multiple images and return structured JSON.

## Selection Checklist

| Requirement | Why it matters |
| --- | --- |
| Multi-image understanding | Pairwise and image-edit tasks send two or more images in one request. |
| Stable JSON output | The pipeline expects `rubrics`, `rank`, or `score` fields. |
| Low position bias | Pairwise judging should not always prefer Image 1 or Image 2. |
| Enough context length | Categorization may include many generated rubrics. |
| Reasonable latency | RPO calls the judge during rollout, so slow endpoints become the bottleneck. |
| Cost control | Rubric generation and validation can multiply API calls by `seed_items * max_epochs`. |

## Practical Model Choices

| Option | Best for | Tradeoff |
| --- | --- | --- |
| Local Qwen3-VL | Private data, reproducible local runs, no per-call API cost. | Requires GPU serving capacity and vLLM setup. |
| GPT-compatible endpoint | Strong general judging, simpler hosted setup. | Cost and rate limits matter; model availability may change. |
| Gemini-compatible endpoint | Strong visual reasoning through OpenAI-compatible API. | Endpoint details and JSON behavior should be tested first. |
| Custom OpenAI-compatible VLM | Internal deployments or lab clusters. | You must verify multi-image payload and JSON compatibility. |

## Local Qwen3-VL

Start a local OpenAI-compatible server:

```bash
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct TP_SIZE=1 PORT=8000 \
  bash rubric_pipeline/vllm_serve.sh
```

Then configure:

```bash
export OPENAI_API_KEY=EMPTY
```

Example YAML:

```yaml
model_name: "Qwen/Qwen3-VL-8B-Instruct"
base_url: "http://localhost:8000/v1"
api_key: "EMPTY"
```

## Hosted OpenAI-Compatible Endpoint

Example GPT-compatible config:

```yaml
model_name: "gpt-5"
base_url: "https://api.openai.com/v1"
api_key: "${OPENAI_API_KEY}"
```

Example Gemini-compatible config:

```yaml
model_name: "gemini-3.1-pro"
base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key: "${GEMINI_API_KEY}"
```

## Before A Long Run

Run a small smoke test before launching RPO:

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml \
  --seed_dataset examples/seed_t2i_pairwise.json \
  --test_dataset examples/test_t2i_pairwise.json \
  --base_url http://localhost:8000/v1 \
  --concurrency_limit 2
```

Inspect three things:

- The output contains generated rubrics.
- The evaluation returns rank arrays of the expected length.
- Accuracy is not obviously random on the tiny test set.

## Suggested Settings

| Situation | Setting |
| --- | --- |
| Endpoint rate limits | Lower `--concurrency_limit`. |
| Weak JSON compliance | Use a stronger judge or reduce prompt complexity. |
| Long categorization prompt | Lower `categories_number` or reduce seed size. |
| High position bias | Test swapped image order and consider a stronger VLM. |
| RPO too slow | Generate and save rubrics once, then reuse `rubrics_file`. |

