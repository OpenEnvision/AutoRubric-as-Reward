<h1 align="center">Auto-Rubric as Reward</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-coming_soon-B31B1B?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/OpenEnvisionLab/Auto-Rubric-as-Reward"><img src="https://img.shields.io/badge/HuggingFace-ARR-FFD21E?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
  <a href="https://openenvision.github.io/AutoRubric-as-Reward/"><img src="https://img.shields.io/badge/Project-Website-2F6FED?logo=googlechrome&logoColor=white" alt="Project Website"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  The official implementation for <strong>Auto-Rubric as Reward: From Implicit Preference to Explicit Generative Criteria</strong>
</p>

<p align="center">
  <a href="#what-this-repo-does">Overview</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#auto-rubric-documentation">Auto-Rubric Docs</a> |
  <a href="#training">Training</a> |
  <a href="#acknowledgements">Acknowledgements</a>
</p>

## What This Repo Does

Auto-Rubric provides a compact implementation of Auto-Rubric as Reward for visual generation. It turns a small set of labeled visual preference examples into explicit, inspectable rubric text, then uses a frozen VLM judge conditioned on those rubrics to produce pairwise rewards for RPO.

```text
labeled visual pairs
  -> auto-generate rubrics
  -> verify and revise criteria
  -> structure/reuse rubric text
  -> VLM judge returns ranks
  -> RPO receives pairwise rewards
```

This release focuses on:


| Area          | Included                                                                        |
| ------------- | ------------------------------------------------------------------------------- |
| Auto-Rubric   | Generation, verification, revision, categorization, grading, reward conversion. |
| Text-to-image | FLUX.1-dev LoRA RPO with pairwise ARR rewards.                                  |
| Image editing | Qwen-Image-Edit LoRA RPO with source-image-aware pairwise ARR rewards.          |
| VLM judging   | OpenAI-compatible local or hosted vision endpoints.                             |


Large checkpoints, processed embeddings, and training outputs are intentionally not committed.

## Key Features

- **Explicit reward criteria**: The "reward model" is readable rubric text rather than a hidden scalar model.
- **Verifiable generation loop**: Candidate rubrics are checked against labeled examples and revised when they fail.
- **Pairwise visual rewards**: Rank 1 receives `1.0`; rank 2 receives `-0.1` for RPO.
- **T2I and edit support**: Prompt-only FLUX and source-image-aware Qwen-Image-Edit paths are both wired.
- **Reusable rubric files**: Generate rubrics once, inspect them, and reuse the same file for deterministic training launches.
- **OpenAI-compatible VLMs**: Use local Qwen3-VL through vLLM or hosted GPT/Gemini-compatible endpoints.

## Repository Map


| Path                               | Purpose                                                                             |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| `judger.py`                        | CLI and Python entry point for rubric generation, evaluation, and reward tensors.   |
| `rubric_pipeline/`                 | Auto-Rubric prompts, VLM graders, model client, categorization, and utilities.      |
| `fastvideo/train_rpo_flux.py`      | FLUX RPO training with ARR rewards.                                                 |
| `fastvideo/train_rpo_qwen_edit.py` | Qwen-Image-Edit RPO training with ARR rewards.                                      |
| `scripts/preprocess/`              | Embedding preprocessing for FLUX and Qwen-Image-Edit.                               |
| `scripts/finetune/`                | 8-GPU launcher examples with paper-aligned RPO defaults.                            |
| `docs/auto_rubric/`                | Detailed Auto-Rubric guide: VLM choice, rubric design, reuse, workflows, debugging. |


## Quick Start

Create the environment:

```bash
cd /path/to/AutoRubric-as-Reward
conda create -n autorubric-as-reward python=3.10 -y
conda activate autorubric-as-reward
bash env_setup.sh
```

If you already installed a different CUDA/PyTorch stack, install the repo dependencies directly:

```bash
pip install -r requirements.txt
pip install -e .
```

Create the expected data folder:

```bash
mkdir -p data rubric_pipeline/rubrics
```

Download base models as needed:


| Model           | Local path            | Link                                                                                                       |
| --------------- | --------------------- | ---------------------------------------------------------------------------------------------------------- |
| FLUX.1-dev      | `data/flux`           | [https://huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| Qwen-Image-Edit | `data/qwenimage_edit` | [https://huggingface.co/Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)                 |
| Qwen3-VL judge  | local or HF cache     | [https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)       |


## Start A VLM Judge

Auto-Rubric talks to an OpenAI-compatible vision API.

Local Qwen3-VL:

```bash
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct TP_SIZE=1 PORT=8000 \
  bash rubric_pipeline/vllm_serve.sh

export OPENAI_API_KEY=EMPTY
```

Hosted endpoint examples:

```yaml
model_name: "gpt-5"
base_url: "https://api.openai.com/v1"
api_key: "${OPENAI_API_KEY}"
```

```yaml
model_name: "gemini-3.1-pro"
base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key: "${GEMINI_API_KEY}"
```

More guidance: [VLM Selection](docs/auto_rubric/vlm_selection.md).

## Generate And Test Rubrics

Text-to-image:

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2i.yaml \
  --seed_dataset examples/seed_t2i_pairwise.json \
  --test_dataset examples/test_t2i_pairwise.json \
  --base_url http://localhost:8000/v1 \
  --concurrency_limit 4
```

Image editing:

```bash
python judger.py \
  --config_path rubric_pipeline/config/qwen3vl_8B_instruct_edit.yaml \
  --seed_dataset examples/seed_edit_pairwise.json \
  --test_dataset examples/test_edit_pairwise.json \
  --base_url http://localhost:8000/v1 \
  --concurrency_limit 4
```

For long runs, save the generated rubric text to `rubric_pipeline/rubrics/*.txt` and load it through `rubrics_file`. See [Rubric Reuse](docs/auto_rubric/rubric_reuse.md).

## Training

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

The launchers use:


| Task            | LR     | Steps | Clip  | KL     | LoRA    |
| --------------- | ------ | ----- | ----- | ------ | ------- |
| FLUX T2I        | `5e-5` | `8`   | `0.2` | `0.01` | rank 16 |
| Qwen-Image-Edit | `1e-5` | `10`  | `0.2` | `0.02` | rank 32 |


Pairwise RPO expects `--use_arr`, `--num_generations 2`, and `--use_group`.

## Auto-Rubric Documentation


| Guide                                                  | Covers                                                             |
| ------------------------------------------------------ | ------------------------------------------------------------------ |
| [Overview](docs/auto_rubric/overview.md)               | Method summary and paper-to-code map.                              |
| [VLM Selection](docs/auto_rubric/vlm_selection.md)     | Local vs hosted judges, JSON reliability, latency, cost.           |
| [Rubric Design](docs/auto_rubric/rubric_design.md)     | Seed pair selection, task descriptions, good/bad rubric patterns.  |
| [Rubric Reuse](docs/auto_rubric/rubric_reuse.md)       | Saved rubric files, versioning, validation before training.        |
| [Workflows](docs/auto_rubric/workflows.md)             | End-to-end commands for generation, testing, saving, and training. |
| [Troubleshooting](docs/auto_rubric/troubleshooting.md) | Common failures and fixes.                                         |
| [Data Formats](docs/data_formats.md)                   | JSON/JSONL layouts accepted by the vision utilities.               |
| [Training Guide](docs/training.md)                     | Preprocess and RPO launch details.                                 |


## Acknowledgements

This code builds on and is inspired by:

- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [diffusers](https://github.com/huggingface/diffusers)
- [OpenJudge](https://github.com/agentscope-ai/OpenJudge)

## Citation

```bibtex
@misc{tian2026autorubricrewardimplicitpreferences,
      title={Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria}, 
      author={Juanxi Tian and Fengyuan Liu and Jiaming Han and Yilei Jiang and Yongliang Wu and Yesheng Liu and Haodong Li and Furong Xu and Wanhua Li},
      year={2026},
      eprint={2605.08354},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.08354}, 
}
```
