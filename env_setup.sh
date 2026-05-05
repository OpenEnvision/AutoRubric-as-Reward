#!/usr/bin/env bash
set -euo pipefail

CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

python -m pip install --upgrade pip
python -m pip install torch==2.5.0 torchvision --index-url "${CUDA_INDEX_URL}"
python -m pip install packaging ninja
python -m pip install flash-attn==2.7.0.post2 --no-build-isolation
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Environment ready. Configure OPENAI_API_KEY or start rubric_pipeline/vllm_serve.sh before using Auto-Rubric rewards."
