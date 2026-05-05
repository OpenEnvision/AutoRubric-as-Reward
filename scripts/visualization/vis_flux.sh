#!/usr/bin/env bash
set -euo pipefail

GPU_NUM="${GPU_NUM:-8}"
MASTER_PORT="${MASTER_PORT:-19022}"

torchrun --nproc_per_node="${GPU_NUM}" --master_port "${MASTER_PORT}" \
  scripts/visualization/vis_flux.py
