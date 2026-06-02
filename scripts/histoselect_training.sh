#!/usr/bin/env bash
set -euo pipefail

# ===============================================================
# HistoSelect main-method training.
#
# Config: configs/histoselect/stage_2_selector.py
#   - Switch dataset by editing DATASET = 'wsi-llava' or 'slidechat'.
#   - Eff batch = 8 (per-device B=1, accum=2, 4 GPU).
#
# Override defaults via env vars:
#   CONFIG, DEEPSPEED_CONFIG, WORK_DIR, NPROC_PER_NODE, PORT, CUDA_VISIBLE_DEVICES
# ===============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${CONFIG:-xtuner/configs/histoselect/stage_2_selector.py}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-xtuner/configs/deepspeed/deepspeed_zero2.json}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/work_dirs/histoselect_run}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PORT="${PORT:-29500}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 \
  --master_port="${PORT}" \
  xtuner/tools/train.py \
  "${CONFIG}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --launcher pytorch
