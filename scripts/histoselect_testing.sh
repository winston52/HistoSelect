#!/usr/bin/env bash
set -euo pipefail

# ===============================================================
# HistoSelect inference on a held-out test set.
#
# Runs tools/test_histoselect.py and dumps per-sample predictions
# to JSON.
#
# Required env vars:
#   CKPT       — trained checkpoint (.pth)
#   TEST_JSON  — evaluation data (SlideChat-style VQA JSON)
#   QEMB       — precomputed CONCH question embeddings (.pt)
#
# Optional:
#   CONFIG, OUT_DIR, CUDA_VISIBLE_DEVICES
# ===============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${CONFIG:-configs/histoselect/stage_2_selector.py}"
CKPT="${CKPT:?must set CKPT to a trained iter_*.pth path}"
TEST_JSON="${TEST_JSON:?must set TEST_JSON to evaluation data path}"
QEMB="${QEMB:?must set QEMB to question_embeddings.pt path}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/eval}"
mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${REPO_ROOT}/xtuner"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python tools/test_histoselect.py \
  "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --test_json_path "${TEST_JSON}" \
  --question_embedding_path "${QEMB}" \
  --output_log_json "${OUT_DIR}/predictions.json" \
  --local_rank 0
