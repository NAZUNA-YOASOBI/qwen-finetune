#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_sampleavg_wd001_run_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="checkpoints/vrsbench_joint/${RUN_TAG}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}"

echo "[RUN_DIR] ${RUN_DIR}"

conda run --no-capture-output -n qwen3-dinov3 \
  accelerate launch --num_processes 2 \
  train_scripts/merger_lora_dinov3_sampleavg_wd001/train_vrsbench_multitask_sampleavg_wd001.py \
  --qwen-model-dir models/Qwen3-VL-8B-Instruct \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --dataset-root datasets/VRSBench \
  --train-json datasets/VRSBench/VRSBench_train.json \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 16777216 \
  --output-dir "${RUN_DIR}" \
  --epochs 10 \
  --batch-size-per-rank 8,8 \
  --grad-accum 2 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --mixed-precision bf16 \
  --gradient-checkpointing \
  --lora-r 64 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --seed 42 \
  --num-workers 2 \
  "$@" \
  2>&1 | tee "${RUN_DIR}/train.log"
