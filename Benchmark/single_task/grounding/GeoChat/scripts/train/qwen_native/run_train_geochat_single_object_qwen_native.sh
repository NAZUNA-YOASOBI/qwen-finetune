#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ -n "${PY_BIN:-}" ]]; then
  ACCELERATE_CMD=("${PY_BIN}" -m accelerate.commands.launch --num_processes 2)
else
  ACCELERATE_CMD=(conda run --no-capture-output -n qwen3-dinov3 accelerate launch --num_processes 2)
fi

OUTPUT_DIR="${OUTPUT_DIR:-VRSBench/checkpoints/single_task/qwen_native/geochat_single_object_merger_lora_sampleavg_wd001}"

"${ACCELERATE_CMD[@]}" VRSBench/train_scripts/single_task/qwen_native/train_single_task_qwen_native.py \
  --dataset geochat_single_object \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --batch-size-ratio "${BATCH_SIZE_RATIO:-1:1}" \
  --grad-accum "${GRAD_ACCUM:-2}" \
  --target-global-batch "${TARGET_GLOBAL_BATCH:-32}" \
  --geochat-val-ratio "${GEOCHAT_VAL_RATIO:-0.1}" \
  --cuda-reserve-free-gb "${CUDA_RESERVE_FREE_GB:-3}"
