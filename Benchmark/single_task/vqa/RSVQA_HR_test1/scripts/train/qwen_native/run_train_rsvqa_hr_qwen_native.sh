#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
while [[ ! -d "${ROOT_DIR}/VRSBench" || ! -d "${ROOT_DIR}/fine-tune-qwen3-vl" ]]; do
  ROOT_DIR="$(dirname "${ROOT_DIR}")"
done
cd "${ROOT_DIR}"

OURS_ROOT="${ROOT_DIR}/fine-tune-qwen3-vl/Benchmark/single_task/vqa/RSVQA_HR_test1"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ -n "${PY_BIN:-}" ]]; then
  ACCELERATE_CMD=("${PY_BIN}" -m accelerate.commands.launch --num_processes 2)
else
  ACCELERATE_CMD=(conda run --no-capture-output -n qwen3-dinov3 accelerate launch --num_processes 2)
fi

OUTPUT_DIR="${OUTPUT_DIR:-${OURS_ROOT}/runtime/checkpoints/qwen_native/rsvqa_hr_merger_lora_sampleavg_wd001}"

"${ACCELERATE_CMD[@]}" "${OURS_ROOT}/scripts/train/qwen_native/train_single_task_qwen_native.py" \
  --dataset rsvqa_hr \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --batch-size-ratio "${BATCH_SIZE_RATIO:-1:1}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --target-global-batch "${TARGET_GLOBAL_BATCH:-32}" \
  --smart-resize-min-pixels "${SMART_RESIZE_MIN_PIXELS:-50176}" \
  --smart-resize-max-pixels "${SMART_RESIZE_MAX_PIXELS:-262144}" \
  --rsvqa-train-ratio "${RSVQA_TRAIN_RATIO:-0.2}" \
  --rsvqa-val-ratio "${RSVQA_VAL_RATIO:-0.2}" \
  --cuda-reserve-free-gb "${CUDA_RESERVE_FREE_GB:-3}"
