#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ -n "${PY_BIN:-}" ]]; then
  PYTHON_CMD=("${PY_BIN}")
  ACCELERATE_CMD=("${PY_BIN}" -m accelerate.commands.launch --num_processes 2)
else
  PYTHON_CMD=(conda run --no-capture-output -n qwen3-dinov3 python)
  ACCELERATE_CMD=(conda run --no-capture-output -n qwen3-dinov3 accelerate launch --num_processes 2)
fi

DINO_RSICD_CKPT="VRSBench/checkpoints/single_task/dinov3/rsicd_merger_lora_sampleavg_wd001"
NATIVE_RSICD_CKPT="VRSBench/checkpoints/single_task/qwen_native/rsicd_merger_lora_sampleavg_wd001"
DINO_RSVQA_CKPT="VRSBench/checkpoints/single_task/dinov3/rsvqa_hr_merger_lora_sampleavg_wd001"
NATIVE_RSVQA_CKPT="VRSBench/checkpoints/single_task/qwen_native/rsvqa_hr_merger_lora_sampleavg_wd001"

DINO_RSICD_OUTPUT="VRSBench/benchmark/single_task/outputs/single_task_dinov3_rsicd_epoch5"
NATIVE_RSICD_OUTPUT="VRSBench/benchmark/single_task/outputs/single_task_qwen_native_rsicd_epoch5"
DINO_RSVQA_OUTPUT="VRSBench/benchmark/single_task/outputs/single_task_dinov3_rsvqa_hr_epoch5"
NATIVE_RSVQA_OUTPUT="VRSBench/benchmark/single_task/outputs/single_task_qwen_native_rsvqa_hr_epoch5"
COMMON_SCRIPT_DIR="VRSBench/benchmark/single_task/common/scripts"

"${ACCELERATE_CMD[@]}" VRSBench/train_scripts/single_task/dinov3/train_single_task_dinov3.py \
  --dataset rsicd \
  --output-dir "${DINO_RSICD_CKPT}" \
  --epochs 5 \
  --batch-size 8 \
  --batch-size-ratio 1:1 \
  --grad-accum 2 \
  --target-global-batch 32 \
  --cuda-reserve-free-gb 3

"${PYTHON_CMD[@]}" "${COMMON_SCRIPT_DIR}/generate_single_task.py" \
  --model-family dinov3 \
  --merger-ckpt "${DINO_RSICD_CKPT}/epoch5/merger.safetensors" \
  --preset-name single_task_dinov3_rsicd_epoch5 \
  --datasets rsicd \
  --output-dir "${DINO_RSICD_OUTPUT}" \
  --batch-size 512 \
  --device-map auto

"${PYTHON_CMD[@]}" "${COMMON_SCRIPT_DIR}/evaluate_single_task.py" \
  --output-dir "${DINO_RSICD_OUTPUT}"

"${ACCELERATE_CMD[@]}" VRSBench/train_scripts/single_task/qwen_native/train_single_task_qwen_native.py \
  --dataset rsicd \
  --output-dir "${NATIVE_RSICD_CKPT}" \
  --epochs 5 \
  --batch-size 8 \
  --batch-size-ratio 1:1 \
  --grad-accum 2 \
  --target-global-batch 32 \
  --cuda-reserve-free-gb 3

"${PYTHON_CMD[@]}" "${COMMON_SCRIPT_DIR}/generate_single_task.py" \
  --model-family qwen_native \
  --merger-ckpt "${NATIVE_RSICD_CKPT}/epoch5/merger.safetensors" \
  --preset-name single_task_qwen_native_rsicd_epoch5 \
  --datasets rsicd \
  --output-dir "${NATIVE_RSICD_OUTPUT}" \
  --batch-size 512 \
  --device-map auto

"${PYTHON_CMD[@]}" "${COMMON_SCRIPT_DIR}/evaluate_single_task.py" \
  --output-dir "${NATIVE_RSICD_OUTPUT}"

"${ACCELERATE_CMD[@]}" VRSBench/train_scripts/single_task/dinov3/train_single_task_dinov3.py \
  --dataset rsvqa_hr \
  --output-dir "${DINO_RSVQA_CKPT}" \
  --epochs 5 \
  --batch-size 8 \
  --batch-size-ratio 1:1 \
  --grad-accum 2 \
  --target-global-batch 32 \
  --cuda-reserve-free-gb 3

"${PYTHON_CMD[@]}" "${COMMON_SCRIPT_DIR}/generate_single_task.py" \
  --model-family dinov3 \
  --merger-ckpt "${DINO_RSVQA_CKPT}/epoch5/merger.safetensors" \
  --preset-name single_task_dinov3_rsvqa_hr_epoch5 \
  --datasets rsvqa_hr_test1,rsvqa_hr_test2 \
  --output-dir "${DINO_RSVQA_OUTPUT}" \
  --batch-size 512 \
  --device-map auto

"${PYTHON_CMD[@]}" "${COMMON_SCRIPT_DIR}/evaluate_single_task.py" \
  --output-dir "${DINO_RSVQA_OUTPUT}"

"${ACCELERATE_CMD[@]}" VRSBench/train_scripts/single_task/qwen_native/train_single_task_qwen_native.py \
  --dataset rsvqa_hr \
  --output-dir "${NATIVE_RSVQA_CKPT}" \
  --epochs 5 \
  --batch-size 8 \
  --batch-size-ratio 1:1 \
  --grad-accum 2 \
  --target-global-batch 32 \
  --cuda-reserve-free-gb 3

"${PYTHON_CMD[@]}" "${COMMON_SCRIPT_DIR}/generate_single_task.py" \
  --model-family qwen_native \
  --merger-ckpt "${NATIVE_RSVQA_CKPT}/epoch5/merger.safetensors" \
  --preset-name single_task_qwen_native_rsvqa_hr_epoch5 \
  --datasets rsvqa_hr_test1,rsvqa_hr_test2 \
  --output-dir "${NATIVE_RSVQA_OUTPUT}" \
  --batch-size 512 \
  --device-map auto

"${PYTHON_CMD[@]}" "${COMMON_SCRIPT_DIR}/evaluate_single_task.py" \
  --output-dir "${NATIVE_RSVQA_OUTPUT}"
