#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
while [[ ! -d "${ROOT_DIR}/VRSBench" || ! -d "${ROOT_DIR}/fine-tune-qwen3-vl" ]]; do
  ROOT_DIR="$(dirname "${ROOT_DIR}")"
done
cd "${ROOT_DIR}"

OURS_ROOT="${ROOT_DIR}/fine-tune-qwen3-vl/Benchmark/single_task/vqa/RSVQA_HR_test1"
COMMON_SCRIPT_DIR="${ROOT_DIR}/VRSBench/benchmark/single_task/common/scripts"
CORE_SCRIPT_DIR="${OURS_ROOT}/scripts/eval/core"
JUDGE_SCRIPT="${CORE_SCRIPT_DIR}/judge_vqa_predictions.py"
MERGE_SCRIPT="${CORE_SCRIPT_DIR}/merge_jsonl_shards.py"
RESULT_DIR="${RESULT_DIR:-${OURS_ROOT}/results/geoground}"

SHARD_WORLD_SIZE="${SHARD_WORLD_SIZE:-2}"
SHARD_GPU_IDS="${SHARD_GPU_IDS:-0,1}"
SHARD_WEIGHTS="${SHARD_WEIGHTS:-1:1}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
CUDA_RESERVE_FREE_GB="${CUDA_RESERVE_FREE_GB:-}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DATASETS="${DATASETS:-rsvqa_hr_test1_10pct}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULT_DIR}}"
EVAL_DIR="${EVAL_DIR:-${RESULT_DIR}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/VRSBench/benchmark/single_task/logs}"
RUN_NAME="${RUN_NAME:-geoground_rsvqa_hr_test1}"
GEOGROUND_MODEL_DIR="${GEOGROUND_MODEL_DIR:-VRSBench/models/GeoGround/llava-v1.5-7b-task-lora-geoground}"
GEOGROUND_MODEL_BASE="${GEOGROUND_MODEL_BASE:-}"
LLAVA_CODE_ROOT="${LLAVA_CODE_ROOT:-VRSBench/models/GeoGround/LLaVA}"
CONV_MODE="${CONV_MODE:-llava_v1}"

IFS=',' read -r -a GPU_ID_ARR <<< "${SHARD_GPU_IDS}"
IFS=',' read -r -a DATASET_ARR <<< "${DATASETS}"
if [[ "${SHARD_WORLD_SIZE}" -lt 1 ]]; then
  echo "[ERROR] SHARD_WORLD_SIZE must be >= 1" >&2
  exit 2
fi
if [[ "${#GPU_ID_ARR[@]}" -lt "${SHARD_WORLD_SIZE}" ]]; then
  echo "[ERROR] SHARD_GPU_IDS has fewer GPUs than SHARD_WORLD_SIZE" >&2
  exit 2
fi

rm -rf "${RESULT_DIR}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${EVAL_DIR}"

if [[ -n "${PY_BIN:-}" ]]; then
  PY_CMD=("${PY_BIN}")
else
  PY_CMD=(conda run --no-capture-output -n geoground python)
fi

EXTRA_ARGS=()
if [[ -n "${CUDA_RESERVE_FREE_GB}" && "${CUDA_RESERVE_FREE_GB}" != "None" ]]; then
  EXTRA_ARGS+=(--cuda-reserve-free-gb "${CUDA_RESERVE_FREE_GB}")
fi

PIDS=()
for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
  gpu_id="${GPU_ID_ARR[rank]}"
  shard_log="${LOG_DIR}/${RUN_NAME}.gpu${rank}.log"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY_CMD[@]}" \
    "${COMMON_SCRIPT_DIR}/generate_single_task.py" \
    --preset geoground \
    --datasets "${DATASETS}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --device-map "${DEVICE_MAP}" \
    "${EXTRA_ARGS[@]}" \
    --geoground-model-dir "${GEOGROUND_MODEL_DIR}" \
    --geoground-model-base "${GEOGROUND_MODEL_BASE}" \
    --llava-code-root "${LLAVA_CODE_ROOT}" \
    --conv-mode "${CONV_MODE}" \
    --shard-world-size "${SHARD_WORLD_SIZE}" \
    --shard-rank "${rank}" \
    --shard-weights "${SHARD_WEIGHTS}" > "${shard_log}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -ne 0 ]]; then
  echo "[FAIL] generation status=${status}" >&2
  exit "${status}"
fi

for dataset_name in "${DATASET_ARR[@]}"; do
  shard_files=()
  for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
    shard_files+=("${OUTPUT_DIR}/${dataset_name}.gpu${rank}.jsonl")
  done
  "${PY_CMD[@]}" "${MERGE_SCRIPT}" \
    --inputs "${shard_files[@]}" \
    --output "${OUTPUT_DIR}/${dataset_name}.jsonl" \
    --key question_id \
    --delete-inputs
done

"${PY_CMD[@]}" "${JUDGE_SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --eval-dir "${EVAL_DIR}" \
  --datasets "${DATASETS}"

echo "[OK] output_dir: ${OUTPUT_DIR}"
echo "[OK] eval_dir: ${EVAL_DIR}"
