#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)"
cd "${ROOT_DIR}"

COMMON_SCRIPT_DIR="VRSBench/benchmark/single_task/common/scripts"

SHARD_WORLD_SIZE="${SHARD_WORLD_SIZE:-2}"
SHARD_GPU_IDS="${SHARD_GPU_IDS:-0,1}"
SHARD_WEIGHTS="${SHARD_WEIGHTS:-1:1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
CUDA_RESERVE_FREE_GB="${CUDA_RESERVE_FREE_GB:-}"
DO_SAMPLE_FLAG="${DO_SAMPLE_FLAG:-}"
CAPTION_SAMPLES="${CAPTION_SAMPLES:-0}"
RUN_NAME="${RUN_NAME:-single_task_geochat_rsicd}"
GEOCHAT_MODEL_DIR="${GEOCHAT_MODEL_DIR:-GeoChat-Bench/model/geochat-7B}"
GEOCHAT_CODE_ROOT="${GEOCHAT_CODE_ROOT:-GeoChat-Bench/GeoChat}"
CONV_MODE="${CONV_MODE:-llava_v1}"

LOG_DIR="${LOG_DIR:-VRSBench/benchmark/single_task/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-VRSBench/benchmark/single_task/outputs/${RUN_NAME}}"
EVAL_DIR="${EVAL_DIR:-VRSBench/benchmark/single_task/eval/${RUN_NAME}}"

IFS=',' read -r -a GPU_ID_ARR <<< "${SHARD_GPU_IDS}"
if [[ "${SHARD_WORLD_SIZE}" -lt 1 ]]; then
  echo "[ERROR] SHARD_WORLD_SIZE must be >= 1" >&2
  exit 2
fi
if [[ "${#GPU_ID_ARR[@]}" -lt "${SHARD_WORLD_SIZE}" ]]; then
  echo "[ERROR] SHARD_GPU_IDS has fewer GPUs than SHARD_WORLD_SIZE" >&2
  exit 2
fi

rm -rf "${OUTPUT_DIR}" "${EVAL_DIR}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${EVAL_DIR}"

if [[ -n "${PY_BIN:-}" ]]; then
  PY_CMD=("${PY_BIN}")
else
  PY_CMD=(conda run --no-capture-output -n geochat python)
fi

if [[ -n "${EVAL_PY_BIN:-}" ]]; then
  read -r -a EVAL_PY_CMD <<< "${EVAL_PY_BIN}"
else
  EVAL_PY_CMD=("${PY_CMD[@]}")
fi

EXTRA_ARGS=()
if [[ -n "${CUDA_RESERVE_FREE_GB}" && "${CUDA_RESERVE_FREE_GB}" != "None" ]]; then
  EXTRA_ARGS+=(--cuda-reserve-free-gb "${CUDA_RESERVE_FREE_GB}")
fi

PIDS=()
SHARD_FILES=()
for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
  gpu_id="${GPU_ID_ARR[rank]}"
  shard_log="${LOG_DIR}/${RUN_NAME}.gpu${rank}.log"
  shard_file="${OUTPUT_DIR}/rsicd.gpu${rank}.jsonl"
  SHARD_FILES+=("${shard_file}")

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY_CMD[@]}" \
    "${COMMON_SCRIPT_DIR}/generate_single_task.py" \
    --preset geochat \
    --datasets rsicd \
    --output-dir "${OUTPUT_DIR}" \
    --caption-samples "${CAPTION_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --device-map "${DEVICE_MAP}" \
    --dtype "${DTYPE}" \
    "${EXTRA_ARGS[@]}" \
    --geochat-model-dir "${GEOCHAT_MODEL_DIR}" \
    --geochat-code-root "${GEOCHAT_CODE_ROOT}" \
    --conv-mode "${CONV_MODE}" \
    --shard-world-size "${SHARD_WORLD_SIZE}" \
    --shard-rank "${rank}" \
    --shard-weights "${SHARD_WEIGHTS}" \
    ${DO_SAMPLE_FLAG} > "${shard_log}" 2>&1 &
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

"${PY_CMD[@]}" "${COMMON_SCRIPT_DIR}/merge_jsonl_shards.py" \
  --inputs "${SHARD_FILES[@]}" \
  --output "${OUTPUT_DIR}/rsicd.jsonl" \
  --key sample_id \
  --delete-inputs

"${EVAL_PY_CMD[@]}" "${COMMON_SCRIPT_DIR}/evaluate_single_task.py" \
  --output-dir "${OUTPUT_DIR}" \
  --summary-out "${EVAL_DIR}/evaluation_summary.json"

echo "[OK] predictions: ${OUTPUT_DIR}/rsicd.jsonl"
echo "[OK] summary: ${EVAL_DIR}/evaluation_summary.json"
