#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUNDING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${GROUNDING_DIR}/../../../.." && pwd)"
cd "${PROJECT_ROOT}"

GROUNDING_SCRIPT="VRSBench/benchmark/single_task/grounding/scripts/eval_grounding_single_object.py"
MERGE_SCRIPT="VRSBench/benchmark/single_task/common/scripts/merge_jsonl_shards.py"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  echo "CHECKPOINT_DIR is required. Example: VRSBench/checkpoints/single_task/qwen_native/.../epoch5" >&2
  exit 1
fi

SHARD_WORLD_SIZE="${SHARD_WORLD_SIZE:-2}"
SHARD_GPU_IDS="${SHARD_GPU_IDS:-0,1}"
SHARD_WEIGHTS="${SHARD_WEIGHTS:-1:1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
CUDA_RESERVE_FREE_GB="${CUDA_RESERVE_FREE_GB:-}"
RUN_NAME="${RUN_NAME:-qwen_native_geochat_single_object_$(date +%Y%m%d_%H%M%S)}"
DATA_JSON="${DATA_JSON:-fine-tune-qwen3-vl/Benchmark/single_task/grounding/GeoChat/data/test.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-VRSBench/benchmark/single_task/grounding/outputs}"
EVAL_ROOT="${EVAL_ROOT:-VRSBench/benchmark/single_task/grounding/eval}"
LOG_ROOT="${LOG_ROOT:-VRSBench/benchmark/single_task/grounding/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
EVAL_DIR="${EVAL_DIR:-${EVAL_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${RUN_NAME}}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-VRSBench/models/Qwen3-VL-8B-Instruct}"
MERGE_LORA="${MERGE_LORA:-0}"

IFS=',' read -r -a GPU_ID_ARR <<< "${SHARD_GPU_IDS}"
if [[ "${SHARD_WORLD_SIZE}" -lt 1 ]]; then
  echo "[ERROR] SHARD_WORLD_SIZE must be >= 1" >&2
  exit 2
fi
if [[ "${#GPU_ID_ARR[@]}" -lt "${SHARD_WORLD_SIZE}" ]]; then
  echo "[ERROR] SHARD_GPU_IDS has fewer GPUs than SHARD_WORLD_SIZE" >&2
  exit 2
fi

rm -rf "${OUTPUT_DIR}" "${EVAL_DIR}" "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}" "${EVAL_DIR}" "${LOG_DIR}"

if [[ -n "${PY_BIN:-}" ]]; then
  PY_CMD=("${PY_BIN}")
else
  PY_CMD=(conda run --no-capture-output -n qwen3-dinov3 python)
fi

ARGS_COMMON=(
  "${GROUNDING_SCRIPT}"
  --family qwen_native
  --data-json "${DATA_JSON}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --qwen-model-dir "${QWEN_MODEL_DIR}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)
if [[ -n "${CUDA_RESERVE_FREE_GB}" && "${CUDA_RESERVE_FREE_GB}" != "None" ]]; then
  ARGS_COMMON+=(--cuda-reserve-free-gb "${CUDA_RESERVE_FREE_GB}")
fi
if [[ "${MERGE_LORA}" == "1" || "${MERGE_LORA}" == "true" ]]; then
  ARGS_COMMON+=(--merge-lora)
fi

PRED_FILE="${OUTPUT_DIR}/predictions.jsonl"
EVAL_FILE="${EVAL_DIR}/evaluation_summary.json"
PIDS=()
SHARD_FILES=()

for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
  gpu_id="${GPU_ID_ARR[rank]}"
  shard_file="${OUTPUT_DIR}/predictions.gpu${rank}.jsonl"
  shard_log="${LOG_DIR}/generate.gpu${rank}.log"
  SHARD_FILES+=("${shard_file}")

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY_CMD[@]}" \
    "${ARGS_COMMON[@]}" \
    --predictions-out "${shard_file}" \
    --generate-only \
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

"${PY_CMD[@]}" "${MERGE_SCRIPT}" \
  --inputs "${SHARD_FILES[@]}" \
  --output "${PRED_FILE}" \
  --key question_id \
  --delete-inputs

"${PY_CMD[@]}" \
  "${GROUNDING_SCRIPT}" \
  --family qwen_native \
  --data-json "${DATA_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --predictions-jsonl "${PRED_FILE}" \
  --summary-out "${EVAL_FILE}" \
  --batch-size "${BATCH_SIZE}" \
  --shard-world-size "${SHARD_WORLD_SIZE}" \
  --shard-weights "${SHARD_WEIGHTS}" > "${LOG_DIR}/summary.log" 2>&1

echo "[OK] predictions: ${PRED_FILE}"
echo "[OK] summary: ${EVAL_FILE}"
