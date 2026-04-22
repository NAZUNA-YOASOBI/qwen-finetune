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
BASE_TAG="${BASE_TAG:-single_task_qwen_native_rsicd}"
EPOCHS="${EPOCHS:-1 2 3 4 5}"

CKPT_ROOT="${CKPT_ROOT:-VRSBench/checkpoints/single_task/qwen_native/rsicd_merger_lora_sampleavg_wd001}"
LOG_DIR="${LOG_DIR:-VRSBench/benchmark/single_task/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-VRSBench/benchmark/single_task/outputs}"
EVAL_ROOT="${EVAL_ROOT:-VRSBench/benchmark/single_task/eval}"

IFS=',' read -r -a GPU_ID_ARR <<< "${SHARD_GPU_IDS}"
if [[ "${SHARD_WORLD_SIZE}" -lt 1 ]]; then
  echo "[ERROR] SHARD_WORLD_SIZE must be >= 1" >&2
  exit 2
fi
if [[ "${#GPU_ID_ARR[@]}" -lt "${SHARD_WORLD_SIZE}" ]]; then
  echo "[ERROR] SHARD_GPU_IDS has fewer GPUs than SHARD_WORLD_SIZE" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${EVAL_ROOT}"

if [[ -n "${PY_BIN:-}" ]]; then
  PY_CMD=("${PY_BIN}")
else
  PY_CMD=(conda run --no-capture-output -n qwen3-dinov3 python)
fi

EXTRA_ARGS=()
if [[ -n "${CUDA_RESERVE_FREE_GB}" && "${CUDA_RESERVE_FREE_GB}" != "None" ]]; then
  EXTRA_ARGS+=(--cuda-reserve-free-gb "${CUDA_RESERVE_FREE_GB}")
fi

run_one_epoch() {
  local epoch="$1"
  local tag="${BASE_TAG}_epoch${epoch}"
  local merger_ckpt="${CKPT_ROOT}/epoch${epoch}/merger.safetensors"
  local output_dir="${OUTPUT_ROOT}/${tag}"
  local eval_dir="${EVAL_ROOT}/${tag}"
  local pred_file="${output_dir}/rsicd.jsonl"
  local eval_file="${eval_dir}/evaluation_summary.json"
  local pids=()
  local shard_files=()

  if [[ ! -f "${merger_ckpt}" ]]; then
    echo "[ERROR] Missing checkpoint: ${merger_ckpt}" >&2
    exit 2
  fi

  rm -rf "${output_dir}" "${eval_dir}"
  mkdir -p "${output_dir}" "${eval_dir}"

  local rank
  for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
    local gpu_id="${GPU_ID_ARR[rank]}"
    local shard_log="${LOG_DIR}/${tag}.gpu${rank}.log"
    local shard_file="${output_dir}/rsicd.gpu${rank}.jsonl"
    shard_files+=("${shard_file}")

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY_CMD[@]}" \
      "${COMMON_SCRIPT_DIR}/generate_single_task.py" \
      --model-family qwen_native \
      --merger-ckpt "${merger_ckpt}" \
      --preset-name "${tag}" \
      --datasets rsicd \
      --output-dir "${output_dir}" \
      --caption-samples "${CAPTION_SAMPLES}" \
      --batch-size "${BATCH_SIZE}" \
      --device-map "${DEVICE_MAP}" \
      --dtype "${DTYPE}" \
      "${EXTRA_ARGS[@]}" \
      --shard-world-size "${SHARD_WORLD_SIZE}" \
      --shard-rank "${rank}" \
      --shard-weights "${SHARD_WEIGHTS}" \
      ${DO_SAMPLE_FLAG} > "${shard_log}" 2>&1 &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "[FAIL] epoch${epoch} generation status=${status}" >&2
    return "${status}"
  fi

  "${PY_CMD[@]}" "${COMMON_SCRIPT_DIR}/merge_jsonl_shards.py" \
    --inputs "${shard_files[@]}" \
    --output "${pred_file}" \
    --key sample_id \
    --delete-inputs

  "${PY_CMD[@]}" "${COMMON_SCRIPT_DIR}/evaluate_single_task.py" \
    --output-dir "${output_dir}" \
    --summary-out "${eval_file}"
}

for epoch in ${EPOCHS}; do
  run_one_epoch "${epoch}"
done
