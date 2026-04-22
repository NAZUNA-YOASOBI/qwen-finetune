#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PY="${PY:-/opt/yanzixi/home/yzx/.conda/envs/qwen3-dinov3/bin/python}"
DTYPE="${DTYPE:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-256}"
FIX_BATCH_SIZE="${FIX_BATCH_SIZE:-256}"
GPU_IDS="${GPU_IDS:-0,1}"
SHARD_WEIGHTS="${SHARD_WEIGHTS:-1:1}"

BASELINE_CAPTION_PROMPT="Describe the image in detail in 2 to 4 sentences."
QWEN_MODEL_DIR="models/Qwen3-VL-8B-Instruct"
DINOV3_DIR="models/dinov3/dinov3-vitl16-pretrain-sat493m"
DINO_CKPT_DIR="checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_sampleavg_wd001_run_20260308_025747/epoch10"
SVA_CKPT_DIR="checkpoints/vrsbench_joint/merger_lora_8b_sva_deepstack_ca_micro8_8_ga2_effective32_taskseq_sampleavg_wd001_run_20260313_083153/epoch10"

BASELINE_OUT="benchmark/vrsbench/outputs/baseline_noftstyle/18_qwen8b_baseline_noftstyle_retest_20260315"
BASELINE_EVAL="benchmark/vrsbench/eval/baseline_noftstyle/18_qwen8b_baseline_noftstyle_retest_20260315"
DINO_OUT="benchmark/vrsbench/outputs/merger_lora_smartbucket512/13_qwen8b_smartbucket_sampleavg_wd001_epoch10_retest_20260315"
DINO_EVAL="benchmark/vrsbench/eval/merger_lora_smartbucket512/13_qwen8b_smartbucket_sampleavg_wd001_epoch10_retest_20260315"
SVA_OUT="benchmark/vrsbench/outputs/sva_dual/02_qwen8b_sva_deepstack_ca_epoch10_sampleavg_retest_20260315"
SVA_EVAL="benchmark/vrsbench/eval/sva_deepstack_ca/02_qwen8b_sva_deepstack_ca_epoch10_sampleavg_retest_20260315"

LOG_ROOT="benchmark/vrsbench/eval/retest_three_models_20260315_logs"
MASTER_LOG="${LOG_ROOT}/master.log"
mkdir -p "${LOG_ROOT}"
: > "${MASTER_LOG}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "[ERR] GPU_IDS must contain two GPUs for this script, got: ${GPU_IDS}" >&2
  exit 1
fi

log() {
  printf '%s\n' "$1" | tee -a "${MASTER_LOG}"
}

require_path() {
  local target="$1"
  if [[ ! -e "${target}" ]]; then
    echo "[ERR] Missing path: ${target}" >&2
    exit 1
  fi
}

prepare_dir() {
  local out_dir="$1"
  local eval_dir="$2"
  mkdir -p "${out_dir}" "${eval_dir}"
  rm -f "${out_dir}"/*.jsonl "${out_dir}"/*.json
  rm -f "${eval_dir}"/*.jsonl "${eval_dir}"/*.json "${eval_dir}"/*.md
}

merge_shards() {
  local key_name="$1"
  local output_file="$2"
  shift 2
  "${PY}" benchmark/vrsbench/scripts/merge_jsonl_shards.py \
    --inputs "$@" \
    --output "${output_file}" \
    --key "${key_name}" \
    --delete-inputs >> "${MASTER_LOG}" 2>&1
}

apply_patches() {
  local key_name="$1"
  local base_file="$2"
  shift 2
  "${PY}" benchmark/vrsbench/scripts/apply_jsonl_patch_by_key.py \
    --base "${base_file}" \
    --patches "$@" \
    --key "${key_name}" \
    --delete-patches >> "${MASTER_LOG}" 2>&1
}

run_dual_generate() {
  local name="$1"
  local key_name="$2"
  local script_path="$3"
  local output_file="$4"
  shift 4

  local shard0="${output_file%.jsonl}.gpu0.jsonl"
  local shard1="${output_file%.jsonl}.gpu1.jsonl"
  local log0="${LOG_ROOT}/${name}.gpu0.log"
  local log1="${LOG_ROOT}/${name}.gpu1.log"

  rm -f "${shard0}" "${shard1}" "${output_file}"
  log "[START] $(date '+%F %T') ${name}"

  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "${PY}" "${script_path}" \
    --output "${shard0}" \
    --batch-size "${GEN_BATCH_SIZE}" \
    --shard-world-size 2 \
    --shard-rank 0 \
    --shard-weights "${SHARD_WEIGHTS}" \
    "$@" > "${log0}" 2>&1 &
  local pid0=$!

  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[1]}" "${PY}" "${script_path}" \
    --output "${shard1}" \
    --batch-size "${GEN_BATCH_SIZE}" \
    --shard-world-size 2 \
    --shard-rank 1 \
    --shard-weights "${SHARD_WEIGHTS}" \
    "$@" > "${log1}" 2>&1 &
  local pid1=$!

  local status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "[ERR] ${name} failed." >&2
    return "${status}"
  fi

  merge_shards "${key_name}" "${output_file}" "${shard0}" "${shard1}"
  log "[END]   $(date '+%F %T') ${name}"
}

run_dual_fix_dino() {
  local name="$1"
  local preds_file="$2"
  local merger_ckpt="$3"
  local lora_dir="$4"
  shift 4

  local patch0="${preds_file%.jsonl}.fix.gpu0.jsonl"
  local patch1="${preds_file%.jsonl}.fix.gpu1.jsonl"
  local log0="${LOG_ROOT}/${name}.gpu0.log"
  local log1="${LOG_ROOT}/${name}.gpu1.log"

  rm -f "${patch0}" "${patch1}"
  log "[START] $(date '+%F %T') ${name}"

  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "${PY}" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py \
    --preds "${preds_file}" \
    --output "${patch0}" \
    --qwen-model-dir "${QWEN_MODEL_DIR}" \
    --dinov3-dir "${DINOV3_DIR}" \
    --merger-ckpt "${merger_ckpt}" \
    --lora-dir "${lora_dir}" \
    --image-size 256 \
    --smart-resize-min-pixels 65536 \
    --smart-resize-max-pixels 16777216 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-retries 10 \
    --batch-size "${FIX_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --shard-world-size 2 \
    --shard-rank 0 \
    --shard-weights "${SHARD_WEIGHTS}" > "${log0}" 2>&1 &
  local pid0=$!

  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[1]}" "${PY}" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py \
    --preds "${preds_file}" \
    --output "${patch1}" \
    --qwen-model-dir "${QWEN_MODEL_DIR}" \
    --dinov3-dir "${DINOV3_DIR}" \
    --merger-ckpt "${merger_ckpt}" \
    --lora-dir "${lora_dir}" \
    --image-size 256 \
    --smart-resize-min-pixels 65536 \
    --smart-resize-max-pixels 16777216 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-retries 10 \
    --batch-size "${FIX_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --shard-world-size 2 \
    --shard-rank 1 \
    --shard-weights "${SHARD_WEIGHTS}" > "${log1}" 2>&1 &
  local pid1=$!

  local status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "[ERR] ${name} failed." >&2
    return "${status}"
  fi

  apply_patches imgid "${preds_file}" "${patch0}" "${patch1}"
  log "[END]   $(date '+%F %T') ${name}"
}

run_dual_fix_sva() {
  local name="$1"
  local preds_file="$2"
  local merger_ckpt="$3"
  local lora_dir="$4"

  local patch0="${preds_file%.jsonl}.fix.gpu0.jsonl"
  local patch1="${preds_file%.jsonl}.fix.gpu1.jsonl"
  local log0="${LOG_ROOT}/${name}.gpu0.log"
  local log1="${LOG_ROOT}/${name}.gpu1.log"

  rm -f "${patch0}" "${patch1}"
  log "[START] $(date '+%F %T') ${name}"

  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" "${PY}" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_sva_deepstack_ca.py \
    --preds "${preds_file}" \
    --output "${patch0}" \
    --qwen-model-dir "${QWEN_MODEL_DIR}" \
    --dinov3-dir "${DINOV3_DIR}" \
    --merger-ckpt "${merger_ckpt}" \
    --lora-dir "${lora_dir}" \
    --smart-resize-min-pixels 65536 \
    --smart-resize-max-pixels 16777216 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-retries 10 \
    --batch-size "${FIX_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --shard-world-size 2 \
    --shard-rank 0 \
    --shard-weights "${SHARD_WEIGHTS}" > "${log0}" 2>&1 &
  local pid0=$!

  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[1]}" "${PY}" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_sva_deepstack_ca.py \
    --preds "${preds_file}" \
    --output "${patch1}" \
    --qwen-model-dir "${QWEN_MODEL_DIR}" \
    --dinov3-dir "${DINOV3_DIR}" \
    --merger-ckpt "${merger_ckpt}" \
    --lora-dir "${lora_dir}" \
    --smart-resize-min-pixels 65536 \
    --smart-resize-max-pixels 16777216 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-retries 10 \
    --batch-size "${FIX_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --shard-world-size 2 \
    --shard-rank 1 \
    --shard-weights "${SHARD_WEIGHTS}" > "${log1}" 2>&1 &
  local pid1=$!

  local status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "[ERR] ${name} failed." >&2
    return "${status}"
  fi

  apply_patches imgid "${preds_file}" "${patch0}" "${patch1}"
  log "[END]   $(date '+%F %T') ${name}"
}

run_caption_eval() {
  local name="$1"
  local preds_file="$2"
  local summary_file="$3"
  local log_file="${LOG_ROOT}/${name}.log"
  log "[START] $(date '+%F %T') ${name}"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${PY}" benchmark/vrsbench/scripts/eval_vrsbench_cap.py \
    --refs benchmark/vrsbench/data/vrsbench_refs_test.json \
    --preds "${preds_file}" \
    --output "${summary_file}" > "${log_file}" 2>&1
  log "[END]   $(date '+%F %T') ${name}"
}

run_grounding_eval() {
  local name="$1"
  local preds_file="$2"
  local summary_file="$3"
  local log_file="${LOG_ROOT}/${name}.log"
  log "[START] $(date '+%F %T') ${name}"
  "${PY}" benchmark/vrsbench/scripts/eval_vrsbench_referring.py \
    --preds "${preds_file}" \
    --meta benchmark/vrsbench/data/vrsbench_referring_meta.json \
    --output "${summary_file}" > "${log_file}" 2>&1
  log "[END]   $(date '+%F %T') ${name}"
}

run_grounding_eval_noftstyle() {
  local name="$1"
  local preds_file="$2"
  local summary_file="$3"
  local log_file="${LOG_ROOT}/${name}.log"
  log "[START] $(date '+%F %T') ${name}"
  "${PY}" benchmark/vrsbench/scripts/eval_referring_baseline_noftstyle.py \
    --preds "${preds_file}" \
    --meta benchmark/vrsbench/data/vrsbench_referring_meta.json \
    --output "${summary_file}" > "${log_file}" 2>&1
  log "[END]   $(date '+%F %T') ${name}"
}

run_baseline() {
  prepare_dir "${BASELINE_OUT}" "${BASELINE_EVAL}"

  run_dual_generate caption_baseline imgid benchmark/vrsbench/scripts/generate_baseline.py \
    "${BASELINE_OUT}/caption_baseline.jsonl" \
    --model-dir "${QWEN_MODEL_DIR}" \
    --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
    --prompt "${BASELINE_CAPTION_PROMPT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}"

  log "[START] $(date '+%F %T') caption_baseline_fix"
  "${PY}" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_baseline.py \
    --preds "${BASELINE_OUT}/caption_baseline.jsonl" \
    --model-dir "${QWEN_MODEL_DIR}" \
    --prompt "${BASELINE_CAPTION_PROMPT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-retries 10 > "${LOG_ROOT}/caption_baseline_fix.log" 2>&1
  log "[END]   $(date '+%F %T') caption_baseline_fix"

  run_caption_eval caption_baseline_eval \
    "${BASELINE_OUT}/caption_baseline.jsonl" \
    "${BASELINE_EVAL}/caption_summary.json"

  run_dual_generate grounding_baseline qid benchmark/vrsbench/scripts/generate_referring_baseline_noftstyle.py \
    "${BASELINE_OUT}/grounding_baseline.jsonl" \
    --model-dir "${QWEN_MODEL_DIR}" \
    --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}"

  run_grounding_eval_noftstyle grounding_baseline_eval \
    "${BASELINE_OUT}/grounding_baseline.jsonl" \
    "${BASELINE_EVAL}/grounding_summary.json"
}

run_dino() {
  prepare_dir "${DINO_OUT}" "${DINO_EVAL}"
  require_path "${DINO_CKPT_DIR}/merger.safetensors"
  require_path "${DINO_CKPT_DIR}/lora"

  run_dual_generate caption_dino imgid benchmark/vrsbench/scripts/generate_dinov3.py \
    "${DINO_OUT}/caption_smartbucket.jsonl" \
    --qwen-model-dir "${QWEN_MODEL_DIR}" \
    --dinov3-dir "${DINOV3_DIR}" \
    --merger-ckpt "${DINO_CKPT_DIR}/merger.safetensors" \
    --lora-dir "${DINO_CKPT_DIR}/lora" \
    --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
    --image-size 256 \
    --smart-resize-min-pixels 65536 \
    --smart-resize-max-pixels 16777216 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}"

  run_dual_fix_dino caption_dino_fix \
    "${DINO_OUT}/caption_smartbucket.jsonl" \
    "${DINO_CKPT_DIR}/merger.safetensors" \
    "${DINO_CKPT_DIR}/lora"

  run_caption_eval caption_dino_eval \
    "${DINO_OUT}/caption_smartbucket.jsonl" \
    "${DINO_EVAL}/caption_summary.json"

  run_dual_generate grounding_dino qid benchmark/vrsbench/scripts/generate_referring_dinov3.py \
    "${DINO_OUT}/grounding_smartbucket.jsonl" \
    --qwen-model-dir "${QWEN_MODEL_DIR}" \
    --dinov3-dir "${DINOV3_DIR}" \
    --merger-ckpt "${DINO_CKPT_DIR}/merger.safetensors" \
    --lora-dir "${DINO_CKPT_DIR}/lora" \
    --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
    --image-size 256 \
    --smart-resize-min-pixels 65536 \
    --smart-resize-max-pixels 16777216 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}"

  run_grounding_eval grounding_dino_eval \
    "${DINO_OUT}/grounding_smartbucket.jsonl" \
    "${DINO_EVAL}/grounding_summary.json"
}

run_sva() {
  prepare_dir "${SVA_OUT}" "${SVA_EVAL}"
  require_path "${SVA_CKPT_DIR}/merger.safetensors"
  require_path "${SVA_CKPT_DIR}/lora"

  run_dual_generate caption_sva imgid benchmark/vrsbench/scripts/generate_sva_deepstack_ca.py \
    "${SVA_OUT}/caption_sva.jsonl" \
    --qwen-model-dir "${QWEN_MODEL_DIR}" \
    --dinov3-dir "${DINOV3_DIR}" \
    --merger-ckpt "${SVA_CKPT_DIR}/merger.safetensors" \
    --lora-dir "${SVA_CKPT_DIR}/lora" \
    --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}"

  run_dual_fix_sva caption_sva_fix \
    "${SVA_OUT}/caption_sva.jsonl" \
    "${SVA_CKPT_DIR}/merger.safetensors" \
    "${SVA_CKPT_DIR}/lora"

  run_caption_eval caption_sva_eval \
    "${SVA_OUT}/caption_sva.jsonl" \
    "${SVA_EVAL}/caption_summary.json"

  run_dual_generate grounding_sva qid benchmark/vrsbench/scripts/generate_referring_sva_deepstack_ca.py \
    "${SVA_OUT}/grounding_sva.jsonl" \
    --qwen-model-dir "${QWEN_MODEL_DIR}" \
    --dinov3-dir "${DINOV3_DIR}" \
    --merger-ckpt "${SVA_CKPT_DIR}/merger.safetensors" \
    --lora-dir "${SVA_CKPT_DIR}/lora" \
    --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --dtype "${DTYPE}"

  run_grounding_eval grounding_sva_eval \
    "${SVA_OUT}/grounding_sva.jsonl" \
    "${SVA_EVAL}/grounding_summary.json"
}

main() {
  require_path "${QWEN_MODEL_DIR}"
  require_path "${DINOV3_DIR}"
  require_path "benchmark/vrsbench/data/vrsbench_images_test.jsonl"
  require_path "benchmark/vrsbench/data/vrsbench_referring_test.jsonl"
  log "[INFO] $(date '+%F %T') retest start"
  log "[INFO] GEN_BATCH_SIZE=${GEN_BATCH_SIZE} FIX_BATCH_SIZE=${FIX_BATCH_SIZE} MAX_NEW_TOKENS=${MAX_NEW_TOKENS} DTYPE=${DTYPE}"
  log "[INFO] GPU_IDS=${GPU_IDS} SHARD_WEIGHTS=${SHARD_WEIGHTS}"

  run_baseline
  run_dino
  run_sva

  log "[INFO] $(date '+%F %T') retest done"
}

main "$@"
