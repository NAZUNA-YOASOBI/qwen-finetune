#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_DIR="${RUN_DIR:-checkpoints/vrsbench_joint/merger_lora_8b_sva_deepstack_ca_micro8_8_ga2_effective32_taskseq_sampleavg_wd001_run_20260313_083153}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark/vrsbench/eval_results/sva_deepstack_ca/01_sva_deepstack_ca_epoch10to1_20260314}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
DINOV3_DIR="${DINOV3_DIR:-models/dinov3/dinov3-vitl16-pretrain-sat493m}"
PY="${PY:-python}"

START_EPOCH="${START_EPOCH:-10}"
END_EPOCH="${END_EPOCH:-1}"

GPU_IDS="${GPU_IDS:-0,1}"
SHARD_WEIGHTS="${SHARD_WEIGHTS:-}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-128}"
CAPTION_FIX_BATCH_SIZE="${CAPTION_FIX_BATCH_SIZE:-256}"
GROUNDING_BATCH_SIZE="${GROUNDING_BATCH_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-bf16}"

SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "[ERR] GPU_IDS is empty." >&2
  exit 1
fi

if [[ "${GPU_COUNT}" -eq 1 ]]; then
  SHARD_WORLD_SIZE=1
else
  SHARD_WORLD_SIZE=2
fi

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

run_bg_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN][BG] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@" &
}

require_path() {
  local target="$1"
  if [[ ! -e "${target}" ]]; then
    echo "[ERR] Missing path: ${target}" >&2
    exit 1
  fi
}

launch_caption_epoch() {
  local epoch="$1"
  local ckpt_dir="${RUN_DIR}/epoch${epoch}"
  local epoch_dir="${OUTPUT_ROOT}/epoch${epoch}"
  local caption_dir="${epoch_dir}/caption"
  local merged_output="${caption_dir}/caption_epoch${epoch}.jsonl"
  local summary_output="${caption_dir}/caption_summary.json"
  local shard0_output="${caption_dir}/caption_epoch${epoch}.gpu0.jsonl"
  local shard1_output="${caption_dir}/caption_epoch${epoch}.gpu1.jsonl"
  local patch0_output="${caption_dir}/caption_epoch${epoch}.fix.gpu0.jsonl"
  local patch1_output="${caption_dir}/caption_epoch${epoch}.fix.gpu1.jsonl"

  mkdir -p "${caption_dir}"
  require_path "${ckpt_dir}/merger.safetensors"
  require_path "${ckpt_dir}/lora"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${summary_output}" ]]; then
    echo "[SKIP] caption epoch${epoch} summary exists: ${summary_output}"
    return 0
  fi

  local log0="${LOG_DIR}/epoch${epoch}_caption_gpu0.log"
  local log1="${LOG_DIR}/epoch${epoch}_caption_gpu1.log"
  local fix_log0="${LOG_DIR}/epoch${epoch}_caption_fix_gpu0.log"
  local fix_log1="${LOG_DIR}/epoch${epoch}_caption_fix_gpu1.log"
  local eval_log="${LOG_DIR}/epoch${epoch}_caption_eval.log"

  if [[ "${SHARD_WORLD_SIZE}" -eq 1 ]]; then
    run_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_caption_sva_deepstack_ca.py \
      --qwen-model-dir '${MODEL_DIR}' \
      --dinov3-dir '${DINOV3_DIR}' \
      --merger-ckpt '${ckpt_dir}/merger.safetensors' \
      --lora-dir '${ckpt_dir}/lora' \
      --output '${shard0_output}' \
      --batch-size '${CAPTION_BATCH_SIZE}' \
      --max-new-tokens '${MAX_NEW_TOKENS}' \
      --dtype '${DTYPE}' \
      --shard-world-size 1 \
      --shard-rank 0 \
      ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log0}' 2>&1"
    run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/merge_jsonl_shards.py \
      --inputs '${shard0_output}' \
      --output '${merged_output}' \
      --key imgid \
      --delete-inputs"
    run_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/fix/fix_max_new_tokens_hits_sva_deepstack_ca.py \
      --preds '${merged_output}' \
      --output '${patch0_output}' \
      --qwen-model-dir '${MODEL_DIR}' \
      --dinov3-dir '${DINOV3_DIR}' \
      --merger-ckpt '${ckpt_dir}/merger.safetensors' \
      --lora-dir '${ckpt_dir}/lora' \
      --prompt 'Describe the image in detail.' \
      --max-new-tokens '${MAX_NEW_TOKENS}' \
      --max-retries 10 \
      --batch-size '${CAPTION_FIX_BATCH_SIZE}' \
      --dtype '${DTYPE}' \
      --shard-world-size 1 \
      --shard-rank 0 > '${fix_log0}' 2>&1"
    run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/apply_jsonl_patch_by_key.py \
      --base '${merged_output}' \
      --patches '${patch0_output}' \
      --key imgid \
      --delete-patches"
  else
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_bg_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_caption_sva_deepstack_ca.py \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --output '${shard0_output}' \
        --batch-size '${CAPTION_BATCH_SIZE}' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 0 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log0}' 2>&1"
      run_bg_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_caption_sva_deepstack_ca.py \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --output '${shard1_output}' \
        --batch-size '${CAPTION_BATCH_SIZE}' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 1 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log1}' 2>&1"
      run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/merge_jsonl_shards.py \
        --inputs '${shard0_output}' '${shard1_output}' \
        --output '${merged_output}' \
        --key imgid \
        --delete-inputs"
      run_bg_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/fix/fix_max_new_tokens_hits_sva_deepstack_ca.py \
        --preds '${merged_output}' \
        --output '${patch0_output}' \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --prompt 'Describe the image in detail.' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --max-retries 10 \
        --batch-size '${CAPTION_FIX_BATCH_SIZE}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 0 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${fix_log0}' 2>&1"
      run_bg_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/fix/fix_max_new_tokens_hits_sva_deepstack_ca.py \
        --preds '${merged_output}' \
        --output '${patch1_output}' \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --prompt 'Describe the image in detail.' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --max-retries 10 \
        --batch-size '${CAPTION_FIX_BATCH_SIZE}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 1 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${fix_log1}' 2>&1"
      run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/apply_jsonl_patch_by_key.py \
        --base '${merged_output}' \
        --patches '${patch0_output}' '${patch1_output}' \
        --key imgid \
        --delete-patches"
    else
      bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_caption_sva_deepstack_ca.py \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --output '${shard0_output}' \
        --batch-size '${CAPTION_BATCH_SIZE}' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 0 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log0}' 2>&1" &
      local pid0=$!
      bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_caption_sva_deepstack_ca.py \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --output '${shard1_output}' \
        --batch-size '${CAPTION_BATCH_SIZE}' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 1 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log1}' 2>&1" &
      local pid1=$!
      wait "${pid0}"
      wait "${pid1}"
      run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/merge_jsonl_shards.py \
        --inputs '${shard0_output}' '${shard1_output}' \
        --output '${merged_output}' \
        --key imgid \
        --delete-inputs"

      bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/fix/fix_max_new_tokens_hits_sva_deepstack_ca.py \
        --preds '${merged_output}' \
        --output '${patch0_output}' \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --prompt 'Describe the image in detail.' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --max-retries 10 \
        --batch-size '${CAPTION_FIX_BATCH_SIZE}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 0 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${fix_log0}' 2>&1" &
      local fix_pid0=$!
      bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/fix/fix_max_new_tokens_hits_sva_deepstack_ca.py \
        --preds '${merged_output}' \
        --output '${patch1_output}' \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --prompt 'Describe the image in detail.' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --max-retries 10 \
        --batch-size '${CAPTION_FIX_BATCH_SIZE}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 1 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${fix_log1}' 2>&1" &
      local fix_pid1=$!
      wait "${fix_pid0}"
      wait "${fix_pid1}"
      run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/apply_jsonl_patch_by_key.py \
        --base '${merged_output}' \
        --patches '${patch0_output}' '${patch1_output}' \
        --key imgid \
        --delete-patches"
    fi
  fi

  run_cmd bash -lc "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/eval/eval_vrsbench_cap.py \
    --refs 'benchmark/vrsbench/data/vrsbench_refs_test.json' \
    --preds '${merged_output}' \
    --output '${summary_output}' > '${eval_log}' 2>&1"
}

launch_grounding_epoch() {
  local epoch="$1"
  local ckpt_dir="${RUN_DIR}/epoch${epoch}"
  local epoch_dir="${OUTPUT_ROOT}/epoch${epoch}"
  local grounding_dir="${epoch_dir}/grounding"
  local merged_output="${grounding_dir}/grounding_epoch${epoch}.jsonl"
  local summary_output="${grounding_dir}/grounding_summary.json"
  local shard0_output="${grounding_dir}/grounding_epoch${epoch}.gpu0.jsonl"
  local shard1_output="${grounding_dir}/grounding_epoch${epoch}.gpu1.jsonl"

  mkdir -p "${grounding_dir}"
  require_path "${ckpt_dir}/merger.safetensors"
  require_path "${ckpt_dir}/lora"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${summary_output}" ]]; then
    echo "[SKIP] grounding epoch${epoch} summary exists: ${summary_output}"
    return 0
  fi

  local log0="${LOG_DIR}/epoch${epoch}_grounding_gpu0.log"
  local log1="${LOG_DIR}/epoch${epoch}_grounding_gpu1.log"
  local eval_log="${LOG_DIR}/epoch${epoch}_grounding_eval.log"

  if [[ "${SHARD_WORLD_SIZE}" -eq 1 ]]; then
    run_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_referring_sva_deepstack_ca.py \
      --qwen-model-dir '${MODEL_DIR}' \
      --dinov3-dir '${DINOV3_DIR}' \
      --merger-ckpt '${ckpt_dir}/merger.safetensors' \
      --lora-dir '${ckpt_dir}/lora' \
      --output '${shard0_output}' \
      --batch-size '${GROUNDING_BATCH_SIZE}' \
      --max-new-tokens '${MAX_NEW_TOKENS}' \
      --dtype '${DTYPE}' \
      --shard-world-size 1 \
      --shard-rank 0 \
      ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log0}' 2>&1"
    run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/merge_jsonl_shards.py \
      --inputs '${shard0_output}' \
      --output '${merged_output}' \
      --key qid \
      --delete-inputs"
  else
    if [[ "${DRY_RUN}" == "1" ]]; then
      run_bg_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_referring_sva_deepstack_ca.py \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --output '${shard0_output}' \
        --batch-size '${GROUNDING_BATCH_SIZE}' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 0 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log0}' 2>&1"
      run_bg_cmd bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_referring_sva_deepstack_ca.py \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --output '${shard1_output}' \
        --batch-size '${GROUNDING_BATCH_SIZE}' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 1 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log1}' 2>&1"
      run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/merge_jsonl_shards.py \
        --inputs '${shard0_output}' '${shard1_output}' \
        --output '${merged_output}' \
        --key qid \
        --delete-inputs"
    else
      bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[0]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_referring_sva_deepstack_ca.py \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --output '${shard0_output}' \
        --batch-size '${GROUNDING_BATCH_SIZE}' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 0 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log0}' 2>&1" &
      local pid0=$!
      bash -lc "CUDA_VISIBLE_DEVICES=${GPU_ARRAY[1]} '${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/generate/generate_referring_sva_deepstack_ca.py \
        --qwen-model-dir '${MODEL_DIR}' \
        --dinov3-dir '${DINOV3_DIR}' \
        --merger-ckpt '${ckpt_dir}/merger.safetensors' \
        --lora-dir '${ckpt_dir}/lora' \
        --output '${shard1_output}' \
        --batch-size '${GROUNDING_BATCH_SIZE}' \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --dtype '${DTYPE}' \
        --shard-world-size 2 \
        --shard-rank 1 \
        ${SHARD_WEIGHTS:+--shard-weights '${SHARD_WEIGHTS}'} > '${log1}' 2>&1" &
      local pid1=$!
      wait "${pid0}"
      wait "${pid1}"
      run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/utils/merge_jsonl_shards.py \
        --inputs '${shard0_output}' '${shard1_output}' \
        --output '${merged_output}' \
        --key qid \
        --delete-inputs"
    fi
  fi

  run_cmd bash -lc "'${PY}' benchmark/vrsbench/eval_scripts/ftqwen3/eval/eval_vrsbench_referring.py \
    --preds '${merged_output}' \
    --meta 'benchmark/vrsbench/data/vrsbench_referring_meta.json' \
    --output '${summary_output}' > '${eval_log}' 2>&1"
}

main() {
  require_path "${RUN_DIR}"
  require_path "${MODEL_DIR}"
  require_path "${DINOV3_DIR}"

  if [[ "${START_EPOCH}" -lt "${END_EPOCH}" ]]; then
    echo "[ERR] START_EPOCH must be >= END_EPOCH for reverse evaluation." >&2
    exit 1
  fi

  for (( epoch=START_EPOCH; epoch>=END_EPOCH; epoch-- )); do
    echo "[INFO] evaluating epoch${epoch}"
    launch_caption_epoch "${epoch}"
    launch_grounding_epoch "${epoch}"
  done

  echo "[OK] reverse evaluation finished: ${OUTPUT_ROOT}"
}

main "$@"
