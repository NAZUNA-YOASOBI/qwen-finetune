#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PY="/opt/yanzixi/home/yzx/.conda/envs/qwen3-dinov3/bin/python"

RUN_NAME="13_qwen8b_qwen_native_single4090_epoch10_sampleavg_20260315"
GROUP_NAME="qwen_native_lora"

OUT_DIR="benchmark/vrsbench/outputs/${GROUP_NAME}/${RUN_NAME}"
EVAL_DIR="benchmark/vrsbench/eval/${GROUP_NAME}/${RUN_NAME}"
LOG_DIR="benchmark/vrsbench/logs/${GROUP_NAME}/${RUN_NAME}"
MASTER_LOG="${LOG_DIR}/master.log"

QWEN_MODEL_DIR="models/Qwen3-VL-8B-Instruct"
MERGER_CKPT="checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_single4090_micro8_ga4_effective32_sampleavg_wd001_run_20260308_192357/epoch10/merger.safetensors"
LORA_DIR="checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_single4090_micro8_ga4_effective32_sampleavg_wd001_run_20260308_192357/epoch10/lora"

CAPTION_BATCH_SIZE=256
GROUNDING_BATCH_SIZE=256
MAX_NEW_TOKENS=256

mkdir -p "$OUT_DIR" "$EVAL_DIR" "$LOG_DIR"
: > "$MASTER_LOG"
rm -f "$OUT_DIR"/*.jsonl "$OUT_DIR"/*.json "$EVAL_DIR"/*.json "$EVAL_DIR"/*.md

log() {
  printf '%s\n' "$1" | tee -a "$MASTER_LOG"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path" >&2
    exit 1
  fi
}

run_sharded_generate() {
  local name="$1"
  local key_name="$2"
  local batch_size="$3"
  local script_path="$4"
  local output_prefix="$5"
  shift 5

  local shard0="${output_prefix}.gpu0.jsonl"
  local shard1="${output_prefix}.gpu1.jsonl"
  local merged="${output_prefix}.jsonl"
  local log0="${LOG_DIR}/${name}.gpu0.log"
  local log1="${LOG_DIR}/${name}.gpu1.log"
  local mergelog="${LOG_DIR}/${name}.merge.log"

  rm -f "$shard0" "$shard1" "$merged"

  log "[START] $(date '+%F %T') $name"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$script_path" \
    --output "$shard0" \
    --batch-size "$batch_size" \
    --shard-world-size 2 \
    --shard-rank 0 \
    --shard-weights 1:1 \
    "$@" > "$log0" 2>&1 &
  local pid0=$!

  CUDA_VISIBLE_DEVICES=1 "$PY" "$script_path" \
    --output "$shard1" \
    --batch-size "$batch_size" \
    --shard-world-size 2 \
    --shard-rank 1 \
    --shard-weights 1:1 \
    "$@" > "$log1" 2>&1 &
  local pid1=$!

  local status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  if [[ "$status" -ne 0 ]]; then
    log "[FAIL]  $(date '+%F %T') $name status=${status}"
    return "$status"
  fi

  "$PY" benchmark/vrsbench/scripts/merge_jsonl_shards.py \
    --inputs "$shard0" "$shard1" \
    --output "$merged" \
    --key "$key_name" \
    --delete-inputs > "$mergelog" 2>&1

  log "[END]   $(date '+%F %T') $name"
}

run_caption_fix() {
  local preds_file="$1"
  local log_file="${LOG_DIR}/caption_epoch10_fix.log"

  log "[START] $(date '+%F %T') caption_epoch10_fix_max_tokens"
  CUDA_VISIBLE_DEVICES=0 "$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_qwen_native.py \
    --preds "$preds_file" \
    --model-dir "$QWEN_MODEL_DIR" \
    --merger-ckpt "$MERGER_CKPT" \
    --lora-dir "$LORA_DIR" \
    --prompt "Describe the image in detail." \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-retries 10 \
    --dtype bf16 > "$log_file" 2>&1
  log "[END]   $(date '+%F %T') caption_epoch10_fix_max_tokens"
}

require_path "$PY"
require_path "$QWEN_MODEL_DIR"
require_path "$MERGER_CKPT"
require_path "$LORA_DIR"
require_path "benchmark/vrsbench/scripts/generate_qwen_native.py"
require_path "benchmark/vrsbench/scripts/generate_referring_qwen_native.py"
require_path "benchmark/vrsbench/scripts/fix_max_new_tokens_hits_qwen_native.py"
require_path "benchmark/vrsbench/scripts/eval_vrsbench_cap.py"
require_path "benchmark/vrsbench/scripts/eval_vrsbench_referring.py"
require_path "benchmark/vrsbench/scripts/merge_jsonl_shards.py"
require_path "benchmark/vrsbench/data/vrsbench_images_test.jsonl"
require_path "benchmark/vrsbench/data/vrsbench_refs_test.json"
require_path "benchmark/vrsbench/data/vrsbench_referring_test.jsonl"
require_path "benchmark/vrsbench/data/vrsbench_referring_meta.json"

log "[INFO] $(date '+%F %T') run=${RUN_NAME}"

run_sharded_generate "caption_epoch10_gen" imgid "$CAPTION_BATCH_SIZE" benchmark/vrsbench/scripts/generate_qwen_native.py \
  "$OUT_DIR/caption_qwen_native" \
  --model-dir "$QWEN_MODEL_DIR" \
  --merger-ckpt "$MERGER_CKPT" \
  --lora-dir "$LORA_DIR" \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype bf16

run_caption_fix "$OUT_DIR/caption_qwen_native.jsonl"

log "[START] $(date '+%F %T') caption_epoch10_eval"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" benchmark/vrsbench/scripts/eval_vrsbench_cap.py \
  --refs benchmark/vrsbench/data/vrsbench_refs_test.json \
  --preds "$OUT_DIR/caption_qwen_native.jsonl" \
  --output "$EVAL_DIR/caption_summary.json" > "$LOG_DIR/caption_epoch10_eval.log" 2>&1
log "[END]   $(date '+%F %T') caption_epoch10_eval"

run_sharded_generate "grounding_epoch10_gen" qid "$GROUNDING_BATCH_SIZE" benchmark/vrsbench/scripts/generate_referring_qwen_native.py \
  "$OUT_DIR/grounding_qwen_native" \
  --model-dir "$QWEN_MODEL_DIR" \
  --merger-ckpt "$MERGER_CKPT" \
  --lora-dir "$LORA_DIR" \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype bf16

log "[START] $(date '+%F %T') grounding_epoch10_eval"
"$PY" benchmark/vrsbench/scripts/eval_vrsbench_referring.py \
  --preds "$OUT_DIR/grounding_qwen_native.jsonl" \
  --meta benchmark/vrsbench/data/vrsbench_referring_meta.json \
  --output "$EVAL_DIR/grounding_summary.json" > "$LOG_DIR/grounding_epoch10_eval.log" 2>&1
log "[END]   $(date '+%F %T') grounding_epoch10_eval"

log "[DONE]  $(date '+%F %T') run=${RUN_NAME}"
