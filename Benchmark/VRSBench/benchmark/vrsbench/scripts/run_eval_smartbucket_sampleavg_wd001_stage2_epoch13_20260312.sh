#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PY="/opt/yanzixi/home/yzx/.conda/envs/qwen3-dinov3/bin/python"

RUN_NAME="17_qwen8b_smartbucket_sampleavg_wd001_stage2_epoch13_20260312"
GROUP_NAME="merger_lora_smartbucket512_stage2"

OUT_DIR="benchmark/vrsbench/outputs/${GROUP_NAME}/${RUN_NAME}"
EVAL_DIR="benchmark/vrsbench/eval/${GROUP_NAME}/${RUN_NAME}"
LOG_DIR="benchmark/vrsbench/logs/${GROUP_NAME}/${RUN_NAME}"
MASTER_LOG="${LOG_DIR}/master.log"

MERGER_CKPT="checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_sampleavg_wd001_stage2_from_epoch10_lr5e5_run_20260311_125829/epoch13/merger.safetensors"
LORA_DIR="checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_sampleavg_wd001_stage2_from_epoch10_lr5e5_run_20260311_125829/epoch13/lora"

QWEN_MODEL_DIR="models/Qwen3-VL-8B-Instruct"
DINO_DIR="models/dinov3/dinov3-vitl16-pretrain-sat493m"

IMAGE_SIZE=512
SMART_MIN_PIXELS=50176
SMART_MAX_PIXELS=262144
CAPTION_BATCH_SIZE=128
CAPTION_FIX_BATCH_SIZE=128
GROUNDING_BATCH_SIZE=128
MAX_NEW_TOKENS=256

mkdir -p "$OUT_DIR" "$EVAL_DIR" "$LOG_DIR"
: > "$MASTER_LOG"
rm -f "$OUT_DIR"/*.jsonl "$OUT_DIR"/*.json "$EVAL_DIR"/*.json "$EVAL_DIR"/*.md

log() {
  printf '%s\n' "$1" | tee -a "$MASTER_LOG"
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

run_dino_fix() {
  local name="$1"
  local preds_file="$2"
  local log0="${LOG_DIR}/${name}.gpu0.log"
  local log1="${LOG_DIR}/${name}.gpu1.log"
  local patch0="${preds_file%.jsonl}.fix.gpu0.jsonl"
  local patch1="${preds_file%.jsonl}.fix.gpu1.jsonl"
  local applylog="${LOG_DIR}/${name}.apply.log"

  rm -f "$patch0" "$patch1"
  log "[START] $(date '+%F %T') $name"

  CUDA_VISIBLE_DEVICES=0 "$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py \
    --preds "$preds_file" \
    --output "$patch0" \
    --qwen-model-dir "$QWEN_MODEL_DIR" \
    --dinov3-dir "$DINO_DIR" \
    --merger-ckpt "$MERGER_CKPT" \
    --lora-dir "$LORA_DIR" \
    --image-size "$IMAGE_SIZE" \
    --smart-resize-min-pixels "$SMART_MIN_PIXELS" \
    --smart-resize-max-pixels "$SMART_MAX_PIXELS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-retries 10 \
    --batch-size "$CAPTION_FIX_BATCH_SIZE" \
    --shard-world-size 2 \
    --shard-rank 0 \
    --shard-weights 1:1 \
    --dtype bf16 > "$log0" 2>&1 &
  local pid0=$!

  CUDA_VISIBLE_DEVICES=1 "$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py \
    --preds "$preds_file" \
    --output "$patch1" \
    --qwen-model-dir "$QWEN_MODEL_DIR" \
    --dinov3-dir "$DINO_DIR" \
    --merger-ckpt "$MERGER_CKPT" \
    --lora-dir "$LORA_DIR" \
    --image-size "$IMAGE_SIZE" \
    --smart-resize-min-pixels "$SMART_MIN_PIXELS" \
    --smart-resize-max-pixels "$SMART_MAX_PIXELS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-retries 10 \
    --batch-size "$CAPTION_FIX_BATCH_SIZE" \
    --shard-world-size 2 \
    --shard-rank 1 \
    --shard-weights 1:1 \
    --dtype bf16 > "$log1" 2>&1 &
  local pid1=$!

  local status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  if [[ "$status" -ne 0 ]]; then
    log "[FAIL]  $(date '+%F %T') $name status=${status}"
    return "$status"
  fi

  "$PY" benchmark/vrsbench/scripts/apply_jsonl_patch_by_key.py \
    --base "$preds_file" \
    --patches "$patch0" "$patch1" \
    --key imgid \
    --delete-patches > "$applylog" 2>&1

  log "[END]   $(date '+%F %T') $name"
}

log "[INFO] $(date '+%F %T') run=${RUN_NAME}"

run_sharded_generate "caption_epoch13_gen" imgid "$CAPTION_BATCH_SIZE" benchmark/vrsbench/scripts/generate_dinov3.py \
  "$OUT_DIR/caption_epoch13" \
  --qwen-model-dir "$QWEN_MODEL_DIR" \
  --dinov3-dir "$DINO_DIR" \
  --merger-ckpt "$MERGER_CKPT" \
  --lora-dir "$LORA_DIR" \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --image-size "$IMAGE_SIZE" \
  --smart-resize-min-pixels "$SMART_MIN_PIXELS" \
  --smart-resize-max-pixels "$SMART_MAX_PIXELS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype bf16

run_dino_fix "caption_epoch13_fix_max_tokens" "$OUT_DIR/caption_epoch13.jsonl"

log "[START] $(date '+%F %T') caption_epoch13_eval"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" benchmark/vrsbench/scripts/eval_vrsbench_cap.py \
  --refs benchmark/vrsbench/data/vrsbench_refs_test.json \
  --preds "$OUT_DIR/caption_epoch13.jsonl" \
  --output "$EVAL_DIR/caption_summary.json" > "$LOG_DIR/caption_epoch13_eval.log" 2>&1
log "[END]   $(date '+%F %T') caption_epoch13_eval"

run_sharded_generate "grounding_epoch13_gen" qid "$GROUNDING_BATCH_SIZE" benchmark/vrsbench/scripts/generate_referring_dinov3.py \
  "$OUT_DIR/grounding_epoch13" \
  --qwen-model-dir "$QWEN_MODEL_DIR" \
  --dinov3-dir "$DINO_DIR" \
  --merger-ckpt "$MERGER_CKPT" \
  --lora-dir "$LORA_DIR" \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --image-size "$IMAGE_SIZE" \
  --smart-resize-min-pixels "$SMART_MIN_PIXELS" \
  --smart-resize-max-pixels "$SMART_MAX_PIXELS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype bf16

log "[START] $(date '+%F %T') grounding_epoch13_eval"
"$PY" benchmark/vrsbench/scripts/eval_vrsbench_referring.py \
  --preds "$OUT_DIR/grounding_epoch13.jsonl" \
  --meta benchmark/vrsbench/data/vrsbench_referring_meta.json \
  --output "$EVAL_DIR/grounding_summary.json" > "$LOG_DIR/grounding_epoch13_eval.log" 2>&1
log "[END]   $(date '+%F %T') grounding_epoch13_eval"

log "[DONE]  $(date '+%F %T') run=${RUN_NAME}"
