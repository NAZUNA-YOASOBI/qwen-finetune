#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PY="/opt/yanzixi/home/yzx/.conda/envs/qwen3-dinov3/bin/python"
OUT_ROOT="benchmark/vrsbench/eval/prompt_modified"
LOG_ROOT="$OUT_ROOT/logs"
MASTER_LOG="$LOG_ROOT/master.log"
BASELINE_CAPTION_PROMPT="Describe the image in detail in 2 to 4 sentences."

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
: > "$MASTER_LOG"

log() {
  printf '%s\n' "$1" | tee -a "$MASTER_LOG"
}

run_sharded_generate() {
  local name="$1"
  local key_name="$2"
  local batch_size="$3"
  local script_path="$4"
  local out_dir="$5"
  shift 5

  local shard0="$out_dir/${name}.gpu0.jsonl"
  local shard1="$out_dir/${name}.gpu1.jsonl"
  local merged="$out_dir/${name}.jsonl"
  local log0="$LOG_ROOT/${name}.gpu0.log"
  local log1="$LOG_ROOT/${name}.gpu1.log"

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
    return "$status"
  fi

  "$PY" benchmark/vrsbench/scripts/merge_jsonl_shards.py \
    --inputs "$shard0" "$shard1" \
    --output "$merged" \
    --key "$key_name" \
    --delete-inputs >> "$MASTER_LOG" 2>&1
  log "[END]   $(date '+%F %T') $name"
}

run_dino_fix() {
  local name="$1"
  local preds_file="$2"
  local merger_ckpt="$3"
  local lora_dir="$4"
  local fix_batch_size="$5"
  shift 5

  local patch0="${preds_file%.jsonl}.fix.gpu0.jsonl"
  local patch1="${preds_file%.jsonl}.fix.gpu1.jsonl"
  local log0="$LOG_ROOT/${name}.gpu0.log"
  local log1="$LOG_ROOT/${name}.gpu1.log"

  rm -f "$patch0" "$patch1"
  log "[START] $(date '+%F %T') $name"

  if [[ -n "$lora_dir" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py \
      --qwen-model-dir models/Qwen3-VL-8B-Instruct \
      --preds "$preds_file" \
      --output "$patch0" \
      --merger-ckpt "$merger_ckpt" \
      --lora-dir "$lora_dir" \
      --max-new-tokens 256 \
      --max-sentences 7 \
      --max-retries 10 \
      --batch-size "$fix_batch_size" \
      --shard-world-size 2 \
      --shard-rank 0 \
      --shard-weights 1:1 \
      "$@" > "$log0" 2>&1 &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 "$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py \
      --qwen-model-dir models/Qwen3-VL-8B-Instruct \
      --preds "$preds_file" \
      --output "$patch1" \
      --merger-ckpt "$merger_ckpt" \
      --lora-dir "$lora_dir" \
      --max-new-tokens 256 \
      --max-sentences 7 \
      --max-retries 10 \
      --batch-size "$fix_batch_size" \
      --shard-world-size 2 \
      --shard-rank 1 \
      --shard-weights 1:1 \
      "$@" > "$log1" 2>&1 &
    local pid1=$!
  else
    CUDA_VISIBLE_DEVICES=0 "$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py \
      --qwen-model-dir models/Qwen3-VL-8B-Instruct \
      --preds "$preds_file" \
      --output "$patch0" \
      --merger-ckpt "$merger_ckpt" \
      --max-new-tokens 256 \
      --max-sentences 7 \
      --max-retries 10 \
      --batch-size "$fix_batch_size" \
      --shard-world-size 2 \
      --shard-rank 0 \
      --shard-weights 1:1 \
      "$@" > "$log0" 2>&1 &
    local pid0=$!

    CUDA_VISIBLE_DEVICES=1 "$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_dinov3.py \
      --qwen-model-dir models/Qwen3-VL-8B-Instruct \
      --preds "$preds_file" \
      --output "$patch1" \
      --merger-ckpt "$merger_ckpt" \
      --max-new-tokens 256 \
      --max-sentences 7 \
      --max-retries 10 \
      --batch-size "$fix_batch_size" \
      --shard-world-size 2 \
      --shard-rank 1 \
      --shard-weights 1:1 \
      "$@" > "$log1" 2>&1 &
    local pid1=$!
  fi

  local status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  if [[ "$status" -ne 0 ]]; then
    return "$status"
  fi

  "$PY" benchmark/vrsbench/scripts/apply_jsonl_patch_by_key.py \
    --base "$preds_file" \
    --patches "$patch0" "$patch1" \
    --key imgid \
    --delete-patches >> "$MASTER_LOG" 2>&1
  log "[END]   $(date '+%F %T') $name"
}

run_caption_eval() {
  local name="$1"
  local preds_file="$2"
  local summary_file="$3"
  local log_file="$LOG_ROOT/${name}.log"
  log "[START] $(date '+%F %T') $name"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" benchmark/vrsbench/scripts/eval_vrsbench_cap.py \
    --refs benchmark/vrsbench/data/vrsbench_refs_test.json \
    --preds "$preds_file" \
    --output "$summary_file" > "$log_file" 2>&1
  log "[END]   $(date '+%F %T') $name"
}

run_grounding_eval() {
  local name="$1"
  local preds_file="$2"
  local summary_file="$3"
  local log_file="$LOG_ROOT/${name}.log"
  log "[START] $(date '+%F %T') $name"
  "$PY" benchmark/vrsbench/scripts/eval_vrsbench_referring.py \
    --preds "$preds_file" \
    --meta benchmark/vrsbench/data/vrsbench_referring_meta.json \
    --output "$summary_file" > "$log_file" 2>&1
  log "[END]   $(date '+%F %T') $name"
}

run_grounding_eval_noftstyle() {
  local name="$1"
  local preds_file="$2"
  local summary_file="$3"
  local log_file="$LOG_ROOT/${name}.log"
  log "[START] $(date '+%F %T') $name"
  "$PY" benchmark/vrsbench/scripts/eval_referring_baseline_noftstyle.py \
    --preds "$preds_file" \
    --meta benchmark/vrsbench/data/vrsbench_referring_meta.json \
    --output "$summary_file" > "$log_file" 2>&1
  log "[END]   $(date '+%F %T') $name"
}

prepare_model_dir() {
  local model_dir="$1"
  mkdir -p "$model_dir"
  rm -f "$model_dir"/*.jsonl "$model_dir"/*.json
}

BASELINE_DIR="$OUT_ROOT/01_baseline_qwen3vl8b"
MERGER_ONLY_DIR="$OUT_ROOT/02_merger_only_epoch10_fixed256"
MERGER_LORA_DIR="$OUT_ROOT/03_merger_lora_epoch10_fixed256"
SMART_DIR="$OUT_ROOT/04_merger_lora_epoch10_smartresize"
QNATIVE_DIR="$OUT_ROOT/05_qwen_native_epoch10"

for d in "$BASELINE_DIR" "$MERGER_ONLY_DIR" "$MERGER_LORA_DIR" "$SMART_DIR" "$QNATIVE_DIR"; do
  prepare_model_dir "$d"
done

log "[INFO] $(date '+%F %T') prompt_modified suite start"

run_sharded_generate caption_baseline imgid 128 benchmark/vrsbench/scripts/generate_baseline.py "$BASELINE_DIR" \
  --model-dir models/Qwen3-VL-8B-Instruct \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --prompt "$BASELINE_CAPTION_PROMPT" \
  --max-new-tokens 256 \
  --dtype bf16
"$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_baseline.py \
  --preds "$BASELINE_DIR/caption_baseline.jsonl" \
  --model-dir models/Qwen3-VL-8B-Instruct \
  --prompt "$BASELINE_CAPTION_PROMPT" \
  --max-new-tokens 256 \
  --max-sentences 4 \
  --max-retries 10 > "$LOG_ROOT/caption_baseline_fix.log" 2>&1
run_caption_eval caption_baseline_eval "$BASELINE_DIR/caption_baseline.jsonl" "$BASELINE_DIR/caption_summary.json"
run_sharded_generate grounding_baseline qid 128 benchmark/vrsbench/scripts/generate_referring_baseline_noftstyle.py "$BASELINE_DIR" \
  --model-dir models/Qwen3-VL-8B-Instruct \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --max-new-tokens 256 \
  --dtype bf16
run_grounding_eval_noftstyle grounding_baseline_eval "$BASELINE_DIR/grounding_baseline.jsonl" "$BASELINE_DIR/grounding_summary.json"

run_sharded_generate caption_merger_only imgid 256 benchmark/vrsbench/scripts/generate_dinov3.py "$MERGER_ONLY_DIR" \
  --qwen-model-dir models/Qwen3-VL-8B-Instruct \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --merger-ckpt checkpoints/vrsbench_joint/merger_only_8b_dinov3_micro8_24_ga1_effective32_taskseq_run_20260210_210858/epoch10/merger.safetensors \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 65536 \
  --max-new-tokens 256 \
  --dtype bf16
run_dino_fix caption_merger_only_fix "$MERGER_ONLY_DIR/caption_merger_only.jsonl" \
  checkpoints/vrsbench_joint/merger_only_8b_dinov3_micro8_24_ga1_effective32_taskseq_run_20260210_210858/epoch10/merger.safetensors \
  "" 256 \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 65536 \
  --dtype bf16
run_caption_eval caption_merger_only_eval "$MERGER_ONLY_DIR/caption_merger_only.jsonl" "$MERGER_ONLY_DIR/caption_summary.json"
run_sharded_generate grounding_merger_only qid 256 benchmark/vrsbench/scripts/generate_referring_dinov3.py "$MERGER_ONLY_DIR" \
  --qwen-model-dir models/Qwen3-VL-8B-Instruct \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --merger-ckpt checkpoints/vrsbench_joint/merger_only_8b_dinov3_micro8_24_ga1_effective32_taskseq_run_20260210_210858/epoch10/merger.safetensors \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 65536 \
  --max-new-tokens 256 \
  --dtype bf16
run_grounding_eval grounding_merger_only_eval "$MERGER_ONLY_DIR/grounding_merger_only.jsonl" "$MERGER_ONLY_DIR/grounding_summary.json"

run_sharded_generate caption_merger_lora imgid 256 benchmark/vrsbench/scripts/generate_dinov3.py "$MERGER_LORA_DIR" \
  --qwen-model-dir models/Qwen3-VL-8B-Instruct \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --merger-ckpt checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro2_6_ga4_effective32_taskseq_resume_from_epoch5_20260208/epoch10/merger.safetensors \
  --lora-dir checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro2_6_ga4_effective32_taskseq_resume_from_epoch5_20260208/epoch10/lora \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 65536 \
  --max-new-tokens 256 \
  --dtype bf16
run_dino_fix caption_merger_lora_fix "$MERGER_LORA_DIR/caption_merger_lora.jsonl" \
  checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro2_6_ga4_effective32_taskseq_resume_from_epoch5_20260208/epoch10/merger.safetensors \
  checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro2_6_ga4_effective32_taskseq_resume_from_epoch5_20260208/epoch10/lora 256 \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 65536 \
  --dtype bf16
run_caption_eval caption_merger_lora_eval "$MERGER_LORA_DIR/caption_merger_lora.jsonl" "$MERGER_LORA_DIR/caption_summary.json"
run_sharded_generate grounding_merger_lora qid 256 benchmark/vrsbench/scripts/generate_referring_dinov3.py "$MERGER_LORA_DIR" \
  --qwen-model-dir models/Qwen3-VL-8B-Instruct \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --merger-ckpt checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro2_6_ga4_effective32_taskseq_resume_from_epoch5_20260208/epoch10/merger.safetensors \
  --lora-dir checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro2_6_ga4_effective32_taskseq_resume_from_epoch5_20260208/epoch10/lora \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 65536 \
  --max-new-tokens 256 \
  --dtype bf16
run_grounding_eval grounding_merger_lora_eval "$MERGER_LORA_DIR/grounding_merger_lora.jsonl" "$MERGER_LORA_DIR/grounding_summary.json"

run_sharded_generate caption_smartbucket imgid 128 benchmark/vrsbench/scripts/generate_dinov3.py "$SMART_DIR" \
  --qwen-model-dir models/Qwen3-VL-8B-Instruct \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --merger-ckpt checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_run_20260213_153823/epoch10/merger.safetensors \
  --lora-dir checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_run_20260213_153823/epoch10/lora \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 16777216 \
  --max-new-tokens 256 \
  --dtype bf16
run_dino_fix caption_smartbucket_fix "$SMART_DIR/caption_smartbucket.jsonl" \
  checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_run_20260213_153823/epoch10/merger.safetensors \
  checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_run_20260213_153823/epoch10/lora 128 \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 16777216 \
  --dtype bf16
run_caption_eval caption_smartbucket_eval "$SMART_DIR/caption_smartbucket.jsonl" "$SMART_DIR/caption_summary.json"
run_sharded_generate grounding_smartbucket qid 128 benchmark/vrsbench/scripts/generate_referring_dinov3.py "$SMART_DIR" \
  --qwen-model-dir models/Qwen3-VL-8B-Instruct \
  --dinov3-dir models/dinov3/dinov3-vitl16-pretrain-sat493m \
  --merger-ckpt checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_run_20260213_153823/epoch10/merger.safetensors \
  --lora-dir checkpoints/vrsbench_joint/merger_lora_8b_dinov3_micro8_8_ga2_effective32_taskseq_smartbucket_run_20260213_153823/epoch10/lora \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --image-size 256 \
  --smart-resize-min-pixels 65536 \
  --smart-resize-max-pixels 16777216 \
  --max-new-tokens 256 \
  --dtype bf16
run_grounding_eval grounding_smartbucket_eval "$SMART_DIR/grounding_smartbucket.jsonl" "$SMART_DIR/grounding_summary.json"

run_sharded_generate caption_qwen_native imgid 128 benchmark/vrsbench/scripts/generate_qwen_native.py "$QNATIVE_DIR" \
  --model-dir models/Qwen3-VL-8B-Instruct \
  --merger-ckpt checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/merger.safetensors \
  --lora-dir checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/lora \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --max-new-tokens 256 \
  --dtype bf16
log "[START] $(date '+%F %T') caption_qwen_native_fix"
CUDA_VISIBLE_DEVICES=0 "$PY" benchmark/vrsbench/scripts/fix_max_new_tokens_hits_qwen_native.py \
  --preds "$QNATIVE_DIR/caption_qwen_native.jsonl" \
  --model-dir models/Qwen3-VL-8B-Instruct \
  --merger-ckpt checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/merger.safetensors \
  --lora-dir checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/lora \
  --prompt "Describe the image in detail." \
  --max-new-tokens 256 \
  --max-sentences 7 \
  --max-retries 10 > "$LOG_ROOT/caption_qwen_native_fix.log" 2>&1
log "[END]   $(date '+%F %T') caption_qwen_native_fix"
run_caption_eval caption_qwen_native_eval "$QNATIVE_DIR/caption_qwen_native.jsonl" "$QNATIVE_DIR/caption_summary.json"
run_sharded_generate grounding_qwen_native qid 256 benchmark/vrsbench/scripts/generate_referring_qwen_native.py "$QNATIVE_DIR" \
  --model-dir models/Qwen3-VL-8B-Instruct \
  --merger-ckpt checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/merger.safetensors \
  --lora-dir checkpoints/vrsbench_joint/merger_lora_8b_qwen_native_micro8_8_ga2_effective32_wd001_taskseq_run_20260302_160151/epoch10/lora \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --max-new-tokens 256 \
  --dtype bf16
run_grounding_eval grounding_qwen_native_eval "$QNATIVE_DIR/grounding_qwen_native.jsonl" "$QNATIVE_DIR/grounding_summary.json"

log "[START] $(date '+%F %T') make_prompt_modified_report"
"$PY" benchmark/vrsbench/scripts/make_report_prompt_modified.py > "$LOG_ROOT/make_report.log" 2>&1
log "[END]   $(date '+%F %T') make_prompt_modified_report"
log "[DONE]  $(date '+%F %T') prompt_modified suite completed"
