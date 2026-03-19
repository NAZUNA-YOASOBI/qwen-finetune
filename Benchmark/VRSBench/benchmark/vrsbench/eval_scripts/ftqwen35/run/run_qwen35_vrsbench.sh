#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$PROJECT_ROOT"

PY="/opt/yanzixi/home/yzx/.conda/envs/qwen3-dinov3.5/bin/python"
OUT_ROOT="benchmark/vrsbench/eval_results/01_qwen35_baselines_20260306"
LOG_ROOT="$OUT_ROOT/logs"
MASTER_LOG="$LOG_ROOT/master.log"

Q4_DIR="$OUT_ROOT/01_qwen35_4b"
Q9_DIR="$OUT_ROOT/02_qwen35_9b"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$Q4_DIR" "$Q9_DIR"
: > "$MASTER_LOG"

log() {
  printf '%s\n' "$1" | tee -a "$MASTER_LOG"
}

prepare_model_dir() {
  local model_dir="$1"
  mkdir -p "$model_dir"
  rm -f "$model_dir"/*.jsonl "$model_dir"/*.json
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
  CUDA_VISIBLE_DEVICES=0 "$PY" "$script_path" --output "$shard0" --batch-size "$batch_size" --shard-world-size 2 --shard-rank 0 --shard-weights 1:1 "$@" > "$log0" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=1 "$PY" "$script_path" --output "$shard1" --batch-size "$batch_size" --shard-world-size 2 --shard-rank 1 --shard-weights 1:1 "$@" > "$log1" 2>&1 &
  local pid1=$!
  local status=0
  wait "$pid0" || status=$?
  wait "$pid1" || status=$?
  if [[ "$status" -ne 0 ]]; then
    return "$status"
  fi
  "$PY" benchmark/vrsbench/eval_scripts/ftqwen35/utils/merge_jsonl_shards.py --inputs "$shard0" "$shard1" --output "$merged" --key "$key_name" --delete-inputs >> "$MASTER_LOG" 2>&1
  log "[END]   $(date '+%F %T') $name"
}

run_caption_eval() {
  local name="$1"
  local preds_file="$2"
  local summary_file="$3"
  local log_file="$LOG_ROOT/${name}.log"
  log "[START] $(date '+%F %T') $name"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" benchmark/vrsbench/eval_scripts/ftqwen35/eval/eval_vrsbench_cap.py --refs benchmark/vrsbench/data/vrsbench_refs_test.json --preds "$preds_file" --output "$summary_file" > "$log_file" 2>&1
  log "[END]   $(date '+%F %T') $name"
}

run_grounding_eval_noftstyle() {
  local name="$1"
  local preds_file="$2"
  local summary_file="$3"
  local log_file="$LOG_ROOT/${name}.log"
  log "[START] $(date '+%F %T') $name"
  "$PY" benchmark/vrsbench/eval_scripts/ftqwen35/eval/eval_referring_baseline_noftstyle.py --preds "$preds_file" --meta benchmark/vrsbench/data/vrsbench_referring_meta.json --output "$summary_file" > "$log_file" 2>&1
  log "[END]   $(date '+%F %T') $name"
}

prepare_model_dir "$Q4_DIR"
prepare_model_dir "$Q9_DIR"

log "[INFO] $(date '+%F %T') qwen3.5 VRSBench eval start"

run_sharded_generate caption_qwen35_4b imgid 128 benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_caption_qwen35_baseline.py "$Q4_DIR" \
  --model-dir /opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-4B \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --prompt "Describe the image in detail in 2 to 4 sentences." \
  --max-new-tokens 256 \
  --dtype bf16
log "[START] $(date '+%F %T') caption_qwen35_4b_fix"
CUDA_VISIBLE_DEVICES=0 "$PY" benchmark/vrsbench/eval_scripts/ftqwen35/fix/fix_max_new_tokens_hits_qwen35.py \
  --preds "$Q4_DIR/caption_qwen35_4b.jsonl" \
  --model-dir /opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-4B \
  --prompt "Describe the image in detail in 2 to 4 sentences." \
  --max-new-tokens 256 \
  --max-retries 10 > "$LOG_ROOT/caption_qwen35_4b_fix.log" 2>&1
log "[END]   $(date '+%F %T') caption_qwen35_4b_fix"
run_caption_eval caption_qwen35_4b_eval "$Q4_DIR/caption_qwen35_4b.jsonl" "$Q4_DIR/caption_summary.json"
run_sharded_generate grounding_qwen35_4b qid 128 benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_referring_qwen35_baseline_noftstyle.py "$Q4_DIR" \
  --model-dir /opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-4B \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --max-new-tokens 256 \
  --dtype bf16
run_grounding_eval_noftstyle grounding_qwen35_4b_eval "$Q4_DIR/grounding_qwen35_4b.jsonl" "$Q4_DIR/grounding_summary.json"

run_sharded_generate caption_qwen35_9b imgid 64 benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_caption_qwen35_baseline.py "$Q9_DIR" \
  --model-dir /opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-9B \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --prompt "Describe the image in detail in 2 to 4 sentences." \
  --max-new-tokens 256 \
  --dtype bf16
log "[START] $(date '+%F %T') caption_qwen35_9b_fix"
CUDA_VISIBLE_DEVICES=0 "$PY" benchmark/vrsbench/eval_scripts/ftqwen35/fix/fix_max_new_tokens_hits_qwen35.py \
  --preds "$Q9_DIR/caption_qwen35_9b.jsonl" \
  --model-dir /opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-9B \
  --prompt "Describe the image in detail in 2 to 4 sentences." \
  --max-new-tokens 256 \
  --max-retries 10 > "$LOG_ROOT/caption_qwen35_9b_fix.log" 2>&1
log "[END]   $(date '+%F %T') caption_qwen35_9b_fix"
run_caption_eval caption_qwen35_9b_eval "$Q9_DIR/caption_qwen35_9b.jsonl" "$Q9_DIR/caption_summary.json"
run_sharded_generate grounding_qwen35_9b qid 64 benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_referring_qwen35_baseline_noftstyle.py "$Q9_DIR" \
  --model-dir /opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-9B \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --max-new-tokens 256 \
  --dtype bf16
run_grounding_eval_noftstyle grounding_qwen35_9b_eval "$Q9_DIR/grounding_qwen35_9b.jsonl" "$Q9_DIR/grounding_summary.json"

log "[START] $(date '+%F %T') make_report"
"$PY" benchmark/vrsbench/eval_scripts/ftqwen35/report/make_report_qwen35_baselines.py > "$LOG_ROOT/make_report.log" 2>&1
log "[END]   $(date '+%F %T') make_report"
log "[DONE]  $(date '+%F %T') qwen3.5 VRSBench eval completed"
