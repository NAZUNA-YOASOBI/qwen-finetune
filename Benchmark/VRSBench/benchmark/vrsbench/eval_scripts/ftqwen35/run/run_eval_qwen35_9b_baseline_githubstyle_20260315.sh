#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$PROJECT_ROOT"

PY="/opt/yanzixi/home/yzx/.conda/envs/qwen3-dinov3.5/bin/python"

RUN_NAME="02_qwen35_9b_baseline_githubstyle_20260315"
GROUP_NAME="qwen35_baseline"

OUT_DIR="benchmark/vrsbench/outputs/${GROUP_NAME}/${RUN_NAME}"
EVAL_DIR="benchmark/vrsbench/eval_results/${GROUP_NAME}/${RUN_NAME}"
LOG_DIR="benchmark/vrsbench/logs/${GROUP_NAME}/${RUN_NAME}"
MASTER_LOG="${LOG_DIR}/master.log"

MODEL_DIR="/opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-9B"

CAPTION_BATCH_SIZE=64
GROUNDING_BATCH_SIZE=64
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

  "$PY" benchmark/vrsbench/eval_scripts/ftqwen35/utils/merge_jsonl_shards.py \
    --inputs "$shard0" "$shard1" \
    --output "$merged" \
    --key "$key_name" \
    --delete-inputs > "$mergelog" 2>&1

  log "[END]   $(date '+%F %T') $name"
}

require_path "$PY"
require_path "$MODEL_DIR"
require_path "benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_caption_qwen35_baseline_githubstyle.py"
require_path "benchmark/vrsbench/eval_scripts/ftqwen35/fix/fix_max_new_tokens_hits_qwen35_githubstyle.py"
require_path "benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_referring_qwen35_baseline_noftstyle.py"
require_path "benchmark/vrsbench/eval_scripts/ftqwen35/eval/eval_referring_baseline_noftstyle.py"
require_path "benchmark/vrsbench/eval_scripts/ftqwen35/eval/eval_vrsbench_cap.py"
require_path "benchmark/vrsbench/eval_scripts/ftqwen35/utils/merge_jsonl_shards.py"
require_path "benchmark/vrsbench/data/vrsbench_images_test.jsonl"
require_path "benchmark/vrsbench/data/vrsbench_refs_test.json"
require_path "benchmark/vrsbench/data/vrsbench_referring_test.jsonl"
require_path "benchmark/vrsbench/data/vrsbench_referring_meta.json"

log "[INFO] $(date '+%F %T') run=${RUN_NAME}"

run_sharded_generate "caption_qwen35_9b_gen" imgid "$CAPTION_BATCH_SIZE" benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_caption_qwen35_baseline_githubstyle.py \
  "$OUT_DIR/caption_qwen35_9b" \
  --model-dir "$MODEL_DIR" \
  --data benchmark/vrsbench/data/vrsbench_images_test.jsonl \
  --prompt "Describe the image in detail in 2 to 4 sentences." \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype bf16

log "[START] $(date '+%F %T') caption_qwen35_9b_fix"
CUDA_VISIBLE_DEVICES=0 "$PY" benchmark/vrsbench/eval_scripts/ftqwen35/fix/fix_max_new_tokens_hits_qwen35_githubstyle.py \
  --preds "$OUT_DIR/caption_qwen35_9b.jsonl" \
  --model-dir "$MODEL_DIR" \
  --prompt "Describe the image in detail in 2 to 4 sentences." \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-retries 10 > "$LOG_DIR/caption_fix.log" 2>&1
log "[END]   $(date '+%F %T') caption_qwen35_9b_fix"

log "[START] $(date '+%F %T') caption_qwen35_9b_eval"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" benchmark/vrsbench/eval_scripts/ftqwen35/eval/eval_vrsbench_cap.py \
  --refs benchmark/vrsbench/data/vrsbench_refs_test.json \
  --preds "$OUT_DIR/caption_qwen35_9b.jsonl" \
  --output "$EVAL_DIR/caption_summary.json" > "$LOG_DIR/caption_eval.log" 2>&1
log "[END]   $(date '+%F %T') caption_qwen35_9b_eval"

run_sharded_generate "grounding_qwen35_9b_gen" qid "$GROUNDING_BATCH_SIZE" benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_referring_qwen35_baseline_noftstyle.py \
  "$OUT_DIR/grounding_qwen35_9b" \
  --model-dir "$MODEL_DIR" \
  --data benchmark/vrsbench/data/vrsbench_referring_test.jsonl \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --dtype bf16

log "[START] $(date '+%F %T') grounding_qwen35_9b_eval"
"$PY" benchmark/vrsbench/eval_scripts/ftqwen35/eval/eval_referring_baseline_noftstyle.py \
  --preds "$OUT_DIR/grounding_qwen35_9b.jsonl" \
  --meta benchmark/vrsbench/data/vrsbench_referring_meta.json \
  --output "$EVAL_DIR/grounding_summary.json" > "$LOG_DIR/grounding_eval.log" 2>&1
log "[END]   $(date '+%F %T') grounding_qwen35_9b_eval"

log "[DONE]  $(date '+%F %T') run=${RUN_NAME}"
