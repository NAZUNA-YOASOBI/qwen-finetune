#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-/opt/data/private/YanZiXi/home/yzx/miniconda3/envs/qwen3.5-dinov3/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-../../../LHRS-Bench/datasets/LHRS-Bench}"
MODEL_DIR="${MODEL_DIR:-/opt/data/private/YanZiXi/project/fine-tune-qwen3.5/models/Qwen3.5-9B}"
RUN_TAG="${RUN_TAG:-01_qwen35_9b_baseline_$(date +%Y%m%d)}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-512}"

DATA_DIR="benchmark/lhrsbench/data"
PAPER_JSON="benchmark/lhrsbench/paper/table5_lhrs_paper.json"
ATTEMPTS_JSONL="${DATA_DIR}/lhrsbench_attempts_r4_seed42.jsonl"
QUESTIONS_JSONL="${DATA_DIR}/lhrsbench_questions.jsonl"
META_JSON="${DATA_DIR}/lhrsbench_meta.json"

OUT_DIR="benchmark/lhrsbench/outputs/${RUN_TAG}"
EVAL_DIR="benchmark/lhrsbench/eval_results/${RUN_TAG}"
LOG_DIR="benchmark/lhrsbench/logs/${RUN_TAG}"
MASTER_LOG="${LOG_DIR}/master.log"

BASELINE_SUMMARY="${BASELINE_SUMMARY:-benchmark/lhrsbench/eval_results/04_lhrsbench_smartbucket_epoch1to10_20260216/baseline8b_summary.json}"
PREDS_JSONL="${OUT_DIR}/qwen35_9b_predictions.jsonl"
SUMMARY_JSON="${EVAL_DIR}/qwen35_9b_summary.json"
REPORT_MD="${EVAL_DIR}/compare_paper_vs_qwen35.md"

mkdir -p "$DATA_DIR" "$OUT_DIR" "$EVAL_DIR" "$LOG_DIR"
: > "$MASTER_LOG"

log() {
  printf '%s\n' "$1" | tee -a "$MASTER_LOG"
}

run_step() {
  local name="$1"
  shift
  log "[START] $(date '+%F %T') ${name}"
  "$@" > "${LOG_DIR}/${name}.log" 2>&1
  log "[END]   $(date '+%F %T') ${name}"
}

run_step prepare_lhrsbench_data \
  "$PY" benchmark/lhrsbench/eval_scripts/ftqwen3/prepare/prepare_lhrsbench_qa.py \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$DATA_DIR" \
  --num-repeats 4 \
  --seed 42

if [[ ! -f "$ATTEMPTS_JSONL" || ! -f "$QUESTIONS_JSONL" || ! -f "$META_JSON" ]]; then
  log "[ERROR] Prepared data files are missing after prepare step."
  exit 1
fi

run_step generate_qwen35_9b \
  "$PY" benchmark/lhrsbench/eval_scripts/ftqwen35/generate/generate_lhrsbench_qwen35_baseline.py \
  --model-dir "$MODEL_DIR" \
  --data "$ATTEMPTS_JSONL" \
  --output "$PREDS_JSONL" \
  --model-tag "qwen3.5-9b-baseline" \
  --max-new-tokens 10 \
  --batch-size "$BATCH_SIZE" \
  --dtype "$DTYPE" \
  --no-sample \
  --num-beams 1

run_step eval_qwen35_9b \
  "$PY" benchmark/lhrsbench/eval_scripts/ftqwen3/eval/eval_lhrsbench_qa.py \
  --data "$ATTEMPTS_JSONL" \
  --preds "$PREDS_JSONL" \
  --output "$SUMMARY_JSON" \
  --num-repeats 4

if [[ -f "$PAPER_JSON" && -f "$BASELINE_SUMMARY" ]]; then
  run_step make_report \
    "$PY" benchmark/lhrsbench/eval_scripts/ftqwen3/report/make_report_lhrs.py \
    --paper "$PAPER_JSON" \
    --baseline "$BASELINE_SUMMARY" \
    --ours-summary "$SUMMARY_JSON" \
    --ours-name "Qwen3.5-9B-baseline" \
    --output "$REPORT_MD"
else
  log "[WARN] Skip report because paper or baseline summary is missing."
fi

log "[DONE]  $(date '+%F %T') LHRS-Bench Qwen3.5-9B baseline pipeline completed"
