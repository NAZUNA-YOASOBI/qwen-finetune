#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${PY_BIN:-}" ]]; then
  PY_CMD=("${PY_BIN}")
else
  PY_CMD=(conda run --no-capture-output -n qwen3-dinov3 python)
fi

API_BASE="${API_BASE:-https://api.gptsapi.net/v1}"
API_KEY_ENV="${API_KEY_ENV:-GPTSAPI_KEY}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DATA_DIR="${DATA_DIR:-VRSBench/benchmark/vrsbench/data}"
OUTPUT_DIR="${OUTPUT_DIR:-VRSBench/benchmark/vrsbench/outputs/official_vqa_gpt/dinov3_epoch10}"
EVAL_DIR="${EVAL_DIR:-VRSBench/benchmark/vrsbench/eval/official_vqa_gpt/dinov3_epoch10}"

"${PY_CMD[@]}" VRSBench/benchmark/vrsbench/scripts/prepare_vrsbench_vqa.py \
  --output-dir "${DATA_DIR}"

"${PY_CMD[@]}" VRSBench/benchmark/vrsbench/scripts/generate_vrsbench_vqa.py \
  --preset dinov3_epoch10 \
  --data "${DATA_DIR}/vrsbench_vqa_test.jsonl" \
  --output "${OUTPUT_DIR}/vqa_predictions.jsonl" \
  --device-map "${DEVICE_MAP}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size 512 \
  --smart-resize-min-pixels 262144 \
  --smart-resize-max-pixels 262144

"${PY_CMD[@]}" VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_vqa_gpt.py \
  --preds "${OUTPUT_DIR}/vqa_predictions.jsonl" \
  --judged-output "${EVAL_DIR}/vqa_predictions_gpt.jsonl" \
  --summary-out "${EVAL_DIR}/vqa_gpt_summary.json" \
  --api-base "${API_BASE}" \
  --api-key-env "${API_KEY_ENV}" \
  --judge-model "${JUDGE_MODEL}"
