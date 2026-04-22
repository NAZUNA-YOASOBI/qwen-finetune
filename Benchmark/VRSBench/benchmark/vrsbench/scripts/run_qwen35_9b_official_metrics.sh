#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/opt/yanzixi/home/yzx/.conda/envs/qwen3-dinov3.5/bin/python}"
API_BASE="${API_BASE:-https://infai.cc/v1}"
API_KEY_ENV="${API_KEY_ENV:-INFAI_API_KEY}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
MODEL_DIR="${MODEL_DIR:-/opt/yanzixi/project/fine-tune-qwen3.5/models/Qwen3.5-9B}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
VQA_GPU0="${VQA_GPU0:-0}"
VQA_GPU1="${VQA_GPU1:-1}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-64}"
VQA_BATCH_SIZE="${VQA_BATCH_SIZE:-512}"

CAP_DATA="${ROOT_DIR}/VRSBench/benchmark/vrsbench/data/vrsbench_images_test.jsonl"
CAP_OUT="${ROOT_DIR}/VRSBench/benchmark/vrsbench/outputs/official_caption_clair/qwen35_9b_base"
CAP_EVAL="${ROOT_DIR}/VRSBench/benchmark/vrsbench/eval/official_caption_clair/qwen35_9b_base"
CAP_LOG_DIR="${ROOT_DIR}/VRSBench/benchmark/vrsbench/logs/official_caption_clair"
VQA_DATA="${ROOT_DIR}/VRSBench/benchmark/vrsbench/data"
VQA_OUT="${ROOT_DIR}/VRSBench/benchmark/vrsbench/outputs/official_vqa_gpt/qwen35_9b_base"
VQA_OUT_SHARDS="${ROOT_DIR}/VRSBench/benchmark/vrsbench/outputs/official_vqa_gpt/qwen35_9b_base_shards"
VQA_EVAL="${ROOT_DIR}/VRSBench/benchmark/vrsbench/eval/official_vqa_gpt/qwen35_9b_base"
VQA_LOG_DIR="${ROOT_DIR}/VRSBench/benchmark/vrsbench/logs/official_vqa_gpt"

mkdir -p "${CAP_OUT}" "${CAP_EVAL}" "${CAP_LOG_DIR}" "${VQA_DATA}" "${VQA_OUT}" "${VQA_OUT_SHARDS}" "${VQA_EVAL}" "${VQA_LOG_DIR}"

echo "[START] $(date '+%F %T') qwen35 caption generate"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" VRSBench/benchmark/vrsbench/eval_scripts/ftqwen35/generate/generate_caption_qwen35_baseline.py \
  --model-dir "${MODEL_DIR}" \
  --data "${CAP_DATA}" \
  --output "${CAP_OUT}/caption_predictions.jsonl" \
  --batch-size "${CAPTION_BATCH_SIZE}" \
  --max-new-tokens 256 \
  --dtype bf16

echo "[START] $(date '+%F %T') qwen35 caption clair"
"${PYTHON_BIN}" VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_cap_clair.py \
  --preds "${CAP_OUT}/caption_predictions.jsonl" \
  --judged-output "${CAP_EVAL}/caption_predictions_clair.jsonl" \
  --summary-output "${CAP_EVAL}/caption_clair_summary.json" \
  --api-base "${API_BASE}" \
  --api-key-env "${API_KEY_ENV}" \
  --judge-model "${JUDGE_MODEL}" \
  --max-workers 512

echo "[START] $(date '+%F %T') qwen35 vqa prepare"
"${PYTHON_BIN}" VRSBench/benchmark/vrsbench/scripts/prepare_vrsbench_vqa.py \
  --output-dir "${VQA_DATA}"

echo "[START] $(date '+%F %T') qwen35 vqa generate"
CUDA_VISIBLE_DEVICES="${VQA_GPU0}" "${PYTHON_BIN}" VRSBench/benchmark/vrsbench/scripts/generate_vrsbench_vqa.py \
  --preset qwen35_9b_base \
  --data "${VQA_DATA}/vrsbench_vqa_test.jsonl" \
  --output "${VQA_OUT_SHARDS}/vqa_predictions.shard0.jsonl" \
  --device-map cuda:0 \
  --batch-size "${VQA_BATCH_SIZE}" \
  --do-sample \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --num-shards 2 \
  --shard-index 0 &
PID0=$!
CUDA_VISIBLE_DEVICES="${VQA_GPU1}" "${PYTHON_BIN}" VRSBench/benchmark/vrsbench/scripts/generate_vrsbench_vqa.py \
  --preset qwen35_9b_base \
  --data "${VQA_DATA}/vrsbench_vqa_test.jsonl" \
  --output "${VQA_OUT_SHARDS}/vqa_predictions.shard1.jsonl" \
  --device-map cuda:0 \
  --batch-size "${VQA_BATCH_SIZE}" \
  --do-sample \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --num-shards 2 \
  --shard-index 1 &
PID1=$!
wait "${PID0}"
wait "${PID1}"
"${PYTHON_BIN}" VRSBench/benchmark/vrsbench/eval_scripts/ftqwen35/utils/merge_jsonl_shards.py \
  --inputs "${VQA_OUT_SHARDS}/vqa_predictions.shard0.jsonl" "${VQA_OUT_SHARDS}/vqa_predictions.shard1.jsonl" \
  --output "${VQA_OUT}/vqa_predictions.jsonl" \
  --key qid \
  --delete-inputs

echo "[START] $(date '+%F %T') qwen35 vqa judge"
"${PYTHON_BIN}" VRSBench/benchmark/vrsbench/scripts/eval_vrsbench_vqa_gpt.py \
  --preds "${VQA_OUT}/vqa_predictions.jsonl" \
  --judged-output "${VQA_EVAL}/vqa_predictions_gpt.jsonl" \
  --summary-out "${VQA_EVAL}/vqa_gpt_summary.json" \
  --api-base "${API_BASE}" \
  --api-key-env "${API_KEY_ENV}" \
  --judge-model "${JUDGE_MODEL}"

echo "[DONE] $(date '+%F %T') qwen35 metrics finished"
