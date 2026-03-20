#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../../../../../../.." && pwd)
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

GEOBENCH_SCRIPT_ROOT="Benchmark/GeoBench-VLM"
OUT_DIR="benchmark/geobench_vlm/outputs/02_qwen35_refdet_xgrammar_v6_pixel_20260320"
EVAL_DIR="benchmark/geobench_vlm/eval_results/02_qwen35_refdet_xgrammar_v6_pixel_20260320"
MODEL_DIR="../../../../fine-tune-qwen3.5/models/Qwen3.5-9B"
ATTEMPT_LOG=$(mktemp)
trap 'rm -f "$ATTEMPT_LOG"' EXIT
mkdir -p "$GEOBENCH_SCRIPT_ROOT/$OUT_DIR" "$GEOBENCH_SCRIPT_ROOT/$EVAL_DIR"

batch_size="${BATCH_SIZE:-256}"
while true; do
  echo "[INFO] qwen35 generation attempt with batch_size=${batch_size}"
  if conda run --no-capture-output -n qwen3-dinov3.5 \
    python "$GEOBENCH_SCRIPT_ROOT/src/xgrammar/geobenchvlm_generate.py" \
    --task ref_det \
    --model-family qwen35 \
    --output "$OUT_DIR/ref_det.jsonl" \
    --model-dir "$MODEL_DIR" \
    --max-new-tokens 512 \
    --batch-size "$batch_size" >"$ATTEMPT_LOG" 2>&1; then
    cat "$ATTEMPT_LOG"
    break
  fi
  status=$?
  cat "$ATTEMPT_LOG"
  if ! grep -qiE 'out of memory|cuda out of memory|oom' "$ATTEMPT_LOG"; then
    exit "$status"
  fi
  if [ "$batch_size" -le 1 ]; then
    exit "$status"
  fi
  next_batch_size=$(( (batch_size + 1) / 2 ))
  if [ "$next_batch_size" -ge "$batch_size" ]; then
    next_batch_size=$(( batch_size - 1 ))
  fi
  echo "[WARN] qwen35 OOM at batch_size=${batch_size}, retry with batch_size=${next_batch_size}"
  batch_size="$next_batch_size"
done

conda run --no-capture-output -n qwen3-dinov3.5 \
  python "$GEOBENCH_SCRIPT_ROOT/benchmark/geobench_vlm/eval_scripts/xgrammar/eval/eval_geobenchvlm_refdet.py" \
  --data ../../../GeoBench-VLM/dataset/GEOBench-VLM/Ref-Det/qa.json \
  --predictions "$OUT_DIR/ref_det.jsonl" \
  --output "$EVAL_DIR/ref_det_summary.json" \
  --details-output "$EVAL_DIR/ref_det_details.json" \
  --expected-model-family qwen35 \
  --expected-model-dir "$MODEL_DIR" \
  --expected-prompt-version bbox2d_pixel_polygon4_json_array_v9_xgrammar_countlocked
