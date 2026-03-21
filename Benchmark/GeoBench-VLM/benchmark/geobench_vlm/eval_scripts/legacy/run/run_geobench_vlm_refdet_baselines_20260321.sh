#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../../../../../../.." && pwd)
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

GEOBENCH_SCRIPT_ROOT="Benchmark/GeoBench-VLM"
DATA_ROOT="../../../GeoBench-VLM/dataset/GEOBench-VLM"
PROMPT_VERSION="bbox2d1000_xyxy_json_array_v2"

model_dir_of() {
  case "$1" in
    qwen3vl) echo '../../../VRSBench/models/Qwen3-VL-8B-Instruct' ;;
    qwen35) echo '../../../../fine-tune-qwen3.5/models/Qwen3.5-9B' ;;
    *)
      echo "Unsupported model family: $1" >&2
      return 1
      ;;
  esac
}

run_one() {
  local env_name="$1"
  local model_family="$2"
  local run_name="$3"
  local batch_size="$4"
  local model_dir
  model_dir=$(model_dir_of "$model_family")

  local output_path="benchmark/geobench_vlm/outputs/$run_name/ref_det.jsonl"
  local summary_path="benchmark/geobench_vlm/eval_results/$run_name/ref_det_summary.json"
  local details_path="benchmark/geobench_vlm/eval_results/$run_name/ref_det_details.json"

  mkdir -p \
    "$GEOBENCH_SCRIPT_ROOT/$(dirname "$output_path")" \
    "$GEOBENCH_SCRIPT_ROOT/$(dirname "$summary_path")"

  conda run --no-capture-output -n "$env_name" \
    python "$GEOBENCH_SCRIPT_ROOT/src/legacy/geobenchvlm_refdet_generate.py" \
    --model-family "$model_family" \
    --model-dir "$model_dir" \
    --data-root "$DATA_ROOT" \
    --output "$output_path" \
    --batch-size "$batch_size"

  conda run --no-capture-output -n "$env_name" \
    python "$GEOBENCH_SCRIPT_ROOT/benchmark/geobench_vlm/eval_scripts/legacy/eval/eval_geobenchvlm_refdet.py" \
    --data "$DATA_ROOT/Ref-Det/qa.json" \
    --predictions "$output_path" \
    --output "$summary_path" \
    --details-output "$details_path" \
    --expected-model-family "$model_family" \
    --expected-model-dir "$model_dir" \
    --expected-prompt-version "$PROMPT_VERSION"
}

run_one qwen3-dinov3 qwen3vl 01_qwen3vl_baseline_20260319_cuda1_default 256
run_one qwen3-dinov3.5 qwen35 01_qwen35_baseline_20260319_cuda1_default 256
