#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../../../../../../.." && pwd)
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

GEOBENCH_SCRIPT_ROOT="Benchmark/GeoBench-VLM"
DATA_ROOT="../../../GeoBench-VLM/dataset/GEOBench-VLM"

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

prompt_version_of() {
  case "$1" in
    single) echo 'official_mcq_single_v1' ;;
    temporal) echo 'official_mcq_temporal_all_frames_v1' ;;
    captioning) echo 'dataset_caption_prompt_v1' ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}

task_data_path() {
  case "$1" in
    single) echo "$DATA_ROOT/Single/qa.json" ;;
    temporal) echo "$DATA_ROOT/Temporal/qa.json" ;;
    captioning) echo "$DATA_ROOT/Captioning/qa.json" ;;
    *)
      echo "Unsupported task: $1" >&2
      return 1
      ;;
  esac
}

run_task() {
  local env_name="$1"
  local model_family="$2"
  local model_dir="$3"
  local run_name="$4"
  local task_name="$5"
  local batch_size="$6"
  local output_path="benchmark/geobench_vlm/outputs/$run_name/${task_name}.jsonl"
  local summary_path="benchmark/geobench_vlm/eval_results/$run_name/${task_name}_summary.json"
  local details_path="benchmark/geobench_vlm/eval_results/$run_name/${task_name}_details.json"
  local data_path
  local prompt_version
  data_path=$(task_data_path "$task_name")
  prompt_version=$(prompt_version_of "$task_name")

  mkdir -p \
    "$GEOBENCH_SCRIPT_ROOT/$(dirname "$output_path")" \
    "$GEOBENCH_SCRIPT_ROOT/$(dirname "$summary_path")"

  conda run --no-capture-output -n "$env_name" \
    python "$GEOBENCH_SCRIPT_ROOT/src/legacy/geobenchvlm_generate.py" \
    --task "$task_name" \
    --model-family "$model_family" \
    --model-dir "$model_dir" \
    --data-root "$DATA_ROOT" \
    --output "$output_path" \
    --batch-size "$batch_size"

  case "$task_name" in
    single|temporal)
      conda run --no-capture-output -n "$env_name" \
        python "$GEOBENCH_SCRIPT_ROOT/benchmark/geobench_vlm/eval_scripts/legacy/eval/eval_geobenchvlm_mcq.py" \
        --data "$data_path" \
        --predictions "$output_path" \
        --output "$summary_path" \
        --details-output "$details_path" \
        --expected-model-family "$model_family" \
        --expected-model-dir "$model_dir" \
        --expected-prompt-version "$prompt_version"
      ;;
    captioning)
      conda run --no-capture-output -n "$env_name" \
        python "$GEOBENCH_SCRIPT_ROOT/benchmark/geobench_vlm/eval_scripts/legacy/eval/eval_geobenchvlm_captioning.py" \
        --data "$data_path" \
        --predictions "$output_path" \
        --output "$summary_path" \
        --details-output "$details_path" \
        --expected-model-family "$model_family" \
        --expected-model-dir "$model_dir" \
        --expected-prompt-version "$prompt_version" \
        --device cuda
      ;;
    *)
      echo "Unsupported task: $task_name" >&2
      return 1
      ;;
  esac
}

run_model() {
  local env_name="$1"
  local model_family="$2"
  local run_name="$3"
  local single_bs="$4"
  local temporal_bs="$5"
  local caption_bs="$6"
  local model_dir
  model_dir=$(model_dir_of "$model_family")

  run_task "$env_name" "$model_family" "$model_dir" "$run_name" single "$single_bs"
  run_task "$env_name" "$model_family" "$model_dir" "$run_name" temporal "$temporal_bs"
  run_task "$env_name" "$model_family" "$model_dir" "$run_name" captioning "$caption_bs"
}

run_model qwen3-dinov3 qwen3vl 01_qwen3vl_baseline_20260319_cuda1_default 16 1 2
run_model qwen3-dinov3.5 qwen35 01_qwen35_baseline_20260319_cuda1_default 1 1 1
