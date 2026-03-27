#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
IMAGE_DIR_TRAIN="${IMAGE_DIR_TRAIN:-data/Images_train}"
IMAGE_DIR_EVAL="${IMAGE_DIR_EVAL:-data/Images_val}"
EVAL_JSON="${EVAL_JSON:-data/VRSBench_EVAL_referring.json}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
BF16_FLAG="${BF16_FLAG:---bf16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}"
DATA_TAG="${DATA_TAG:-fullclip}"

SHOTS=(1 5 10)
VARIANTS=(v1 v2 v3 v4)

for shot in "${SHOTS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    run_name="vrsbench_${DATA_TAG}_${shot}shot_${variant}"
    output_dir="$ROOT_DIR/outputs/$run_name"
    pred_file="$ROOT_DIR/vg_eval_predictions_${DATA_TAG}_${shot}shot_${variant}.jsonl"
    metrics_file="$ROOT_DIR/vg_eval_metrics_${DATA_TAG}_${shot}shot_${variant}.json"
    train_file="$ROOT_DIR/data/VRSBench/RL_VRSBench_VG_full_${shot}shots_vlmr1_size_new_clip.jsonl"

    echo "[train] shot=$shot lora_variant=$variant train_file=$train_file output=$output_dir"
    PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" python train_sft_vrsbench.py \
      --model_name_or_path "$MODEL_NAME" \
      --shots "$shot" \
      --image_dir "$IMAGE_DIR_TRAIN" \
      --output_dir "$output_dir" \
      --lora \
      --lora_variant "$variant" \
      --gradient_checkpointing \
      $BF16_FLAG \
      --per_device_train_batch_size "$BATCH_SIZE_TRAIN" \
      --gradient_accumulation_steps "$GRAD_ACCUM" \
      --max_length "$MAX_LENGTH"

    echo "[eval] shot=$shot lora_variant=$variant model=$output_dir"
    python eval_referring.py \
      --model "$output_dir" \
      --eval_json "$EVAL_JSON" \
      --image_dir "$IMAGE_DIR_EVAL" \
      --dataset_output "$pred_file" \
      --metrics_output "$metrics_file" \
      --batch_size "$EVAL_BATCH_SIZE"

    echo "[done] shot=$shot variant=$variant model: $output_dir"
    echo "[done] shot=$shot variant=$variant predictions: $pred_file"
    echo "[done] shot=$shot variant=$variant metrics: $metrics_file"
  done
done
