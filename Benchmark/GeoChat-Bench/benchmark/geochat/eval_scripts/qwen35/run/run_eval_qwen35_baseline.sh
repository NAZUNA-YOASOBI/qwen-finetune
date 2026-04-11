#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${ROOT_DIR}"

PY_BIN="${PY_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-../../fine-tune-qwen3.5/models/Qwen3.5-9B}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
OUT_TAG="${OUT_TAG:-01_qwen35_baseline}"
SHARD_WORLD_SIZE="${SHARD_WORLD_SIZE:-1}"
SHARD_GPU_IDS="${SHARD_GPU_IDS:-0}"
SHARD_WEIGHTS="${SHARD_WEIGHTS:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

AID_DATA="${AID_DATA:-dataset/GeoChat-Bench/aid.jsonl}"
UCMERCED_DATA="${UCMERCED_DATA:-dataset/GeoChat-Bench/UCmerced.jsonl}"
HRBEN_DATA="${HRBEN_DATA:-dataset/GeoChat-Bench/hrben.jsonl}"
LRBEN_DATA="${LRBEN_DATA:-dataset/GeoChat-Bench/lrben.jsonl}"
REGION_CAPTION_DATA="${REGION_CAPTION_DATA:-dataset/GeoChat-Bench/region_captioning.jsonl}"
REFERRING_DATA="${REFERRING_DATA:-dataset/GeoChat-Bench/referring.jsonl}"

OUT_DIR="benchmark/geochat/outputs/${OUT_TAG}"
EVAL_DIR="benchmark/geochat/eval/${OUT_TAG}"
LOG_DIR="benchmark/geochat/logs/${OUT_TAG}"
mkdir -p "${OUT_DIR}" "${EVAL_DIR}" "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"
LOCK_FILE="${LOG_DIR}/run.lock"
if command -v flock > /dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  flock -n 9 || {
    echo "[ERROR] Another run is already using OUT_TAG=${OUT_TAG}. lock=${LOCK_FILE}" >&2
    exit 2
  }
fi
: > "${MASTER_LOG}"

AID_IMAGE_ROOT="${AID_IMAGE_ROOT:-}"
UCMERCED_IMAGE_ROOT="${UCMERCED_IMAGE_ROOT:-}"
HRBEN_IMAGE_ROOT="${HRBEN_IMAGE_ROOT:-}"
LRBEN_IMAGE_ROOT="${LRBEN_IMAGE_ROOT:-}"
HRBEN_RAW_QUESTIONS_FILE="${HRBEN_RAW_QUESTIONS_FILE:-}"
HRBEN_RAW_ANSWERS_FILE="${HRBEN_RAW_ANSWERS_FILE:-}"
HRBEN_RAW_IMAGES_FILE="${HRBEN_RAW_IMAGES_FILE:-}"
HRBEN_GT_FILE="${HRBEN_GT_FILE:-}"
LRBEN_GT_FILE="${LRBEN_GT_FILE:-}"
GEOCHAT_IMAGE_ROOT="${GEOCHAT_IMAGE_ROOT:-}"
REGION_IMAGE_ROOT="${REGION_IMAGE_ROOT:-${GEOCHAT_IMAGE_ROOT}}"
REFERRING_IMAGE_ROOT="${REFERRING_IMAGE_ROOT:-${GEOCHAT_IMAGE_ROOT}}"

if [[ -z "${SHARD_WEIGHTS}" ]]; then
  if [[ "${SHARD_WORLD_SIZE}" == "1" ]]; then
    SHARD_WEIGHTS="1"
  elif [[ "${SHARD_WORLD_SIZE}" == "2" ]]; then
    SHARD_WEIGHTS="1:1"
  fi
fi

IFS=',' read -r -a GPU_ID_ARR <<< "${SHARD_GPU_IDS}"

log() {
  printf '%s\n' "$1" | tee -a "${MASTER_LOG}"
}

is_truthy() {
  local raw="${1:-}"
  raw="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  [[ "${raw}" == "1" || "${raw}" == "true" || "${raw}" == "yes" || "${raw}" == "y" ]]
}

json_file_is_valid() {
  local path="$1"
  [[ -s "${path}" ]] || return 1
  "${PY_BIN}" - "${path}" <<'PY' > /dev/null 2>&1
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

has_shard_files() {
  local glob_pattern="$1"
  compgen -G "${glob_pattern}" > /dev/null
}

clear_generation_outputs() {
  local pred_file="$1"
  rm -f "${pred_file}" "${pred_file%.jsonl}".gpu*.jsonl
}

clear_task_outputs() {
  local pred_file="$1"
  local summary_file="$2"
  clear_generation_outputs "${pred_file}"
  rm -f "${summary_file}"
}

clear_merged_prediction_output() {
  local pred_file="$1"
  rm -f "${pred_file}"
}

task_completed() {
  local pred_file="$1"
  local summary_file="$2"
  local data_file="$3"
  local task_kind="$4"
  if ! json_file_is_valid "${summary_file}"; then
    return 1
  fi
  local status=0
  prediction_status "${pred_file}" "${data_file}" "${task_kind}" || status=$?
  [[ "${status}" -eq 0 ]]
}

task_can_eval_only() {
  local pred_file="$1"
  local summary_file="$2"
  local data_file="$3"
  local task_kind="$4"
  if json_file_is_valid "${summary_file}"; then
    return 1
  fi
  local status=0
  prediction_status "${pred_file}" "${data_file}" "${task_kind}" || status=$?
  if [[ "${status}" -eq 0 ]]; then
    return 0
  fi
  if [[ "${SHARD_WORLD_SIZE}" -gt 1 ]] && has_shard_files "${pred_file%.jsonl}.gpu*.jsonl"; then
    return 1
  fi
  return 1
}

task_can_resume_generate() {
  local pred_file="$1"
  local summary_file="$2"
  local data_file="$3"
  local task_kind="$4"
  if json_file_is_valid "${summary_file}"; then
    return 1
  fi
  if [[ "${SHARD_WORLD_SIZE}" -gt 1 ]]; then
    local any_shard=0
    local rank
    for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
      local shard_file="${pred_file%.jsonl}.gpu${rank}.jsonl"
      if [[ ! -f "${shard_file}" ]]; then
        continue
      fi
      any_shard=1
      local shard_status=0
      shard_prediction_status "${shard_file}" "${data_file}" "${task_kind}" "${rank}" || shard_status=$?
      if [[ "${shard_status}" -ne 0 && "${shard_status}" -ne 3 ]]; then
        return 1
      fi
    done
    [[ "${any_shard}" -eq 1 ]]
    return $?
  fi
  local status=0
  prediction_status "${pred_file}" "${data_file}" "${task_kind}" || status=$?
  [[ "${status}" -eq 3 ]]
}

prediction_status() {
  local pred_file="$1"
  local data_file="$2"
  local status=0
  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/utils/check_geochat_prediction_integrity.py \
    --preds "${pred_file}" \
    --data "${data_file}" > /dev/null 2>&1 || status=$?
  return "${status}"
}

shard_prediction_status() {
  local shard_file="$1"
  local data_file="$2"
  local task_kind="$3"
  local rank="$4"
  local status=0
  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/utils/check_geochat_prediction_integrity.py \
    --preds "${shard_file}" \
    --data "${data_file}" \
    --shard-world-size "${SHARD_WORLD_SIZE}" \
    --shard-rank "${rank}" \
    --shard-weights "${SHARD_WEIGHTS}" > /dev/null 2>&1 || status=$?
  return "${status}"
}

maybe_reset_invalid_predictions() {
  local name="$1"
  local pred_file="$2"
  local data_file="$3"
  local task_kind="$4"
  if [[ "${SHARD_WORLD_SIZE}" -gt 1 ]] && has_shard_files "${pred_file%.jsonl}.gpu*.jsonl"; then
    local rank
    for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
      local shard_file="${pred_file%.jsonl}.gpu${rank}.jsonl"
      if [[ ! -f "${shard_file}" ]]; then
        continue
      fi
      local shard_status=0
      shard_prediction_status "${shard_file}" "${data_file}" "${task_kind}" "${rank}" || shard_status=$?
      if [[ "${shard_status}" -ne 0 && "${shard_status}" -ne 3 ]]; then
        log "[RESET] $(date '+%F %T') ${name} found invalid shard predictions; clear and regenerate."
        clear_generation_outputs "${pred_file}"
        return 0
      fi
    done
    local merged_status=0
    prediction_status "${pred_file}" "${data_file}" "${task_kind}" || merged_status=$?
    if [[ "${merged_status}" -ne 0 && "${merged_status}" -ne 3 ]] && [[ -f "${pred_file}" ]]; then
      log "[RESET] $(date '+%F %T') ${name} found invalid merged predictions; remove merged file and keep valid shard files."
      clear_merged_prediction_output "${pred_file}"
    fi
    return 0
  fi
  local status=0
  prediction_status "${pred_file}" "${data_file}" "${task_kind}" || status=$?
  if [[ "${status}" -ne 0 && "${status}" -ne 3 ]]; then
    log "[RESET] $(date '+%F %T') ${name} found unusable predictions (integrity status=${status}); clear and regenerate."
    clear_generation_outputs "${pred_file}"
  fi
  return 0
}

referring_file_is_current() {
  local pred_file="$1"
  [[ -f "${pred_file}" ]] || return 1
  "${PY_BIN}" - "${pred_file}" <<'PY' > /dev/null 2>&1
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
allowed = {
    "bbox2d1000_qwen_native_style_auto_v3",
}

with path.open("r", encoding="utf-8") as f:
    for line in f:
        raw = line.strip()
        if not raw:
            continue
        row = json.loads(raw)
        if str(row.get("task", "")) != "referring":
            raise SystemExit(4)
        if str(row.get("prompt_version", "")) not in allowed:
            raise SystemExit(4)
raise SystemExit(0)
PY
}

maybe_reset_stale_referring_predictions() {
  local pred_file="$1"
  local summary_file="$2"
  if [[ -f "${pred_file}" ]] && ! referring_file_is_current "${pred_file}"; then
    log "[RESET] $(date '+%F %T') referring found stale predictions; clear and regenerate."
    clear_task_outputs "${pred_file}" "${summary_file}"
    return 0
  fi
  if [[ "${SHARD_WORLD_SIZE}" -gt 1 ]] && has_shard_files "${pred_file%.jsonl}.gpu*.jsonl"; then
    local rank
    for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
      local shard_file="${pred_file%.jsonl}.gpu${rank}.jsonl"
      if [[ ! -f "${shard_file}" ]]; then
        continue
      fi
      if ! referring_file_is_current "${shard_file}"; then
        log "[RESET] $(date '+%F %T') referring found stale shard predictions; clear and regenerate."
        clear_task_outputs "${pred_file}" "${summary_file}"
        return 0
      fi
    done
  fi
  return 0
}

vqa_overlap_status() {
  local data_file="$1"
  local gt_file="$2"
  local status=0
  "${PY_BIN}" - "${data_file}" "${gt_file}" <<'PY'
import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
gt_path = Path(sys.argv[2])
data_qids = set()
with data_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "question_id" in row:
            data_qids.add(str(row["question_id"]))

if gt_path.suffix.lower() == ".jsonl":
    gt_rows = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gt_rows.append(json.loads(line))
else:
    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
    if isinstance(gt_data, dict) and isinstance(gt_data.get("answers"), list):
        gt_rows = gt_data["answers"]
    else:
        gt_rows = gt_data if isinstance(gt_data, list) else []

gt_qids = {
    str(row["question_id"])
    for row in gt_rows
    if isinstance(row, dict)
    and "question_id" in row
    and ("active" not in row or row.get("active") is True)
}

overlap = len(data_qids & gt_qids)
print(f"[INFO] {data_path.name}: data_qids={len(data_qids)} gt_qids={len(gt_qids)} overlap={overlap}")
sys.exit(0 if overlap > 0 else 3)
PY
  status=$?
  return "${status}"
}

run_sharded_generate() {
  local name="$1"
  local key_name="$2"
  local script_path="$3"
  local pred_file="$4"
  shift 4

  local merge_log="${LOG_DIR}/${name}.merge.log"
  local status=0
  local shard_files=()
  local pids=()

  if is_truthy "${FORCE_RERUN}"; then
    clear_generation_outputs "${pred_file}"
  fi
  log "[START] $(date '+%F %T') ${name}"

  if [[ "${SHARD_WORLD_SIZE}" -le 1 ]]; then
    local log_file="${LOG_DIR}/${name}.gen.log"
    local gpu_id="${GPU_ID_ARR[0]:-}"
    if [[ -z "${gpu_id}" ]]; then
      echo "[ERROR] SHARD_GPU_IDS must provide at least one GPU id for single-card inference." >&2
      return 2
    fi
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY_BIN}" "${script_path}" \
      --output "${pred_file}" \
      --batch-size "${BATCH_SIZE}" \
      --device-map "${DEVICE_MAP}" \
      --dtype "${DTYPE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      "$@" > "${log_file}" 2>&1
    log "[END]   $(date '+%F %T') ${name}"
    return 0
  fi

  if [[ "${#GPU_ID_ARR[@]}" -lt "${SHARD_WORLD_SIZE}" ]]; then
    echo "[ERROR] SHARD_GPU_IDS (${SHARD_GPU_IDS}) has fewer GPUs than SHARD_WORLD_SIZE (${SHARD_WORLD_SIZE})" >&2
    return 2
  fi

  local rank
  for ((rank = 0; rank < SHARD_WORLD_SIZE; rank++)); do
    local gpu_id="${GPU_ID_ARR[rank]}"
    local shard_file="${pred_file%.jsonl}.gpu${rank}.jsonl"
    local shard_log="${LOG_DIR}/${name}.gpu${rank}.gen.log"
    shard_files+=("${shard_file}")

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY_BIN}" "${script_path}" \
      --output "${shard_file}" \
      --batch-size "${BATCH_SIZE}" \
      --device-map "${DEVICE_MAP}" \
      --dtype "${DTYPE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --shard-world-size "${SHARD_WORLD_SIZE}" \
      --shard-rank "${rank}" \
      --shard-weights "${SHARD_WEIGHTS}" \
      "$@" > "${shard_log}" 2>&1 &
    pids+=("$!")
  done

  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=$?
  done
  if [[ "${status}" -ne 0 ]]; then
    log "[FAIL]  $(date '+%F %T') ${name} status=${status}"
    return "${status}"
  fi

  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/utils/merge_jsonl_shards.py \
    --inputs "${shard_files[@]}" \
    --output "${pred_file}" \
    --key "${key_name}" \
    --delete-inputs > "${merge_log}" 2>&1

  log "[END]   $(date '+%F %T') ${name}"
}

run_scene() {
  local name="$1"
  local data_file="$2"
  local image_root="$3"
  local pred_file="${OUT_DIR}/${name}.jsonl"
  local summary_file="${EVAL_DIR}/${name}_summary.json"
  local eval_log="${LOG_DIR}/${name}.eval.log"

  if [[ -z "${image_root}" || ! -d "${image_root}" ]]; then
    log "[WARN] Skip ${name}: image root missing -> ${image_root}"
    return 0
  fi

  if is_truthy "${FORCE_RERUN}"; then
    clear_task_outputs "${pred_file}" "${summary_file}"
  fi

  maybe_reset_invalid_predictions "${name}" "${pred_file}" "${data_file}" "scene"

  if task_completed "${pred_file}" "${summary_file}" "${data_file}" "scene" && ! is_truthy "${FORCE_RERUN}"; then
    log "[SKIP]  $(date '+%F %T') ${name} already has predictions and summary."
    return 0
  fi

  if task_can_eval_only "${pred_file}" "${summary_file}" "${data_file}" "scene" && ! is_truthy "${FORCE_RERUN}"; then
    log "[RESUME] $(date '+%F %T') ${name} reuse existing predictions and rerun eval only."
  elif task_can_resume_generate "${pred_file}" "${summary_file}" "${data_file}" "scene" && ! is_truthy "${FORCE_RERUN}"; then
    log "[RESUME] $(date '+%F %T') ${name} resume generation from partial predictions."
    run_sharded_generate "${name}" question_id benchmark/geochat/eval_scripts/qwen35/generate/generate_scene_qwen35.py "${pred_file}" \
      --model-dir "${MODEL_DIR}" \
      --data "${data_file}" \
      --image-root "${image_root}"
  else
    run_sharded_generate "${name}" question_id benchmark/geochat/eval_scripts/qwen35/generate/generate_scene_qwen35.py "${pred_file}" \
      --model-dir "${MODEL_DIR}" \
      --data "${data_file}" \
      --image-root "${image_root}"
  fi

  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/eval/eval_geochat_scene.py \
    --preds "${pred_file}" \
    --data "${data_file}" \
    --output "${summary_file}" > "${eval_log}" 2>&1
}

run_vqa() {
  local name="$1"
  local data_file="$2"
  local image_root="$3"
  local gt_file="$4"
  local exclude_categories="$5"
  local pred_file="${OUT_DIR}/${name}.jsonl"
  local summary_file="${EVAL_DIR}/${name}_summary.json"
  local eval_log="${LOG_DIR}/${name}.eval.log"

  if [[ -z "${image_root}" || ! -d "${image_root}" ]]; then
    log "[WARN] Skip ${name}: image root missing -> ${image_root}"
    return 0
  fi
  if [[ -z "${gt_file}" || ! -f "${gt_file}" ]]; then
    log "[WARN] Skip ${name}: ground-truth file missing -> ${gt_file}"
    return 0
  fi
  local overlap_status=0
  vqa_overlap_status "${data_file}" "${gt_file}" || overlap_status=$?
  if [[ "${overlap_status}" -eq 3 ]]; then
    log "[WARN] Skip ${name}: no overlapping question_id between ${data_file} and ${gt_file}"
    return 0
  elif [[ "${overlap_status}" -ne 0 ]]; then
    echo "[ERROR] Failed to inspect VQA GT overlap for ${name}: status=${overlap_status}" >&2
    return "${overlap_status}"
  fi

  if is_truthy "${FORCE_RERUN}"; then
    clear_task_outputs "${pred_file}" "${summary_file}"
  fi

  maybe_reset_invalid_predictions "${name}" "${pred_file}" "${data_file}" "vqa"

  if task_completed "${pred_file}" "${summary_file}" "${data_file}" "vqa" && ! is_truthy "${FORCE_RERUN}"; then
    log "[SKIP]  $(date '+%F %T') ${name} already has predictions and summary."
    return 0
  fi

  if task_can_eval_only "${pred_file}" "${summary_file}" "${data_file}" "vqa" && ! is_truthy "${FORCE_RERUN}"; then
    log "[RESUME] $(date '+%F %T') ${name} reuse existing predictions and rerun eval only."
  elif task_can_resume_generate "${pred_file}" "${summary_file}" "${data_file}" "vqa" && ! is_truthy "${FORCE_RERUN}"; then
    log "[RESUME] $(date '+%F %T') ${name} resume generation from partial predictions."
    run_sharded_generate "${name}" question_id benchmark/geochat/eval_scripts/qwen35/generate/generate_vqa_qwen35.py "${pred_file}" \
      --model-dir "${MODEL_DIR}" \
      --data "${data_file}" \
      --image-root "${image_root}"
  else
    run_sharded_generate "${name}" question_id benchmark/geochat/eval_scripts/qwen35/generate/generate_vqa_qwen35.py "${pred_file}" \
      --model-dir "${MODEL_DIR}" \
      --data "${data_file}" \
      --image-root "${image_root}"
  fi

  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/eval/eval_geochat_vqa.py \
    --preds "${pred_file}" \
    --data "${data_file}" \
    --ground-truth-file "${gt_file}" \
    --exclude-categories "${exclude_categories}" \
    --output "${summary_file}" > "${eval_log}" 2>&1
}

prepare_hrben_gt() {
  if [[ -n "${HRBEN_GT_FILE}" && -f "${HRBEN_GT_FILE}" ]] && ! is_truthy "${FORCE_RERUN}"; then
    log "[SKIP]  $(date '+%F %T') prepare_hrben_gt reuse existing file -> ${HRBEN_GT_FILE}"
    return 0
  fi
  if [[ -z "${HRBEN_GT_FILE}" ]]; then
    log "[WARN] Skip HRBEN GT build: output file path missing -> ${HRBEN_GT_FILE}"
    return 0
  fi
  if [[ -z "${HRBEN_RAW_QUESTIONS_FILE}" || ! -f "${HRBEN_RAW_QUESTIONS_FILE}" ]]; then
    log "[WARN] Skip HRBEN GT build: questions file missing -> ${HRBEN_RAW_QUESTIONS_FILE}"
    return 0
  fi
  if [[ -z "${HRBEN_RAW_ANSWERS_FILE}" || ! -f "${HRBEN_RAW_ANSWERS_FILE}" ]]; then
    log "[WARN] Skip HRBEN GT build: answers file missing -> ${HRBEN_RAW_ANSWERS_FILE}"
    return 0
  fi
  if [[ -z "${HRBEN_RAW_IMAGES_FILE}" || ! -f "${HRBEN_RAW_IMAGES_FILE}" ]]; then
    log "[WARN] Skip HRBEN GT build: images file missing -> ${HRBEN_RAW_IMAGES_FILE}"
    return 0
  fi

  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/prepare/prepare_geochat_hrben_gt.py \
    --bench-data "${HRBEN_DATA}" \
    --raw-questions "${HRBEN_RAW_QUESTIONS_FILE}" \
    --raw-answers "${HRBEN_RAW_ANSWERS_FILE}" \
    --raw-images "${HRBEN_RAW_IMAGES_FILE}" \
    --output "${HRBEN_GT_FILE}"
}

run_region_caption_fix() {
  local pred_file="$1"
  local fix_log="${LOG_DIR}/region_caption.fix.log"
  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/fix/fix_geochat_max_new_tokens.py \
    --preds "${pred_file}" \
    --model-family qwen35 \
    --model-dir "${MODEL_DIR}" \
    --device-map "${DEVICE_MAP}" \
    --dtype "${DTYPE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" > "${fix_log}" 2>&1
}

run_region_caption() {
  local name="region_caption"
  local pred_file="${OUT_DIR}/${name}.jsonl"
  local summary_file="${EVAL_DIR}/${name}_summary.json"
  local eval_log="${LOG_DIR}/${name}.eval.log"

  if [[ -z "${REGION_IMAGE_ROOT}" || ! -d "${REGION_IMAGE_ROOT}" ]]; then
    log "[WARN] Skip ${name}: image root missing -> ${REGION_IMAGE_ROOT}"
    return 0
  fi

  if is_truthy "${FORCE_RERUN}"; then
    clear_task_outputs "${pred_file}" "${summary_file}"
  fi

  maybe_reset_invalid_predictions "${name}" "${pred_file}" "${REGION_CAPTION_DATA}" "region_caption"

  if task_completed "${pred_file}" "${summary_file}" "${REGION_CAPTION_DATA}" "region_caption" && ! is_truthy "${FORCE_RERUN}"; then
    log "[SKIP]  $(date '+%F %T') ${name} already has predictions and summary."
    return 0
  fi

  if task_can_eval_only "${pred_file}" "${summary_file}" "${REGION_CAPTION_DATA}" "region_caption" && ! is_truthy "${FORCE_RERUN}"; then
    log "[RESUME] $(date '+%F %T') ${name} reuse existing predictions and rerun eval only."
  elif task_can_resume_generate "${pred_file}" "${summary_file}" "${REGION_CAPTION_DATA}" "region_caption" && ! is_truthy "${FORCE_RERUN}"; then
    log "[RESUME] $(date '+%F %T') ${name} resume generation from partial predictions."
    run_sharded_generate "${name}" question_id benchmark/geochat/eval_scripts/qwen35/generate/generate_region_caption_qwen35.py "${pred_file}" \
      --model-dir "${MODEL_DIR}" \
      --data "${REGION_CAPTION_DATA}" \
      --image-root "${REGION_IMAGE_ROOT}"
  else
    run_sharded_generate "${name}" question_id benchmark/geochat/eval_scripts/qwen35/generate/generate_region_caption_qwen35.py "${pred_file}" \
      --model-dir "${MODEL_DIR}" \
      --data "${REGION_CAPTION_DATA}" \
      --image-root "${REGION_IMAGE_ROOT}"
  fi

  run_region_caption_fix "${pred_file}"

  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/eval/eval_geochat_region_caption.py \
    --preds "${pred_file}" \
    --data "${REGION_CAPTION_DATA}" \
    --output "${summary_file}" > "${eval_log}" 2>&1
}

run_referring() {
  local name="referring"
  local pred_file="${OUT_DIR}/${name}.jsonl"
  local summary_file="${EVAL_DIR}/${name}_summary.json"
  local eval_log="${LOG_DIR}/${name}.eval.log"

  if [[ -z "${REFERRING_IMAGE_ROOT}" || ! -d "${REFERRING_IMAGE_ROOT}" ]]; then
    log "[WARN] Skip ${name}: image root missing -> ${REFERRING_IMAGE_ROOT}"
    return 0
  fi

  if is_truthy "${FORCE_RERUN}"; then
    clear_task_outputs "${pred_file}" "${summary_file}"
  fi

  maybe_reset_stale_referring_predictions "${pred_file}" "${summary_file}"
  maybe_reset_invalid_predictions "${name}" "${pred_file}" "${REFERRING_DATA}" "referring"

  if task_completed "${pred_file}" "${summary_file}" "${REFERRING_DATA}" "referring" && ! is_truthy "${FORCE_RERUN}"; then
    log "[SKIP]  $(date '+%F %T') ${name} already has predictions and summary."
    return 0
  fi

  if task_can_eval_only "${pred_file}" "${summary_file}" "${REFERRING_DATA}" "referring" && ! is_truthy "${FORCE_RERUN}"; then
    log "[RESUME] $(date '+%F %T') ${name} reuse existing predictions and rerun eval only."
  elif task_can_resume_generate "${pred_file}" "${summary_file}" "${REFERRING_DATA}" "referring" && ! is_truthy "${FORCE_RERUN}"; then
    log "[RESUME] $(date '+%F %T') ${name} resume generation from partial predictions."
    run_sharded_generate "${name}" question_id benchmark/geochat/eval_scripts/qwen35/generate/generate_referring_qwen35.py "${pred_file}" \
      --model-dir "${MODEL_DIR}" \
      --data "${REFERRING_DATA}" \
      --image-root "${REFERRING_IMAGE_ROOT}"
  else
    run_sharded_generate "${name}" question_id benchmark/geochat/eval_scripts/qwen35/generate/generate_referring_qwen35.py "${pred_file}" \
      --model-dir "${MODEL_DIR}" \
      --data "${REFERRING_DATA}" \
      --image-root "${REFERRING_IMAGE_ROOT}"
  fi

  "${PY_BIN}" benchmark/geochat/eval_scripts/shared/eval/eval_geochat_referring.py \
    --preds "${pred_file}" \
    --data "${REFERRING_DATA}" \
    --image-root "${REFERRING_IMAGE_ROOT}" \
    --output "${summary_file}" > "${eval_log}" 2>&1
}

run_scene "scene_aid" "${AID_DATA}" "${AID_IMAGE_ROOT}"
run_scene "scene_ucmerced" "${UCMERCED_DATA}" "${UCMERCED_IMAGE_ROOT}"
prepare_hrben_gt
run_vqa "vqa_hrben" "${HRBEN_DATA}" "${HRBEN_IMAGE_ROOT}" "${HRBEN_GT_FILE}" ""
run_vqa "vqa_lrben" "${LRBEN_DATA}" "${LRBEN_IMAGE_ROOT}" "${LRBEN_GT_FILE}" "count"
run_region_caption
run_referring

echo "[OK] Finished GeoChat baseline eval for Qwen3.5."
