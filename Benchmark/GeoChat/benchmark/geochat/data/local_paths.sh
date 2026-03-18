#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAW_DIR="${ROOT_DIR}/dataset/raw"

export AID_IMAGE_ROOT="${AID_IMAGE_ROOT:-${RAW_DIR}/AID/test}"
export UCMERCED_IMAGE_ROOT="${UCMERCED_IMAGE_ROOT:-${RAW_DIR}/UCMerced/UCMerced_LandUse/Images}"
export HRBEN_IMAGE_ROOT="${HRBEN_IMAGE_ROOT:-${RAW_DIR}/HRBEN/Data}"
export LRBEN_IMAGE_ROOT="${LRBEN_IMAGE_ROOT:-${RAW_DIR}/LRBEN/Images_LR}"
export GEOCHAT_IMAGE_ROOT="${GEOCHAT_IMAGE_ROOT:-${RAW_DIR}/GeoChat_Instruct/images/share/softwares/kartik/GeoChat_finetuning/final_images_llava}"

export HRBEN_RAW_QUESTIONS_FILE="${HRBEN_RAW_QUESTIONS_FILE:-${RAW_DIR}/HRBEN/USGS_split_test_phili_questions.json}"
export HRBEN_RAW_ANSWERS_FILE="${HRBEN_RAW_ANSWERS_FILE:-${RAW_DIR}/HRBEN/USGS_split_test_phili_answers.json}"
export HRBEN_RAW_IMAGES_FILE="${HRBEN_RAW_IMAGES_FILE:-${RAW_DIR}/HRBEN/USGS_split_test_phili_images.json}"
export HRBEN_GT_FILE="${HRBEN_GT_FILE:-${RAW_DIR}/HRBEN/geochat_hrben_test_phili_gt.jsonl}"
export LRBEN_GT_FILE="${LRBEN_GT_FILE:-${RAW_DIR}/LRBEN/LR_split_test_answers.json}"

export REGION_IMAGE_ROOT="${REGION_IMAGE_ROOT:-${GEOCHAT_IMAGE_ROOT}}"
export REFERRING_IMAGE_ROOT="${REFERRING_IMAGE_ROOT:-${GEOCHAT_IMAGE_ROOT}}"
