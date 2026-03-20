#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${ROOT_DIR}"

source /opt/yanzixi/home/yzx/miniconda3/etc/profile.d/conda.sh
conda activate qwen3-dinov3.5
source benchmark/geochat/data/local_paths.sh

export PY_BIN="${PY_BIN:-python}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
export SHARD_WORLD_SIZE="${SHARD_WORLD_SIZE:-2}"
export SHARD_GPU_IDS="${SHARD_GPU_IDS:-0,1}"
export SHARD_WEIGHTS="${SHARD_WEIGHTS:-1:1}"
export FORCE_RERUN="${FORCE_RERUN:-0}"

bash benchmark/geochat/eval_scripts/qwen35/run/run_eval_qwen35_baseline.sh
