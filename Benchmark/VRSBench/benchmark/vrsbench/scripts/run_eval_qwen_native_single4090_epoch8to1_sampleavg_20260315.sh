#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

for spec in \
  "8 15" \
  "7 16" \
  "6 17" \
  "5 18" \
  "4 19" \
  "3 20" \
  "2 21" \
  "1 22"; do
  set -- $spec
  epoch="$1"
  run_id="$2"
  echo "[BATCH] $(date '+%F %T') start epoch=${epoch} run_id=${run_id}"
  ./scripts/run_eval_qwen_native_single4090_epoch_sampleavg_20260315.sh "$epoch" "$run_id"
  echo "[BATCH] $(date '+%F %T') done epoch=${epoch} run_id=${run_id}"
done
