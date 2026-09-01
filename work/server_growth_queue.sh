#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
BASE_TAG=multi_anchor_growth
BASE_PID_FILE="$ROOT/work/multi_anchor_growth.pid"
LOG="$ROOT/logs/residual_growth.log"

cd "$ROOT"
base_pid=$(cat "$BASE_PID_FILE")
echo "waiting for multi-anchor pid=$base_pid at $(date --iso-8601=seconds)" > "$LOG"
while kill -0 "$base_pid" 2>/dev/null; do
  sleep 30
done

for required in \
  "$ROOT/work/${BASE_TAG}_report.json" \
  "$ROOT/work/${BASE_TAG}_oof.npz" \
  "$ROOT/work/${BASE_TAG}_final.npy"
do
  if [[ ! -s "$required" ]]; then
    echo "missing required artifact: $required" >> "$LOG"
    exit 1
  fi
done

echo "starting residual growth at $(date --iso-8601=seconds)" >> "$LOG"
exec env \
  ECUP_ROOT="$ROOT" \
  ECUP_MAT="$ROOT/work/mat" \
  ECUP_OUT="$ROOT/work" \
  ECUP_LGB_THREADS=20 \
  ECUP_BASE_TAG="$BASE_TAG" \
  ECUP_TAG=residual_growth \
  "$ROOT/.venv/bin/python" "$ROOT/work/train_residual_growth.py" >> "$LOG" 2>&1
