#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_classifier_gpu_event_frozen.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

tail_pid=$(cat "$ROOT/work/server_five_hour_tail.pid")
echo "WAIT seed-2718 profile tail pid=$tail_pid $(date --iso-8601=seconds)"
while kill -0 "$tail_pid" 2>/dev/null; do
  sleep 30
done
if [ ! -s "$ROOT/work/mark409s2718_holdout378_history.json" ]; then
  echo "seed-2718 profile tail ended without final history" >&2
  exit 1
fi
if [ ! -s "$TRAIN" ]; then
  echo "missing frozen control trainer" >&2
  exit 1
fi

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed 2718
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --calendar --private-selection --anchor-start 43
)

echo "START control409s2718 selection fold342 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" control409s2718_select342 "${COMMON[@]}" --val 342 --epochs 3 \
  > "$LOGS/control409s2718_select342.log" 2>&1
echo "START control409s2718 untouched fold378 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" control409s2718_holdout378 "${COMMON[@]}" --val 378 --epochs 3 \
  > "$LOGS/control409s2718_holdout378.log" 2>&1
echo "FIVE_HOUR_CONTROL_TAIL_DONE $(date --iso-8601=seconds)"
