#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
REG="$ROOT/work/train_classifier_gpu_regularity_frozen.py"
MARK="$ROOT/work/train_classifier_gpu_marked_frozen.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

marked_pid=$(cat "$ROOT/work/server_marked_event_growth.pid")
echo "WAIT marked-event supervisor pid=$marked_pid $(date --iso-8601=seconds)"
while kill -0 "$marked_pid" 2>/dev/null; do
  sleep 30
done
if [ ! -s "$ROOT/work/mark409_growth_report.json" ]; then
  echo "marked-event handoff ended without report" >&2
  exit 1
fi
for file in "$REG" "$MARK"; do
  if [ ! -s "$file" ]; then
    echo "missing frozen trainer: $file" >&2
    exit 1
  fi
done

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed 2718
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --event-profile --calendar --private-selection --anchor-start 43
)

run_pair() {
  local trainer=$1
  local stem=$2
  echo "START $stem selection fold342 $(date --iso-8601=seconds)"
  "$PY" "$trainer" "${stem}_select342" "${COMMON[@]}" --val 342 --epochs 3 \
    > "$LOGS/${stem}_select342.log" 2>&1
  echo "START $stem untouched fold378 $(date --iso-8601=seconds)"
  "$PY" "$trainer" "${stem}_holdout378" "${COMMON[@]}" --val 378 --epochs 3 \
    > "$LOGS/${stem}_holdout378.log" 2>&1
  echo "DONE $stem $(date --iso-8601=seconds)"
}

# The seed-1310 regularity result is a decisive rejection (worse untouched
# holdout and negative ridge weight in every resample), so repeating it would
# no longer be informative.  The marked-event branch remains frozen.
run_pair "$MARK" mark409s2718

echo "FIVE_HOUR_TAIL_DONE $(date --iso-8601=seconds)"
