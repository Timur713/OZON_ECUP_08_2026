#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_classifier_gpu.py"
STAGED="$ROOT/work/train_classifier_gpu_direct_event_staged.py"
BUILD="$ROOT/work/build_direct_event_holdout.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

event_pid=$(cat "$ROOT/work/server_event_summary_growth.pid")
echo "WAIT event-summary supervisor pid=$event_pid $(date --iso-8601=seconds)"
while kill -0 "$event_pid" 2>/dev/null; do
  sleep 30
done
if [ ! -s "$ROOT/work/event409_growth_report.json" ]; then
  echo "event-summary handoff ended without report" >&2
  exit 1
fi
if [ ! -s "$STAGED" ]; then
  echo "staged direct-event trainer is missing" >&2
  exit 1
fi
cp "$STAGED" "$TRAIN"

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed 1320
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --event-summary --calendar --private-selection --anchor-start 43
  --require-30-target --head-selection direct --class-weight 0
  --magnitude-weight 0 --direct-weight 1 --mix 0
)

echo "START direct-event selection fold 342 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" directevent409_select342 "${COMMON[@]}" --val 342 --epochs 3 \
  > "$LOGS/directevent409_select342.log" 2>&1

selected_epoch=$(
  "$PY" - "$ROOT/work/directevent409_select342_history.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
if len(rows) != 3:
    raise SystemExit(f"expected 3 selection epochs, received {len(rows)}")
print(min(rows, key=lambda row: row["score"])["epoch"])
PY
)
echo "SELECTED epoch=$selected_epoch from fold 342"

echo "START untouched direct-event fold 378 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" directevent409_holdout378 "${COMMON[@]}" --val 378 --epochs 3 \
  > "$LOGS/directevent409_holdout378.log" 2>&1

"$PY" "$BUILD" > "$LOGS/directevent409_growth_build.log" 2>&1
cat "$LOGS/directevent409_growth_build.log"
echo "DIRECT_EVENT_GROWTH_DONE $(date --iso-8601=seconds)"
