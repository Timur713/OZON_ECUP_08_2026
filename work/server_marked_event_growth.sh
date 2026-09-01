#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_classifier_gpu_marked_frozen.py"
DEPENDENCY="$ROOT/work/train_classifier_gpu_regularity_frozen.py"
BUILD="$ROOT/work/build_event_profile_holdout.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

control_pid=$(cat "$ROOT/work/server_event_control.pid")
echo "WAIT matched-control supervisor pid=$control_pid $(date --iso-8601=seconds)"
while kill -0 "$control_pid" 2>/dev/null; do
  sleep 30
done
if [ ! -s "$ROOT/work/control409_growth_report.json" ]; then
  echo "matched-control handoff ended without report" >&2
  exit 1
fi
for file in "$TRAIN" "$DEPENDENCY" "$BUILD"; do
  if [ ! -s "$file" ]; then
    echo "missing frozen marked-event input: $file" >&2
    exit 1
  fi
done

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed 1310
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --event-profile --calendar --private-selection --anchor-start 43
)

echo "START marked-event selection fold 342 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" mark409_select342 "${COMMON[@]}" --val 342 --epochs 3 \
  > "$LOGS/mark409_select342.log" 2>&1

selected_epoch=$(
  "$PY" - "$ROOT/work/mark409_select342_history.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
if len(rows) != 3:
    raise SystemExit(f"expected 3 selection epochs, received {len(rows)}")
print(min(rows, key=lambda row: row["score"])["epoch"])
PY
)
echo "SELECTED epoch=$selected_epoch from fold 342"

echo "START untouched marked-event fold 378 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" mark409_holdout378 "${COMMON[@]}" --val 378 --epochs 3 \
  > "$LOGS/mark409_holdout378.log" 2>&1

ECUP_SELECT_TAG=mark409_select342 \
ECUP_HOLDOUT_TAG=mark409_holdout378 \
ECUP_TAG=mark409_growth \
  "$PY" "$BUILD" > "$LOGS/mark409_growth_build.log" 2>&1
cat "$LOGS/mark409_growth_build.log"
echo "MARKED_EVENT_GROWTH_DONE $(date --iso-8601=seconds)"
