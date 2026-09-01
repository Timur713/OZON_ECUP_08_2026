#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_classifier_gpu.py"
STAGED="$ROOT/work/train_classifier_gpu_event_staged.py"
BUILD="$ROOT/work/build_event_summary_holdout.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

survival_pid=$(cat "$ROOT/work/server_survival_growth.pid")
echo "WAIT survival supervisor pid=$survival_pid $(date --iso-8601=seconds)"
while kill -0 "$survival_pid" 2>/dev/null; do
  sleep 30
done

if [ ! -s "$ROOT/work/surv409_growth_report.json" ]; then
  echo "survival handoff ended without report" >&2
  exit 1
fi
if [ ! -s "$STAGED" ]; then
  echo "staged event-summary trainer is missing" >&2
  exit 1
fi
cp "$STAGED" "$TRAIN"

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed 1310
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --event-summary --calendar --private-selection --anchor-start 43
)

echo "START event-summary selection fold 342 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" event409_select342 "${COMMON[@]}" --val 342 --epochs 3 \
  > "$LOGS/event409_select342.log" 2>&1

selected_epoch=$(
  "$PY" - "$ROOT/work/event409_select342_history.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
if len(rows) != 3:
    raise SystemExit(f"expected 3 selection epochs, received {len(rows)}")
print(min(rows, key=lambda row: row["score"])["epoch"])
PY
)
echo "SELECTED epoch=$selected_epoch from fold 342"

echo "START untouched event-summary fold 378 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" event409_holdout378 "${COMMON[@]}" --val 378 --epochs 3 \
  > "$LOGS/event409_holdout378.log" 2>&1

"$PY" "$BUILD" > "$LOGS/event409_growth_build.log" 2>&1
cat "$LOGS/event409_growth_build.log"
echo "EVENT_SUMMARY_GROWTH_DONE $(date --iso-8601=seconds)"
