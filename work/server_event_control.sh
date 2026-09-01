#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_classifier_gpu.py"
FROZEN="$ROOT/work/train_classifier_gpu_event_frozen.py"
BUILD="$ROOT/work/build_event_control_holdout.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

profile_pid=$(cat "$ROOT/work/server_event_profile_growth.pid")
echo "WAIT event-profile supervisor pid=$profile_pid $(date --iso-8601=seconds)"
while kill -0 "$profile_pid" 2>/dev/null; do
  sleep 30
done
if [ ! -s "$ROOT/work/reg409_growth_report.json" ]; then
  echo "event-profile handoff ended without report" >&2
  exit 1
fi
if [ ! -s "$FROZEN" ]; then
  echo "frozen matched-control trainer is missing" >&2
  exit 1
fi
cp "$FROZEN" "$TRAIN"

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed 1310
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --calendar --private-selection --anchor-start 43
)

echo "START matched-control selection fold 342 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" control409_select342 "${COMMON[@]}" --val 342 --epochs 3 \
  > "$LOGS/control409_select342.log" 2>&1

selected_epoch=$(
  "$PY" - "$ROOT/work/control409_select342_history.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
if len(rows) != 3:
    raise SystemExit(f"expected 3 selection epochs, received {len(rows)}")
print(min(rows, key=lambda row: row["score"])["epoch"])
PY
)
echo "SELECTED epoch=$selected_epoch from fold 342"

echo "START untouched matched-control fold 378 $(date --iso-8601=seconds)"
"$PY" "$TRAIN" control409_holdout378 "${COMMON[@]}" --val 378 --epochs 3 \
  > "$LOGS/control409_holdout378.log" 2>&1

"$PY" "$BUILD" > "$LOGS/control409_growth_build.log" 2>&1
cat "$LOGS/control409_growth_build.log"
echo "EVENT_CONTROL_DONE $(date --iso-8601=seconds)"
