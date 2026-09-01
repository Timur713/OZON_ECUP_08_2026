#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
BUILD="$ROOT/work/build_frozen_profile_report.py"
AUDIT="$ROOT/work/evaluate_validation_ridge.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"

control_tail_pid=$(cat "$ROOT/work/server_five_hour_control_tail.pid")
echo "WAIT seed-2718 control tail pid=$control_tail_pid $(date --iso-8601=seconds)"
while kill -0 "$control_tail_pid" 2>/dev/null; do
  sleep 30
done
if [ ! -s "$ROOT/work/control409s2718_holdout378_history.json" ]; then
  echo "control tail ended without final history" >&2
  exit 1
fi

"$PY" "$BUILD" --select-tag mark409s2718_select342 \
  --holdout-tag mark409s2718_holdout378 --output-tag mark409s2718_growth \
  --seed 2718 --event-profile --role marked_event_seed_replication \
  > "$LOGS/mark409s2718_build.log" 2>&1
"$PY" "$BUILD" --select-tag control409s2718_select342 \
  --holdout-tag control409s2718_holdout378 --output-tag control409s2718_growth \
  --seed 2718 --role matched_control_seed_replication \
  > "$LOGS/control409s2718_build.log" 2>&1

for tag in mark409s2718_growth control409s2718_growth; do
  "$PY" "$AUDIT" "$ROOT/work/${tag}_val.npy" --repeats 96 \
    > "$ROOT/work/${tag}_ridge96.json"
  "$PY" "$AUDIT" "$ROOT/work/w409c_val.npy" "$ROOT/work/${tag}_val.npy" \
    --joint --repeats 96 > "$ROOT/work/${tag}_w409c_joint96.json"
done
"$PY" "$AUDIT" "$ROOT/work/control409s2718_growth_val.npy" \
  "$ROOT/work/mark409s2718_growth_val.npy" --joint --repeats 96 \
  > "$ROOT/work/mark409s2718_control_joint96.json"
echo "FIVE_HOUR_AUTONOMOUS_AUDIT_DONE $(date --iso-8601=seconds)"
