#!/usr/bin/env bash
# One-shot handoff after the first exact-season pair is safely committed.
set -uo pipefail

ROOT=/home/ubuntu/ecup
LOGS=$ROOT/logs
OLD_SUPERVISOR=49534

while ! grep -q '^DONE cls43hold_full$' "$LOGS/morning_priority.log"; do
  sleep 2
done

date '+%Y-%m-%d %H:%M:%S %Z'
echo CLS43HOLD_COMMITTED
children=$(ps -o pid= --ppid "$OLD_SUPERVISOR" 2>/dev/null || true)
if [[ -n $children ]]; then
  kill $children 2>/dev/null || true
fi
kill "$OLD_SUPERVISOR" 2>/dev/null || true
while kill -0 "$OLD_SUPERVISOR" 2>/dev/null; do sleep 1; done

mv "$LOGS/morning_priority.log" \
  "$LOGS/morning_priority_before_exact_holdout_20260825.log"
cd "$ROOT"
nohup bash work/server_morning_priority.sh >"$LOGS/morning_priority.log" 2>&1 </dev/null &
echo NEW_SUPERVISOR=$!
date '+%Y-%m-%d %H:%M:%S %Z'
echo EXACT_HOLDOUT_HANDOFF_DONE
