#!/usr/bin/env bash
# One-shot safe handoff: preserve the completed target-profile pair, stop only
# the just-started tail child, and restart the updated queue with cls43hold next.
set -uo pipefail

ROOT=/home/ubuntu/ecup
LOGS=$ROOT/logs
OLD_SUPERVISOR=43500

while ! grep -q '^DONE cls300tprof_val$' "$LOGS/morning_priority.log"; do
  sleep 2
done

best_score=$(
  "$ROOT/.venv/bin/python" - "$ROOT/work/cls300tprof_val_history.json" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
print(min(row['score'] for row in rows))
PY
)
if "$ROOT/.venv/bin/python" - "$best_score" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 1.72 else 1)
PY
then
  echo "TARGET_PROFILE_REJECTED validation=$best_score"
  touch "$LOGS/skip_cls300tprof"
  while ! grep -q '^START cls300tprof_full ' "$LOGS/morning_priority.log"; do
    sleep 1
  done
else
  echo "TARGET_PROFILE_ACCEPTED validation=$best_score"
  while ! grep -q '^DONE cls300tprof_full$' "$LOGS/morning_priority.log"; do
    sleep 2
  done
fi

date '+%Y-%m-%d %H:%M:%S %Z'
echo TARGET_PROFILE_COMMITTED
children=$(ps -o pid= --ppid "$OLD_SUPERVISOR" 2>/dev/null || true)
if [[ -n $children ]]; then
  kill $children 2>/dev/null || true
fi
kill "$OLD_SUPERVISOR" 2>/dev/null || true
while kill -0 "$OLD_SUPERVISOR" 2>/dev/null; do sleep 1; done

mv "$LOGS/morning_priority.log" \
  "$LOGS/morning_priority_before_season_holdout_20260825.log"
cd "$ROOT"
nohup bash work/server_morning_priority.sh >"$LOGS/morning_priority.log" 2>&1 </dev/null &
echo NEW_SUPERVISOR=$!
date '+%Y-%m-%d %H:%M:%S %Z'
echo SEASON_HOLDOUT_HANDOFF_DONE
