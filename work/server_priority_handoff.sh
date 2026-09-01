#!/usr/bin/env bash
# One-shot handoff: let the valuable market pair finish, then replace the old
# capacity-first tail with the priority-ordered morning queue.
set -euo pipefail

ROOT=/home/ubuntu/ecup
LOGS=$ROOT/logs
cd "$ROOT"
date '+HANDOFF_WAIT %Y-%m-%d %H:%M:%S %Z'
while ! grep -q '^DONE cls300mkt_full$' "$LOGS/extra_supervisor.log"; do sleep 15; done
date '+MARKET_COMMITTED %Y-%m-%d %H:%M:%S %Z'

# Both old supervisors are only queue owners at this point. Stop them before
# the primary can spend the morning on the lower-priority wide branch.
pkill -TERM -f '[s]erver_followup_run.sh' || true
pkill -TERM -f '[s]erver_extra_run.sh' || true
sleep 3
# Cover the small race in which wide training was spawned between the DONE log
# line and termination of its parent.
pkill -TERM -f '[t]rain_classifier_gpu.py cls409wide' || true
sleep 2

nohup bash work/server_morning_priority.sh \
  >"$LOGS/morning_priority.log" 2>&1 </dev/null &
echo $! >"$LOGS/morning_priority.pid"
date '+HANDOFF_DONE %Y-%m-%d %H:%M:%S %Z'
cat "$LOGS/morning_priority.pid"
