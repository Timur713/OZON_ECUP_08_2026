#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
cd "$ROOT"
exact_pid=$(cat work/server_w409_exact_decay.pid)
echo "WAIT exact-w409c decay pid=$exact_pid $(date --iso-8601=seconds)"
while kill -0 "$exact_pid" 2>/dev/null; do sleep 30; done
if [ ! -s work/w409_exact_decay_decision.json ]; then
  echo "exact-w409c tail ended without a decision" >&2
  exit 1
fi

echo "RESUME position tail $(date --iso-8601=seconds)"
nohup bash work/server_position_tail.sh > logs/server_position_tail_resumed.log 2>&1 &
position_pid=$!
echo "$position_pid" > work/server_position_tail.pid
wait "$position_pid"

echo "START hidden-decay tail $(date --iso-8601=seconds)"
nohup bash work/server_hidden_decay_tail.sh > logs/server_hidden_decay_tail_resumed.log 2>&1 &
hidden_pid=$!
echo "$hidden_pid" > work/server_hidden_decay_tail.pid
wait "$hidden_pid"
echo "RESUMED_RESEARCH_DONE $(date --iso-8601=seconds)"
