#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
cd "$ROOT"
combined_pid=$(cat work/server_w409_exact_position_decay.pid)
echo "WAIT position-decay pid=$combined_pid $(date --iso-8601=seconds)"
while kill -0 "$combined_pid" 2>/dev/null; do sleep 30; done
if [ ! -s work/w409_exact_position_decay_decision.json ]; then
  echo "position-decay tail ended without decision" >&2
  exit 1
fi
echo "RESUME long position tail $(date --iso-8601=seconds)"
nohup bash work/server_position_tail.sh > logs/server_position_tail_after_combined.log 2>&1 &
position_pid=$!
echo "$position_pid" > work/server_position_tail.pid
wait "$position_pid"
echo "START long hidden-decay tail $(date --iso-8601=seconds)"
nohup bash work/server_hidden_decay_tail.sh > logs/server_hidden_decay_after_combined.log 2>&1 &
hidden_pid=$!
echo "$hidden_pid" > work/server_hidden_decay_tail.pid
wait "$hidden_pid"
echo "LONG_TAIL_DONE $(date --iso-8601=seconds)"
