#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
REMOTE=/home/ubuntu/ecup
SSH=(ssh -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "$HOST")

while ! "${SSH[@]}" "test -s $REMOTE/work/server_morning_tail.pid"; do sleep 15; done
while "${SSH[@]}" "kill -0 \$(cat $REMOTE/work/server_morning_tail.pid) 2>/dev/null"; do
  "${SSH[@]}" "date; ps -eo args | grep '$REMOTE/.venv/bin/python.*train_classifier_gpu' | grep -v grep | head -1 || true"
  sleep 45
done
if ! "${SSH[@]}" "grep -q MORNING_TAIL_DONE $REMOTE/logs/server_morning_tail.log"; then
  echo "morning tail ended without completion marker" >&2
  exit 1
fi

files=()
files+=(
  "$HOST:$REMOTE/work/profile_promotion_decision.json"
  "$HOST:$REMOTE/work/position_promotion_decision.json"
  "$HOST:$REMOTE/work/w409c_ridge96_current.json"
)
for prefix in control409s31415 pos409s31415 mark409s31415; do
  files+=(
    "$HOST:$REMOTE/work/${prefix}_select342_history.json"
    "$HOST:$REMOTE/work/${prefix}_select342_config.json"
    "$HOST:$REMOTE/work/${prefix}_holdout378_history.json"
    "$HOST:$REMOTE/work/${prefix}_holdout378_config.json"
    "$HOST:$REMOTE/work/${prefix}_growth_report.json"
    "$HOST:$REMOTE/work/${prefix}_growth_ridge96.json"
    "$HOST:$REMOTE/work/${prefix}_growth_w409c_joint96.json"
  )
done
scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "${files[@]}" "$ROOT/work/"
ECUP_ROOT="$ROOT" "$ROOT/.venv/bin/python" \
  "$ROOT/work/evaluate_seed31415_confirmation.py" \
  > "$ROOT/logs/seed31415_confirmation.log" 2>&1
cat "$ROOT/logs/seed31415_confirmation.log"
