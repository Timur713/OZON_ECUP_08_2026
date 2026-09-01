#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
REMOTE=/home/ubuntu/ecup
SSH=(ssh -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "$HOST")

cd "$ROOT"
while ! "${SSH[@]}" "test -s $REMOTE/work/server_event_control.pid"; do
  sleep 15
done
while "${SSH[@]}" "kill -0 \$(cat $REMOTE/work/server_event_control.pid) 2>/dev/null"; do
  "${SSH[@]}" "date; tail -1 $REMOTE/logs/event_control_supervisor.log 2>/dev/null || true; tail -1 $REMOTE/logs/control409_select342.log 2>/dev/null || true; tail -1 $REMOTE/logs/control409_holdout378.log 2>/dev/null || true"
  sleep 30
done

if ! "${SSH[@]}" "test -s $REMOTE/work/control409_growth_report.json"; then
  echo "event-control pipeline stopped without a report" >&2
  "${SSH[@]}" "tail -100 $REMOTE/logs/event_control_supervisor.log; tail -100 $REMOTE/logs/control409_select342.log 2>/dev/null || true; tail -100 $REMOTE/logs/control409_holdout378.log 2>/dev/null || true"
  exit 1
fi

scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
  "$HOST:$REMOTE/work/control409_growth_report.json" \
  "$HOST:$REMOTE/work/control409_growth_val.npy" \
  "$HOST:$REMOTE/work/control409_select342_history.json" \
  "$HOST:$REMOTE/work/control409_holdout378_history.json" \
  "$ROOT/work/"

"$ROOT/.venv/bin/python" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/control409_growth_val.npy" --repeats 96 \
  > "$ROOT/work/control409_growth_ridge96.json"

"$ROOT/.venv/bin/python" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/w409c_val.npy" "$ROOT/work/control409_growth_val.npy" \
  --joint --repeats 96 > "$ROOT/work/control409_growth_w409c_joint96.json"

"$ROOT/.venv/bin/python" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/control409_growth_val.npy" "$ROOT/work/event409_growth_val.npy" \
  --joint --repeats 96 > "$ROOT/work/event409_control_joint96.json"

echo "event-control artifacts fetched and 96-split audits completed"
