#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
REMOTE=/home/ubuntu/ecup
SSH=(ssh -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "$HOST")

cd "$ROOT"
while "${SSH[@]}" "kill -0 \$(cat $REMOTE/work/server_survival_growth.pid) 2>/dev/null"; do
  "${SSH[@]}" "date; tail -1 $REMOTE/logs/survival_growth_supervisor.log; tail -1 $REMOTE/logs/surv409_select342.log 2>/dev/null || true; tail -1 $REMOTE/logs/surv409_holdout378.log 2>/dev/null || true"
  sleep 30
done

if ! "${SSH[@]}" "test -s $REMOTE/work/surv409_growth_report.json"; then
  echo "survival pipeline stopped without a report" >&2
  "${SSH[@]}" "tail -100 $REMOTE/logs/survival_growth_supervisor.log; tail -100 $REMOTE/logs/surv409_select342.log; tail -100 $REMOTE/logs/surv409_holdout378.log 2>/dev/null || true"
  exit 1
fi

scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
  "$HOST:$REMOTE/work/surv409_growth_report.json" \
  "$HOST:$REMOTE/work/surv409_growth_val.npy" \
  "$HOST:$REMOTE/work/surv409_select342_history.json" \
  "$HOST:$REMOTE/work/surv409_holdout378_history.json" \
  "$ROOT/work/"

"$ROOT/.venv/bin/python" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/surv409_growth_val.npy" --repeats 96 \
  > "$ROOT/work/surv409_growth_ridge96.json"

"$ROOT/.venv/bin/python" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/w409c_val.npy" "$ROOT/work/surv409_growth_val.npy" \
  --joint --repeats 96 > "$ROOT/work/surv409_growth_w409c_joint96.json"

"$ROOT/.venv/bin/python" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/w409c_val.npy" "$ROOT/work/w409twoa_val.npy" \
  "$ROOT/work/surv409_growth_val.npy" --joint --repeats 96 \
  > "$ROOT/work/surv409_growth_full_joint96.json"

echo "survival artifacts fetched and 96-split audits completed"
