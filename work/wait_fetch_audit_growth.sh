#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
REMOTE=/home/ubuntu/ecup
SSH=(ssh -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "$HOST")

cd "$ROOT"
while ! "${SSH[@]}" "test -s $REMOTE/work/residual_growth_report.json"; do
  "${SSH[@]}" "date; tail -1 $REMOTE/logs/multi_anchor_growth.log; tail -1 $REMOTE/logs/residual_growth.log 2>/dev/null || true"
  if ! "${SSH[@]}" "kill -0 \$(cat $REMOTE/work/multi_anchor_growth.pid) 2>/dev/null || kill -0 \$(cat $REMOTE/work/server_growth_queue.pid) 2>/dev/null"; then
    echo "growth pipeline stopped without a residual report" >&2
    "${SSH[@]}" "tail -80 $REMOTE/logs/multi_anchor_growth.log; tail -80 $REMOTE/logs/residual_growth.log"
    exit 1
  fi
  sleep 30
done

scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
  "$HOST:$REMOTE/work/multi_anchor_growth_report.json" \
  "$HOST:$REMOTE/work/multi_anchor_growth_oof.npz" \
  "$HOST:$REMOTE/work/multi_anchor_growth_val.npy" \
  "$HOST:$REMOTE/work/multi_anchor_growth_final.npy" \
  "$HOST:$REMOTE/work/multi_anchor_growth_final_components.npz" \
  "$HOST:$REMOTE/work/residual_growth_report.json" \
  "$HOST:$REMOTE/work/residual_growth_val.npy" \
  "$HOST:$REMOTE/work/residual_growth_candidate_final.npy" \
  "$ROOT/work/"

"$ROOT/.venv/bin/python" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/multi_anchor_growth_val.npy" \
  "$ROOT/work/residual_growth_val.npy" \
  --repeats 96 > "$ROOT/work/growth_individual_ridge96.json"

"$ROOT/.venv/bin/python" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/w409c_val.npy" \
  "$ROOT/work/w409twoa_val.npy" \
  "$ROOT/work/multi_anchor_growth_val.npy" \
  --joint --repeats 96 > "$ROOT/work/growth_joint_ridge96.json"

echo "growth artifacts fetched and 96-split audits completed"
