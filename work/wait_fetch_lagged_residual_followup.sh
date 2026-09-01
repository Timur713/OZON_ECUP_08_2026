#!/usr/bin/env bash
set -euo pipefail

ROOT="${ECUP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
KEY="${ECUP_KEY:?set ECUP_KEY to your SSH private key path}"
HOST="${ECUP_HOST:?set ECUP_HOST to ubuntu@<your-server>}"
REMOTE=/home/ubuntu/ecup
SSH=(ssh -o BatchMode=yes -o IPQoS=throughput -i "$KEY" "$HOST")

cd "$ROOT"
while ! "${SSH[@]}" "test -s $REMOTE/work/server_lagged_residual_followup.pid"; do
  sleep 15
done
while "${SSH[@]}" "kill -0 \$(cat $REMOTE/work/server_lagged_residual_followup.pid) 2>/dev/null"; do
  "${SSH[@]}" "date; tail -1 $REMOTE/logs/lagged_residual_followup_supervisor.log 2>/dev/null || true; tail -1 $REMOTE/logs/lagged_residual_clean_candidate.log 2>/dev/null || true"
  sleep 30
done

if ! "${SSH[@]}" "test -s $REMOTE/work/lagged_residual_clean_candidate_meta.json"; then
  echo "lagged-residual follow-up stopped without verdict" >&2
  "${SSH[@]}" "tail -100 $REMOTE/logs/lagged_residual_followup_supervisor.log; tail -100 $REMOTE/logs/lagged_residual_clean_candidate.log 2>/dev/null || true"
  exit 1
fi

scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
  "$HOST:$REMOTE/work/lagged_residual_clean_candidate_meta.json" \
  "$ROOT/work/"
if "${SSH[@]}" "test -s $REMOTE/submissions/candidate_lagged_residual_clean.csv"; then
  scp -o BatchMode=yes -o IPQoS=throughput -i "$KEY" \
    "$HOST:$REMOTE/submissions/candidate_lagged_residual_clean.csv" \
    "$ROOT/submissions/"
  "$ROOT/.venv/bin/python" "$ROOT/work/validate_submissions.py" \
    "$ROOT/submissions/candidate_lagged_residual_clean.csv"
fi
echo "lagged-residual verdict fetched; CSV fetched only if every frozen gate passed"
