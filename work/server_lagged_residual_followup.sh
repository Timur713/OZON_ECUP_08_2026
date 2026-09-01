#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/work/build_lagged_residual_clean_candidate.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_OUT="$ROOT/work"
mkdir -p "$LOGS"

pair_pid=$(cat "$ROOT/work/server_pairsafe_residual.pid")
echo "WAIT pair-safe supervisor pid=$pair_pid $(date --iso-8601=seconds)"
while kill -0 "$pair_pid" 2>/dev/null; do
  sleep 30
done
if [ ! -s "$ROOT/work/residcurve_pairsafe_report.json" ]; then
  echo "pair-safe handoff ended without report" >&2
  exit 1
fi
if [ ! -s "$SCRIPT" ]; then
  echo "lagged-residual builder is missing" >&2
  exit 1
fi

echo "START conditional lagged-residual follow-up $(date --iso-8601=seconds)"
"$PY" "$SCRIPT" > "$LOGS/lagged_residual_clean_candidate.log" 2>&1
test -s "$ROOT/work/lagged_residual_clean_candidate_meta.json"
cat "$ROOT/work/lagged_residual_clean_candidate_meta.json"
echo "LAGGED_RESIDUAL_FOLLOWUP_DONE $(date --iso-8601=seconds)"
