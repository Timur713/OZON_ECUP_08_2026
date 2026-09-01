#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/work/exp_residcurve_pairsafe.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_OUT="$ROOT/work"
export ECUP_NUSERS=250000
export ECUP_TRAIN_FRAC=0.30
export ECUP_MAX_ROUNDS=300
export ECUP_MAX_PAIRS=0
export ECUP_LGB_THREADS=24
mkdir -p "$LOGS"

if [ ! -s "$SCRIPT" ]; then
  echo "pair-safe diagnostic script is missing" >&2
  exit 1
fi

echo "START pair-safe residual diagnostic in CPU/GPU parallel $(date --iso-8601=seconds)"
"$PY" "$SCRIPT" > "$LOGS/residcurve_pairsafe.log" 2>&1
test -s "$ROOT/work/residcurve_pairsafe_report.json"
cat "$ROOT/work/residcurve_pairsafe_report.json"
echo "PAIRSAFE_RESIDUAL_DONE $(date --iso-8601=seconds)"
