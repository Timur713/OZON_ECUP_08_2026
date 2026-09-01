#!/usr/bin/env bash
# Fill otherwise unused GPU capacity while the short-window holdout runs.
# Failure is safe: the primary queue retries cls300tprofhead by itself.
set -uo pipefail

ROOT=/home/ubuntu/ecup
PY=$ROOT/.venv/bin/python
SCRIPT=$ROOT/work/train_classifier_gpu.py
LOGS=$ROOT/logs
export ECUP_ROOT=$ROOT
export ECUP_MAT=$ROOT/work/mat
export ECUP_OUT=$ROOT/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

best_row() {
  "$PY" - "$1" "$2" <<'PY'
import json,sys
row=min(json.load(open(sys.argv[1])),key=lambda value: value['score'])
print(row.get(sys.argv[2], 0.7))
PY
}

if [[ -s $ROOT/work/cls300tprofhead_full_final.npy ]]; then
  echo "PRESENT cls300tprofhead_full"
  exit 0
fi

date '+%Y-%m-%d %H:%M:%S %Z'
echo START cls300tprofhead_val
if ! "$PY" "$SCRIPT" cls300tprofhead_val --mode validate --epochs 3 \
    --window 300 --width 256 --blocks 8 --seed 1100 --stride 4 --frac 0.25 \
    --channels all --bs 2048 --pred-bs 2048 --calendar --market \
    --target-profile --target-profile-head --anchor-start 43 \
    >"$LOGS/cls300tprofhead_val.log" 2>&1; then
  echo FAILED cls300tprofhead_val
  exit 0
fi

epoch=$(best_row "$ROOT/work/cls300tprofhead_val_history.json" epoch)
mix=$(best_row "$ROOT/work/cls300tprofhead_val_history.json" best_hurdle_weight)
echo "SELECT epoch=$epoch mix=$mix"
date '+%Y-%m-%d %H:%M:%S %Z'
echo START cls300tprofhead_full
if "$PY" "$SCRIPT" cls300tprofhead_full --mode final --epochs "$epoch" \
    --window 300 --width 256 --blocks 8 --seed 1100 --stride 4 --frac 0.25 \
    --channels all --bs 2048 --pred-bs 2048 --calendar --market \
    --target-profile --target-profile-head --anchor-start 43 --mix "$mix" \
    >"$LOGS/cls300tprofhead_full.log" 2>&1; then
  echo DONE cls300tprofhead_full
else
  echo FAILED cls300tprofhead_full
fi
