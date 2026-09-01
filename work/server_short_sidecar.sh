#!/usr/bin/env bash
# Short-window queue that coexists with the long target-profile process.
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
run_pair() {
  local tag=$1
  local validation_epochs=$2
  shift 2
  if [[ -s $ROOT/work/${tag}_full_final.npy ]]; then
    echo "PRESENT ${tag}_full"
    return 0
  fi
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "START ${tag}_val"
  if ! "$PY" "$SCRIPT" "${tag}_val" --mode validate \
      --epochs "$validation_epochs" "$@" >"$LOGS/${tag}_val.log" 2>&1; then
    echo "FAILED ${tag}_val"
    return 0
  fi
  local epoch mix
  epoch=$(best_row "$ROOT/work/${tag}_val_history.json" epoch)
  mix=$(best_row "$ROOT/work/${tag}_val_history.json" best_hurdle_weight)
  echo "SELECT $tag epoch=$epoch mix=$mix"
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "START ${tag}_full"
  if "$PY" "$SCRIPT" "${tag}_full" --mode final --epochs "$epoch" "$@" \
      --mix "$mix" >"$LOGS/${tag}_full.log" 2>&1; then
    echo "DONE ${tag}_full"
  else
    echo "FAILED ${tag}_full"
  fi
}

while pgrep -f '[t]rain_classifier_gpu.py cls43exactclass_' >/dev/null; do
  sleep 2
done

run_pair cls43gift 3 --window 43 --width 256 --blocks 8 --seed 1000 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --summary --anchor-start 43 \
  --special-anchor 43 --special-repeat 8

run_pair cls120hyb 2 --window 120 --width 256 --blocks 8 --seed 1004 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 --summary
