#!/usr/bin/env bash
# Additional short-window structural queue. It starts only after the first
# sidecar and yields while a wide 409 model or high VRAM pressure is present.
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
row=min(json.load(open(sys.argv[1])), key=lambda value: value['score'])
print(row.get(sys.argv[2], 0.7))
PY
}

wait_for_capacity() {
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | head -1 | tr -d ' ')
    if ! pgrep -f '[t]rain_classifier_gpu.py cls409wide' >/dev/null \
        && [[ ${used:-99999} -lt 16000 ]]; then
      return 0
    fi
    echo "WAIT_CAPACITY used_mib=${used:-unknown}"
    sleep 30
  done
}

run_pair() {
  local tag=$1
  local validation_epochs=$2
  shift 2
  if [[ -s $ROOT/work/${tag}_full_final.npy ]]; then
    echo "PRESENT ${tag}_full"
    return 0
  fi
  wait_for_capacity
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
  wait_for_capacity
  date '+%Y-%m-%d %H:%M:%S %Z'
  echo "START ${tag}_full"
  if "$PY" "$SCRIPT" "${tag}_full" --mode final --epochs "$epoch" "$@" \
      --mix "$mix" >"$LOGS/${tag}_full.log" 2>&1; then
    echo "DONE ${tag}_full"
  else
    echo "FAILED ${tag}_full"
  fi
}

while pgrep -f '[s]erver_short_sidecar.sh' >/dev/null; do
  sleep 5
done

# Market/calendar was the strongest new structural mechanism at window 300.
# Test whether shorter information horizons improve its bias/variance tradeoff.
run_pair cls120hybmkt 2 --window 120 --width 256 --blocks 8 --seed 1301 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --summary --calendar --market --anchor-start 43
run_pair cls180hybmkt 2 --window 180 --width 256 --blocks 8 --seed 1302 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --summary --calendar --market --anchor-start 43
run_pair cls90hybmkt 2 --window 90 --width 256 --blocks 8 --seed 1303 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --summary --calendar --market --anchor-start 43
run_pair cls60hybmkt 2 --window 60 --width 256 --blocks 8 --seed 1304 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --summary --calendar --market --anchor-start 43

# Season-neutral horizon coverage if the richer branches all finish early.
run_pair cls240hyb 2 --window 240 --width 256 --blocks 8 --seed 1305 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 --summary

date '+%Y-%m-%d %H:%M:%S %Z'
echo SHORT_TAIL_DONE
