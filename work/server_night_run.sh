#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY=$ROOT/.venv/bin/python
SCRIPT=$ROOT/work/train_classifier_gpu.py
LOGS=$ROOT/logs
export ECUP_ROOT=$ROOT
export ECUP_MAT=$ROOT/work/mat
export ECUP_OUT=$ROOT/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

stamp() { date '+%Y-%m-%d %H:%M:%S %Z'; }
run_model() {
  local tag=$1
  shift
  stamp
  echo "START $tag $*"
  if "$PY" "$SCRIPT" "$tag" "$@" >"$LOGS/$tag.log" 2>&1; then
    echo "DONE $tag"
    return 0
  fi
  echo "FAILED $tag"
  tail -80 "$LOGS/$tag.log"
  return 1
}
best_epoch() {
  "$PY" - "$1" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
print(min(rows,key=lambda row: row['score'])['epoch'])
PY
}

echo "Waiting for baseline replication"
while pgrep -f '[t]rain_seq2.py tcn409rep' >/dev/null; do sleep 20; done
if ! grep -q '^BEST tcn409rep:' "$LOGS/tcn409rep.log"; then
  echo "Baseline replication did not finish successfully"
  tail -100 "$LOGS/tcn409rep.log"
  exit 1
fi

run_model cls300_val --mode validate --window 300 --width 256 --blocks 8 \
  --epochs 3 --seed 101 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
A_EPOCH=$(best_epoch "$ROOT/work/cls300_val_history.json")
echo "cls300 best epoch: $A_EPOCH"
run_model cls300_full --mode final --window 300 --width 256 --blocks 8 \
  --epochs "$A_EPOCH" --seed 101 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048

run_model cls409_val --mode validate --window 409 --width 256 --blocks 8 \
  --epochs 2 --seed 202 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
B_EPOCH=$(best_epoch "$ROOT/work/cls409_val_history.json")
echo "cls409 best epoch: $B_EPOCH"
run_model cls409_full --mode final --window 409 --width 256 --blocks 8 \
  --epochs "$B_EPOCH" --seed 202 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048

stamp
echo "NIGHT_RUN_DONE"
