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

echo "Waiting for the medium-model night run"
while pgrep -f '[s]erver_night_run.sh' >/dev/null; do sleep 30; done
if ! grep -q '^NIGHT_RUN_DONE$' "$LOGS/night_supervisor.log"; then
  echo "Medium-model night run did not finish successfully"
  tail -100 "$LOGS/night_supervisor.log"
  exit 1
fi

# Calendar-aware target-window model, including the exact 2025 analogue at
# anchor 43 (history through Feb 13, target starts Feb 14).
run_model cls300cal_val --mode validate --window 300 --width 256 --blocks 8 \
  --epochs 3 --seed 808 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --anchor-start 43
CAL_EPOCH=$(best_epoch "$ROOT/work/cls300cal_val_history.json")
echo "cls300cal best epoch: $CAL_EPOCH"
run_model cls300cal_full --mode final --window 300 --width 256 --blocks 8 \
  --epochs "$CAL_EPOCH" --seed 808 --stride 4 --frac 0.25 --channels all \
  --bs 2048 --pred-bs 2048 --calendar --anchor-start 43

# Condition each user's history on the aggregate market trajectory.  This is
# the non-leaking version of season conditioning: every anchor only sees the
# population history available up to that anchor, while the calendar channels
# describe the known target window.
run_model cls300mkt_val --mode validate --window 300 --width 256 --blocks 8 \
  --epochs 3 --seed 909 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --anchor-start 43
MARKET_EPOCH=$(best_epoch "$ROOT/work/cls300mkt_val_history.json")
echo "cls300mkt best epoch: $MARKET_EPOCH"
run_model cls300mkt_full --mode final --window 300 --width 256 --blocks 8 \
  --epochs "$MARKET_EPOCH" --seed 909 --stride 4 --frac 0.25 --channels all \
  --bs 2048 --pred-bs 2048 --calendar --market --anchor-start 43

# First capacity experiment: a materially wider/deeper 409-day classifier.
# Batch 2048 was preflighted and OOMed; 1024 leaves enough headroom on a 24 GB 4090.
WIDE_TAG=cls409wide
if ! run_model ${WIDE_TAG}_val --mode validate --window 409 --width 384 --blocks 12 \
  --epochs 2 --seed 303 --stride 4 --frac 0.25 --channels all --bs 1024 --pred-bs 1024; then
  WIDE_TAG=cls409wide768
  run_model ${WIDE_TAG}_val --mode validate --window 409 --width 384 --blocks 12 \
    --epochs 2 --seed 303 --stride 4 --frac 0.25 --channels all --bs 768 --pred-bs 768
fi
WIDE_EPOCH=$(best_epoch "$ROOT/work/${WIDE_TAG}_val_history.json")
echo "$WIDE_TAG best epoch: $WIDE_EPOCH"
if [[ $WIDE_TAG == cls409wide ]]; then
  WIDE_BATCH=1024
else
  WIDE_BATCH=768
fi
run_model ${WIDE_TAG}_full --mode final --window 409 --width 384 --blocks 12 \
  --epochs "$WIDE_EPOCH" --seed 303 --stride 4 --frac 0.25 --channels all \
  --bs "$WIDE_BATCH" --pred-bs "$WIDE_BATCH"

# Second structural experiment: make representation learning primarily serve
# the buy/no-buy task, which accounts for most of the attainable variance.
run_model cls409class_val --mode validate --window 409 --width 256 --blocks 8 \
  --epochs 3 --seed 404 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --mix 0.9 --class-weight 4.0 --magnitude-weight 0.25 --direct-weight 0.10
CLASS_EPOCH=$(best_epoch "$ROOT/work/cls409class_val_history.json")
echo "cls409class best epoch: $CLASS_EPOCH"
run_model cls409class_full --mode final --window 409 --width 256 --blocks 8 \
  --epochs "$CLASS_EPOCH" --seed 404 --stride 4 --frac 0.25 --channels all \
  --bs 2048 --pred-bs 2048 --mix 0.9 --class-weight 4.0 \
  --magnitude-weight 0.25 --direct-weight 0.10

# Third experiment: preserve the new multitask objective but change the
# information horizon, the one diversity source that repeatedly helped the stack.
run_model cls120_val --mode validate --window 120 --width 256 --blocks 8 \
  --epochs 2 --seed 505 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
SHORT_EPOCH=$(best_epoch "$ROOT/work/cls120_val_history.json")
echo "cls120 best epoch: $SHORT_EPOCH"
run_model cls120_full --mode final --window 120 --width 256 --blocks 8 \
  --epochs "$SHORT_EPOCH" --seed 505 --stride 4 --frac 0.25 --channels all \
  --bs 2048 --pred-bs 2048

# Replicate the strongest medium configuration with an independent seed.
run_model cls300b_val --mode validate --window 300 --width 256 --blocks 8 \
  --epochs 3 --seed 606 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
SEED_EPOCH=$(best_epoch "$ROOT/work/cls300b_val_history.json")
echo "cls300b best epoch: $SEED_EPOCH"
run_model cls300b_full --mode final --window 300 --width 256 --blocks 8 \
  --epochs "$SEED_EPOCH" --seed 606 --stride 4 --frac 0.25 --channels all \
  --bs 2048 --pred-bs 2048

# An extreme short-memory classifier: weak alone may still be useful to ridge.
run_model cls60_val --mode validate --window 60 --width 256 --blocks 8 \
  --epochs 2 --seed 707 --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
TINY_EPOCH=$(best_epoch "$ROOT/work/cls60_val_history.json")
echo "cls60 best epoch: $TINY_EPOCH"
run_model cls60_full --mode final --window 60 --width 256 --blocks 8 \
  --epochs "$TINY_EPOCH" --seed 707 --stride 4 --frac 0.25 --channels all \
  --bs 2048 --pred-bs 2048

stamp
echo "EXTRA_RUN_DONE"
