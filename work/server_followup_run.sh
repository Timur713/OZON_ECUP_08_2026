#!/usr/bin/env bash
set -uo pipefail

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
run_pair() {
  local tag=$1
  local validation_epochs=$2
  shift 2
  if ! run_model "${tag}_val" --mode validate --epochs "$validation_epochs" "$@"; then
    echo "SKIP ${tag}_full"
    return 0
  fi
  local selected_epoch
  selected_epoch=$(best_epoch "$ROOT/work/${tag}_val_history.json")
  echo "$tag best epoch: $selected_epoch"
  if ! run_model "${tag}_full" --mode final --epochs "$selected_epoch" "$@"; then
    echo "SKIP_AFTER_FAILURE ${tag}_full"
  fi
}
run_pair_if_missing() {
  local tag=$1
  shift
  if [[ -s $ROOT/work/${tag}_full_final.npy ]]; then
    echo "PRESENT ${tag}_full"
    return 0
  fi
  echo "RECOVER $tag"
  run_pair "$tag" "$@"
}

echo "Waiting for the primary extra queue"
while pgrep -f '[s]erver_extra_run.sh' >/dev/null; do sleep 30; done
if ! grep -q '^EXTRA_RUN_DONE$' "$LOGS/extra_supervisor.log"; then
  echo "WARNING: primary extra queue ended early; using the free GPU for follow-up work"
  tail -100 "$LOGS/extra_supervisor.log"
fi

# Recovery comes first. A failure in the primary queue must not discard every
# model after it; already completed files are detected and skipped cheaply.
run_pair_if_missing cls300cal 3 --window 300 --width 256 --blocks 8 --seed 808 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --anchor-start 43
run_pair_if_missing cls300mkt 3 --window 300 --width 256 --blocks 8 --seed 909 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --anchor-start 43
if [[ ! -s $ROOT/work/cls409wide_full_final.npy && \
      ! -s $ROOT/work/cls409wide768_full_final.npy ]]; then
  run_pair cls409wide768 2 --window 409 --width 384 --blocks 12 --seed 303 \
    --stride 4 --frac 0.25 --channels all --bs 768 --pred-bs 768
fi
run_pair_if_missing cls409class 3 --window 409 --width 256 --blocks 8 --seed 404 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --mix 0.9 --class-weight 4.0 --magnitude-weight 0.25 --direct-weight 0.10
run_pair_if_missing cls120 2 --window 120 --width 256 --blocks 8 --seed 505 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
run_pair_if_missing cls300b 3 --window 300 --width 256 --blocks 8 --seed 606 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
run_pair_if_missing cls60 2 --window 60 --width 256 --blocks 8 --seed 707 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048

# Highest-priority untested mechanism: give the trunk explicit multi-window
# averages and nonzero frequencies. Regularity is the dominant feature family,
# while the convolutional pooling only exposes global mean/max and last 14 days.
run_pair cls43gift 3 --window 43 --width 256 --blocks 8 --seed 1000 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --summary --anchor-start 43 \
  --special-anchor 43 --special-repeat 8

run_pair cls409hyb 3 --window 409 --width 256 --blocks 8 --seed 1001 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 --summary

# Same compute budget spread across twice as many calendar anchors.
run_pair cls409dense 2 --window 409 --width 256 --blocks 8 --seed 1002 \
  --stride 2 --frac 0.125 --channels all --bs 2048 --pred-bs 2048

# Condition the strongest classifier horizon on both known calendar and the
# observable aggregate-market history.
run_pair cls409mkt 3 --window 409 --width 256 --blocks 8 --seed 1003 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --anchor-start 43

# Hybrid summaries at a shorter information horizon for ensemble diversity.
run_pair cls120hyb 2 --window 120 --width 256 --blocks 8 --seed 1004 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 --summary

# Combined hybrid/seasonal model; lower priority but keeps the GPU occupied.
run_pair cls300hybmkt 3 --window 300 --width 256 --blocks 8 --seed 1005 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --summary --calendar --market --anchor-start 43

# Extreme horizons are deliberately last: weak alone can still add orthogonal
# residuals, and these jobs are cheap enough to fill remaining morning time.
run_pair cls30 2 --window 30 --width 256 --blocks 8 --seed 1006 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
run_pair cls240 2 --window 240 --width 256 --blocks 8 --seed 1007 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048

stamp
echo "FOLLOWUP_RUN_DONE"
