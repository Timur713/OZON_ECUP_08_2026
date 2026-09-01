#!/usr/bin/env bash
# Priority-ordered continuation after cls300mkt. Independent failures do not
# discard the tail; the queue is intentionally longer than the morning window.
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
best_mix() {
  "$PY" - "$1" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
row=min(rows,key=lambda value: value['score'])
print(row.get('best_hurdle_weight', 0.7))
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
  if ! run_model "${tag}_val" --mode validate --epochs "$validation_epochs" "$@"; then
    echo "SKIP ${tag}_full"
    return 0
  fi
  local selected_epoch
  local selected_mix
  selected_epoch=$(best_epoch "$ROOT/work/${tag}_val_history.json")
  selected_mix=$(best_mix "$ROOT/work/${tag}_val_history.json")
  echo "$tag best epoch: $selected_epoch hurdle mix: $selected_mix"
  if ! run_model "${tag}_full" --mode final --epochs "$selected_epoch" "$@" \
      --mix "$selected_mix"; then
    echo "SKIP_AFTER_FAILURE ${tag}_full"
  fi
}

# The current market run must be safely committed before any handoff.
while ! grep -q '^DONE cls300mkt_full$' "$LOGS/extra_supervisor.log"; do sleep 30; done

# Target-season market shape was the remaining unimplemented part of the
# season-conditioning brief. Validation/final use a previous-year proxy rather
# than future aggregates, while historical training anchors expose the mapping.
if [[ -e $LOGS/skip_cls300tprof ]]; then
  echo "SKIP cls300tprof_full"
else
  run_pair cls300tprof 3 --window 300 --width 256 --blocks 8 --seed 1100 \
    --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
    --calendar --market --target-profile --anchor-start 43
fi

# Exact previous-year gift season, evaluated on a fixed 20% user holdout that
# is excluded from every training anchor. This mirrors the 50k/200k split and
# selects the epoch without consulting the leaderboard.
run_pair cls43hold 3 --window 43 --width 256 --blocks 8 --seed 1200 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --summary --anchor-start 43 \
  --special-anchor 43 --special-repeat 8 \
  --user-holdout-anchor 43 --user-holdout-frac 0.2

# Controlled counterpart on the identical user split/seed: expose the exact
# previous-year target-season aggregate profile that is also available at
# final inference. Keep both so the holdout can falsify this covariate.
if [[ -e $LOGS/skip_cls43holdtprof ]]; then
  echo "SKIP cls43holdtprof_full"
else
  run_pair cls43holdtprof 3 --window 43 --width 256 --blocks 8 --seed 1200 \
    --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
    --calendar --market --summary --target-profile --target-profile-head \
    --anchor-start 43 \
    --special-anchor 43 --special-repeat 8 \
    --user-holdout-anchor 43 --user-holdout-frac 0.2
fi

# Near-pure exact-season transfer: with stride 400, anchor 43 is the only
# source of 30-day direct/magnitude labels; the latest anchor contributes only
# short-horizon classification. The fixed user split is identical to cls43hold.
run_pair cls43exacthold 3 --window 43 --width 256 --blocks 8 --seed 1200 \
  --stride 400 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --summary --anchor-start 43 \
  --special-anchor 43 --special-repeat 31 \
  --user-holdout-anchor 43 --user-holdout-frac 0.2

# Same exact-season design with most capacity pressure on buy/no-buy. Kept as
# a queued falsification, not assumed better than the balanced objective.
run_pair cls43exactclass 3 --window 43 --width 256 --blocks 8 --seed 1200 \
  --stride 400 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --summary --anchor-start 43 \
  --special-anchor 43 --special-repeat 31 \
  --user-holdout-anchor 43 --user-holdout-frac 0.2 \
  --mix 0.9 --class-weight 4.0 --magnitude-weight 0.25 --direct-weight 0.10

# Temporal YoY validation counterpart to the failed Conv1d-static ablation.
if [[ -e $LOGS/skip_cls300tprofhead ]]; then
  echo "SKIP cls300tprofhead_full"
else
  while pgrep -f '[t]rain_classifier_gpu.py cls300tprofhead_' >/dev/null; do
    echo "WAIT cls300tprofhead sidecar"
    sleep 30
  done
  run_pair cls300tprofhead 3 --window 300 --width 256 --blocks 8 --seed 1100 \
    --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
    --calendar --market --target-profile --target-profile-head --anchor-start 43
fi

# Next expected value: explicit regularity summaries and a denser anchor
# distribution change the information/estimation problem, not just capacity.
run_pair cls409hyb 3 --window 409 --width 256 --blocks 8 --seed 1001 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 --summary

run_pair cls409dense 2 --window 409 --width 256 --blocks 8 --seed 1002 \
  --stride 2 --frac 0.125 --channels all --bs 2048 --pred-bs 2048

# Most model capacity serves the dominant buy/no-buy component.
run_pair cls409class 3 --window 409 --width 256 --blocks 8 --seed 404 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --mix 0.9 --class-weight 4.0 --magnitude-weight 0.25 --direct-weight 0.10

# Exact gift-season analogue, deliberately after robust multi-anchor changes.
run_pair cls43gift 3 --window 43 --width 256 --blocks 8 --seed 1000 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --summary --anchor-start 43 \
  --special-anchor 43 --special-repeat 8

# Capacity branch. Batch 2048 OOMed in preflight; 1024 is primary, 768 fallback.
if [[ ! -s $ROOT/work/cls409wide_full_final.npy && \
      ! -s $ROOT/work/cls409wide768_full_final.npy ]]; then
  if run_model cls409wide_val --mode validate --window 409 --width 384 --blocks 12 \
    --epochs 2 --seed 303 --stride 4 --frac 0.25 --channels all \
    --bs 1024 --pred-bs 1024; then
    WIDE_EPOCH=$(best_epoch "$ROOT/work/cls409wide_val_history.json")
    run_model cls409wide_full --mode final --window 409 --width 384 --blocks 12 \
      --epochs "$WIDE_EPOCH" --seed 303 --stride 4 --frac 0.25 --channels all \
      --bs 1024 --pred-bs 1024 || true
  else
    run_pair cls409wide768 2 --window 409 --width 384 --blocks 12 --seed 303 \
      --stride 4 --frac 0.25 --channels all --bs 768 --pred-bs 768
  fi
fi

# Long tail: structural variants first, then replication/horizon coverage.
run_pair cls120hyb 2 --window 120 --width 256 --blocks 8 --seed 1004 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 --summary
run_pair cls409mkt 3 --window 409 --width 256 --blocks 8 --seed 1003 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --calendar --market --anchor-start 43
run_pair cls120 2 --window 120 --width 256 --blocks 8 --seed 505 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
run_pair cls300b 3 --window 300 --width 256 --blocks 8 --seed 606 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
run_pair cls60 2 --window 60 --width 256 --blocks 8 --seed 707 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
run_pair cls300hybmkt 3 --window 300 --width 256 --blocks 8 --seed 1005 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048 \
  --summary --calendar --market --anchor-start 43
run_pair cls30 2 --window 30 --width 256 --blocks 8 --seed 1006 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
run_pair cls240 2 --window 240 --width 256 --blocks 8 --seed 1007 \
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048

stamp
echo "MORNING_PRIORITY_DONE"
