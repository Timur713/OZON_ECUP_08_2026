#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
CONTROL="$ROOT/work/train_classifier_gpu_event_frozen.py"
POSITION="$ROOT/work/train_classifier_gpu_position_frozen.py"
MARKED="$ROOT/work/train_classifier_gpu_marked_frozen.py"
BUILD="$ROOT/work/build_frozen_profile_report.py"
AUDIT="$ROOT/work/evaluate_validation_ridge.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

position_pid=$(cat "$ROOT/work/server_position_tail.pid")
echo "WAIT two-seed position tail pid=$position_pid $(date --iso-8601=seconds)"
while kill -0 "$position_pid" 2>/dev/null; do sleep 30; done
if [ ! -s "$ROOT/work/position_promotion_decision.json" ]; then
  echo "position dependency ended without decision" >&2
  exit 1
fi
for file in "$CONTROL" "$POSITION" "$MARKED" "$BUILD" "$AUDIT"; do
  if [ ! -s "$file" ]; then echo "missing morning-tail dependency $file" >&2; exit 1; fi
done

SEED=31415
COMMON=(
  --mode validate --window 409 --width 256 --blocks 8 --seed "$SEED"
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --calendar --private-selection --anchor-start 43
)
run_pair() {
  local trainer=$1 prefix=$2 role=$3 event_profile=${4:-false}
  local -a profile_args=()
  if [ "$event_profile" = true ]; then profile_args+=(--event-profile); fi
  echo "START morning $prefix selection $(date --iso-8601=seconds)"
  "$PY" "$trainer" "${prefix}_select342" "${COMMON[@]}" "${profile_args[@]}" \
    --val 342 --epochs 3 \
    > "$LOGS/${prefix}_select342.log" 2>&1
  echo "START morning $prefix untouched $(date --iso-8601=seconds)"
  "$PY" "$trainer" "${prefix}_holdout378" "${COMMON[@]}" "${profile_args[@]}" \
    --val 378 --epochs 3 \
    > "$LOGS/${prefix}_holdout378.log" 2>&1
  "$PY" "$BUILD" --select-tag "${prefix}_select342" \
    --holdout-tag "${prefix}_holdout378" --output-tag "${prefix}_growth" \
    --seed "$SEED" --role "$role" "${profile_args[@]}" \
    > "$LOGS/${prefix}_growth_build.log" 2>&1
  "$PY" "$AUDIT" "$ROOT/work/${prefix}_growth_val.npy" --repeats 96 \
    > "$ROOT/work/${prefix}_growth_ridge96.json"
  "$PY" "$AUDIT" "$ROOT/work/w409c_val.npy" \
    "$ROOT/work/${prefix}_growth_val.npy" --joint --repeats 96 \
    > "$ROOT/work/${prefix}_growth_w409c_joint96.json"
}

# Order is fixed before any seed-31415 result.  Control makes the positional
# comparison causal; the third marked-event seed measures stability of the
# other still-live representation hypothesis rather than filling idle time.
run_pair "$CONTROL" control409s31415 matched_control_third_seed
run_pair "$POSITION" pos409s31415 relative_position_third_seed
run_pair "$MARKED" mark409s31415 marked_event_third_seed true
echo "MORNING_TAIL_DONE $(date --iso-8601=seconds)"
