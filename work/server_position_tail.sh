#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_classifier_gpu_position_frozen.py"
BUILD="$ROOT/work/build_frozen_profile_report.py"
AUDIT="$ROOT/work/evaluate_validation_ridge.py"
DECIDE="$ROOT/work/evaluate_position_promotion.py"
AVERAGE="$ROOT/work/average_profile_finals.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

dependency_pid=$(cat "$ROOT/work/server_conditional_profile_full.pid")
echo "WAIT conditional profile full pid=$dependency_pid $(date --iso-8601=seconds)"
while kill -0 "$dependency_pid" 2>/dev/null; do sleep 30; done
for file in "$TRAIN" "$BUILD" "$AUDIT" "$DECIDE" "$AVERAGE"; do
  if [ ! -s "$file" ]; then echo "missing position-tail dependency $file" >&2; exit 1; fi
done

COMMON=(
  --mode validate --window 409 --width 256 --blocks 8
  --stride 4 --frac 0.25 --channels all --bs 2048 --pred-bs 2048
  --summary --calendar --private-selection --anchor-start 43
)
for seed in 1310 2718; do
  prefix=pos409
  if [ "$seed" = 2718 ]; then prefix=pos409s2718; fi
  echo "START position selection seed=$seed $(date --iso-8601=seconds)"
  "$PY" "$TRAIN" "${prefix}_select342" "${COMMON[@]}" --seed "$seed" \
    --val 342 --epochs 3 > "$LOGS/${prefix}_select342.log" 2>&1
  echo "START position untouched seed=$seed $(date --iso-8601=seconds)"
  "$PY" "$TRAIN" "${prefix}_holdout378" "${COMMON[@]}" --seed "$seed" \
    --val 378 --epochs 3 > "$LOGS/${prefix}_holdout378.log" 2>&1
  "$PY" "$BUILD" --select-tag "${prefix}_select342" \
    --holdout-tag "${prefix}_holdout378" --output-tag "${prefix}_growth" \
    --seed "$seed" --role relative_position_channels \
    > "$LOGS/${prefix}_growth_build.log" 2>&1
  "$PY" "$AUDIT" "$ROOT/work/${prefix}_growth_val.npy" --repeats 96 \
    > "$ROOT/work/${prefix}_growth_ridge96.json"
  "$PY" "$AUDIT" "$ROOT/work/w409c_val.npy" \
    "$ROOT/work/${prefix}_growth_val.npy" --joint --repeats 96 \
    > "$ROOT/work/${prefix}_growth_w409c_joint96.json"
done

"$PY" "$DECIDE" > "$LOGS/position_promotion_decision.log" 2>&1
cat "$LOGS/position_promotion_decision.log"
passed=$("$PY" - "$ROOT/work/position_promotion_decision.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1]))["tier"] != "reject" else "0")
PY
)
if [ "$passed" = 1 ]; then
  finals=()
  for seed in 1310 2718; do
    prefix=pos409
    if [ "$seed" = 2718 ]; then prefix=pos409s2718; fi
    read -r epoch mix < <("$PY" - "$ROOT/work/${prefix}_select342_history.json" <<'PY'
import json, sys
row=min(json.load(open(sys.argv[1])), key=lambda value: value["score"])
print(row["epoch"], row["best_hurdle_weight"])
PY
)
    tag="promote_position_${seed}_full"
    "$PY" "$TRAIN" "$tag" --mode final --window 409 --width 256 --blocks 8 \
      --seed "$seed" --stride 4 --frac 0.25 --channels all --bs 2048 \
      --pred-bs 2048 --summary --calendar --anchor-start 43 \
      --epochs "$epoch" --mix "$mix" > "$LOGS/${tag}.log" 2>&1
    finals+=("$ROOT/work/${tag}_final.npy")
  done
  "$PY" "$AVERAGE" position "${finals[0]}" "${finals[1]}" \
    > "$LOGS/promote_position_average.log" 2>&1
fi
echo "POSITION_TAIL_DONE passed=$passed $(date --iso-8601=seconds)"
