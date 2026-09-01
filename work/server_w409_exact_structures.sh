#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_w409_exact_decay.py"
AUDIT="$ROOT/work/evaluate_validation_ridge.py"
DECIDE="$ROOT/work/evaluate_w409_exact_structures.py"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
mkdir -p "$ROOT/logs"

for variant in position event; do
  tag="w409_exact_${variant}_s93"
  echo "START $tag $(date --iso-8601=seconds)"
  "$PY" "$TRAIN" "$tag" --variant "$variant" --seed 93 \
    > "$ROOT/logs/${tag}.log" 2>&1
  "$PY" "$AUDIT" "$ROOT/work/${tag}_val.npy" --repeats 96 \
    > "$ROOT/work/${tag}_ridge96.json"
  "$PY" "$AUDIT" "$ROOT/work/w409c_val.npy" "$ROOT/work/${tag}_val.npy" \
    --joint --repeats 96 > "$ROOT/work/${tag}_w409c_joint96.json"
done
"$PY" "$DECIDE" > "$ROOT/logs/w409_exact_structures_decision.log" 2>&1
cat "$ROOT/logs/w409_exact_structures_decision.log"
echo "W409_EXACT_STRUCTURES_DONE $(date --iso-8601=seconds)"
