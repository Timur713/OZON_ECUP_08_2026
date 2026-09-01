#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT" ECUP_MAT="$ROOT/work/mat" ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
for seed in 93 1310; do
  tag="exch17_s${seed}"
  [ -s "$ROOT/work/${tag}_val.npy" ] && { echo "SKIP $tag"; continue; }
  echo "START $tag $(date --iso-8601=seconds)"
  "$PY" work/train_exact_scale.py "$tag" --variant plain --window 409 --channels all \
    --seed "$seed" --epochs 2 --batch-size 512 --prediction-batch-size 2048 \
    > "$ROOT/logs/${tag}.log" 2>&1 || { echo "FAIL $tag"; continue; }
  "$PY" work/evaluate_validation_ridge.py \
    "$ROOT/work/w409c_val.npy" "$ROOT/work/w409_exact_decay_s93_val.npy" \
    "$ROOT/work/${tag}_val.npy" --joint --repeats 96 \
    > "$ROOT/work/${tag}_admitted_joint96.json" 2>/dev/null
  echo "DONE $tag $(grep -h DONE "$ROOT/logs/${tag}.log" | tail -1)"
done
echo "CHANNELS_DONE $(date --iso-8601=seconds)"
