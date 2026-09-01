#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT" ECUP_MAT="$ROOT/work/mat" ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

audit () {
  tag=$1
  "$PY" work/evaluate_validation_ridge.py "$ROOT/work/${tag}_val.npy" \
    --repeats 96 > "$ROOT/work/${tag}_ridge96.json" 2>/dev/null
  "$PY" work/evaluate_validation_ridge.py \
    "$ROOT/work/w409c_val.npy" "$ROOT/work/w409_exact_decay_s93_val.npy" \
    "$ROOT/work/${tag}_val.npy" --joint --repeats 96 \
    > "$ROOT/work/${tag}_admitted_joint96.json" 2>/dev/null
}

run () {
  tag=$1; shift
  if [ -s "$ROOT/work/${tag}_val.npy" ]; then echo "SKIP $tag"; return; fi
  echo "START $tag $(date --iso-8601=seconds)"
  "$PY" work/train_exact_scale.py "$tag" --variant plain --window 409 --seed 93 "$@" \
    > "$ROOT/logs/${tag}.log" 2>&1 || { echo "FAIL $tag"; return; }
  audit "$tag"
  echo "DONE $tag $(date --iso-8601=seconds) $(grep -h DONE "$ROOT/logs/${tag}.log" | tail -1)"
}

run exw409_s93       --anchor-start 186 --anchor-stride 12 --width 96  --epochs 2
run excap_d4_w96     --anchor-start 43  --anchor-stride 4  --width 96  --epochs 2
run excap_d4_w192    --anchor-start 43  --anchor-stride 4  --width 192 --epochs 2
run excap_d4_w192_e3 --anchor-start 43  --anchor-stride 4  --width 192 --epochs 3
echo "CAPACITY_DONE $(date --iso-8601=seconds)"
