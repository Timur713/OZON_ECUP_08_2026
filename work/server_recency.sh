#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT" ECUP_MAT="$ROOT/work/mat" ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
run () {
  tag=$1; start=$2; end=$3
  [ -s "$ROOT/work/${tag}_val.npy" ] && { echo "SKIP $tag"; return; }
  echo "START $tag anchors ${start}..${end} $(date --iso-8601=seconds)"
  "$PY" work/train_exact_scale.py "$tag" --variant plain --window 409 --seed 93 --epochs 2 \
    --anchor-start "$start" --anchor-end "$end" > "$ROOT/logs/${tag}.log" 2>&1 || { echo "FAIL $tag"; return; }
  echo "DONE $tag $(grep -h DONE "$ROOT/logs/${tag}.log" | tail -1)"
}
run exrec_older  150 306
run exrec_oldest  90 246
echo "RECENCY_DONE $(date --iso-8601=seconds)"
