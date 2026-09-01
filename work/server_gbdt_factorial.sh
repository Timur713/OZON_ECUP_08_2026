#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT" ECUP_MAT="$ROOT/work/mat" ECUP_OUT="$ROOT/work"
cd "$ROOT"

# do not compete for cores with the running scaled run
while pgrep -f "train_gbdt_scale.py gbdtscale" > /dev/null; do sleep 30; done

# G3 cheapest first, then G2, then G1: each isolates ONE factor against the control
echo "START gbdtG3_leaves $(date --iso-8601=seconds)"
"$PY" work/train_gbdt_scale.py gbdtG3_leaves --anchor-stride 12 --learning-rate 0.05 --leaves 255 --seeds 2 \
  > logs/gbdtG3_leaves.log 2>&1 || echo "FAIL G3"
grep -h FOLD378 logs/gbdtG3_leaves.log || true

echo "START gbdtG2_lr $(date --iso-8601=seconds)"
"$PY" work/train_gbdt_scale.py gbdtG2_lr --anchor-stride 12 --learning-rate 0.02 --rounds 1500 --leaves 127 --seeds 2 \
  > logs/gbdtG2_lr.log 2>&1 || echo "FAIL G2"
grep -h FOLD378 logs/gbdtG2_lr.log || true

echo "START gbdtG1_density $(date --iso-8601=seconds)"
"$PY" work/train_gbdt_scale.py gbdtG1_density --anchor-stride 4 --learning-rate 0.05 --leaves 127 --seeds 2 \
  > logs/gbdtG1_density.log 2>&1 || echo "FAIL G1"
grep -h FOLD378 logs/gbdtG1_density.log || true

echo "GBDT_FACTORIAL_DONE $(date --iso-8601=seconds)"
