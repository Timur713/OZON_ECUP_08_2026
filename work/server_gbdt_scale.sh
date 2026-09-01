#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT" ECUP_MAT="$ROOT/work/mat" ECUP_OUT="$ROOT/work"
cd "$ROOT"
echo "START control $(date --iso-8601=seconds)"
"$PY" work/train_gbdt_scale.py gbdtctl --anchor-stride 12 --learning-rate 0.05 --leaves 127 --seeds 2 \
  > logs/gbdtctl.log 2>&1 || echo "FAIL control"
grep -h FOLD378 logs/gbdtctl.log || true
echo "START scaled $(date --iso-8601=seconds)"
"$PY" work/train_gbdt_scale.py gbdtscale --anchor-stride 4 --learning-rate 0.02 --rounds 1500 --leaves 255 --seeds 2 \
  > logs/gbdtscale.log 2>&1 || echo "FAIL scaled"
grep -h FOLD378 logs/gbdtscale.log || true
echo "GBDT_SCALE_DONE $(date --iso-8601=seconds)"
