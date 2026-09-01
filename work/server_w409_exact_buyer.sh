#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
structures_pid=$(cat "$ROOT/work/server_w409_exact_structures.pid")
echo "WAIT structures pid=$structures_pid $(date --iso-8601=seconds)"
while kill -0 "$structures_pid" 2>/dev/null; do sleep 20; done
tag=w409_exact_buyer_s93
"$PY" "$ROOT/work/train_w409_exact_decay.py" "$tag" --variant buyer --seed 93 \
  > "$ROOT/logs/${tag}.log" 2>&1
"$PY" "$ROOT/work/evaluate_validation_ridge.py" "$ROOT/work/${tag}_val.npy" \
  --repeats 96 > "$ROOT/work/${tag}_ridge96.json"
"$PY" "$ROOT/work/evaluate_validation_ridge.py" "$ROOT/work/w409c_val.npy" \
  "$ROOT/work/${tag}_val.npy" --joint --repeats 96 \
  > "$ROOT/work/${tag}_w409c_joint96.json"
"$PY" "$ROOT/work/evaluate_w409_exact_buyer.py" \
  > "$ROOT/logs/w409_exact_buyer_decision.log" 2>&1
cat "$ROOT/logs/w409_exact_buyer_decision.log"
echo "W409_EXACT_BUYER_DONE $(date --iso-8601=seconds)"
