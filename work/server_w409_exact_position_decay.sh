#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
buyer_pid=$(cat "$ROOT/work/server_w409_exact_buyer.pid")
echo "WAIT exact buyer pid=$buyer_pid $(date --iso-8601=seconds)"
while kill -0 "$buyer_pid" 2>/dev/null; do sleep 20; done
if [ ! -s "$ROOT/work/w409_exact_buyer_decision.json" ]; then
  echo "buyer tail ended without decision" >&2
  exit 1
fi
tag=w409_exact_position_decay_s93
"$PY" "$ROOT/work/train_w409_exact_decay.py" "$tag" \
  --variant position_decay --seed 93 > "$ROOT/logs/${tag}.log" 2>&1
"$PY" "$ROOT/work/evaluate_validation_ridge.py" "$ROOT/work/${tag}_val.npy" \
  --repeats 96 > "$ROOT/work/${tag}_ridge96.json"
"$PY" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/w409c_val.npy" \
  "$ROOT/work/w409_exact_decay_s93_val.npy" \
  "$ROOT/work/w409_exact_position_s93_val.npy" \
  --joint --repeats 96 > "$ROOT/work/w409_exact_separate_families_joint96.json"
"$PY" "$ROOT/work/evaluate_validation_ridge.py" \
  "$ROOT/work/w409c_val.npy" \
  "$ROOT/work/w409_exact_decay_s93_val.npy" \
  "$ROOT/work/w409_exact_position_s93_val.npy" \
  "$ROOT/work/${tag}_val.npy" \
  --joint --repeats 96 > "$ROOT/work/w409_exact_position_decay_full_joint96.json"
"$PY" "$ROOT/work/evaluate_w409_exact_position_decay.py" \
  > "$ROOT/logs/w409_exact_position_decay_decision.log" 2>&1
cat "$ROOT/logs/w409_exact_position_decay_decision.log"
echo "W409_EXACT_POSITION_DECAY_DONE $(date --iso-8601=seconds)"
