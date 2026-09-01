#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/ecup
PY="$ROOT/.venv/bin/python"
TRAIN="$ROOT/work/train_w409_exact_decay.py"
AUDIT="$ROOT/work/evaluate_validation_ridge.py"
AVERAGE="$ROOT/work/average_profile_finals.py"
DECIDE="$ROOT/work/evaluate_w409_exact_decay.py"
LOGS="$ROOT/logs"
export ECUP_ROOT="$ROOT"
export ECUP_MAT="$ROOT/work/mat"
export ECUP_OUT="$ROOT/work"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGS"

position_pid=$(cat "$ROOT/work/server_position_tail.pid")
echo "WAIT position tail pid=$position_pid $(date --iso-8601=seconds)"
while kill -0 "$position_pid" 2>/dev/null; do sleep 20; done
for file in "$TRAIN" "$AUDIT" "$AVERAGE" "$DECIDE"; do
  if [ ! -s "$file" ]; then echo "missing exact-decay dependency $file" >&2; exit 1; fi
done

for seed in 93 1310; do
  tag="w409_exact_decay_s${seed}"
  echo "START exact-w409c decay seed=$seed $(date --iso-8601=seconds)"
  "$PY" "$TRAIN" "$tag" --seed "$seed" > "$LOGS/${tag}.log" 2>&1
  "$PY" "$AUDIT" "$ROOT/work/${tag}_val.npy" --repeats 96 \
    > "$ROOT/work/${tag}_ridge96.json"
done

"$PY" "$AVERAGE" w409_exact_decay_val \
  "$ROOT/work/w409_exact_decay_s93_val.npy" \
  "$ROOT/work/w409_exact_decay_s1310_val.npy" \
  > "$LOGS/w409_exact_decay_val_average.log" 2>&1
"$PY" "$AVERAGE" w409_exact_decay \
  "$ROOT/work/w409_exact_decay_s93_final.npy" \
  "$ROOT/work/w409_exact_decay_s1310_final.npy" \
  > "$LOGS/w409_exact_decay_final_average.log" 2>&1
"$PY" "$AUDIT" "$ROOT/work/promote_w409_exact_decay_val_seedavg.npy" --repeats 96 \
  > "$ROOT/work/w409_exact_decay_seedavg_ridge96.json"
"$PY" "$AUDIT" "$ROOT/work/w409c_val.npy" \
  "$ROOT/work/promote_w409_exact_decay_val_seedavg.npy" --joint --repeats 96 \
  > "$ROOT/work/w409_exact_decay_seedavg_w409c_joint96.json"
"$PY" "$DECIDE" > "$LOGS/w409_exact_decay_decision.log" 2>&1
cat "$LOGS/w409_exact_decay_decision.log"
echo "W409_EXACT_DECAY_DONE $(date --iso-8601=seconds)"
