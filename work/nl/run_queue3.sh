#!/bin/bash
# 176 - schedule length. The pool's networks all train for exactly 2 epochs.
# FINDINGS 2.6 measured epochs 3 and beyond as worse, but that table came from
# train_seq2 without a one-cycle schedule. Here the schedule is rebuilt for the
# longer run, so the question "is 2 optimal, or was it optimal only because the
# learning rate finished annealing there" is actually being asked.
set -u
cd /home/ubuntu/ecup3
export ECUP_ROOT=/home/ubuntu/ecup3
export ECUP_OUT=/home/ubuntu/ecup3/work/div
PY=./.venv/bin/python
run () {
  local tag="$1"; shift
  if [ -f "work/div/${tag}_final.npy" ]; then echo "SKIP $tag"; return; fi
  echo "=== START $tag $* $(date -u +%H:%M:%S) ==="
  $PY work/train_w409_diverse.py "$tag" "$@" 2>&1 | tail -5
  echo "=== END $tag $(date -u +%H:%M:%S) ==="
}
run divG_e4      --seed 261 --variant decay --epochs 4
run divG_e6      --seed 262 --variant decay --epochs 6
run divG_e4_w192 --seed 263 --variant decay --epochs 4 --width 192
run divG_e4_st24 --seed 264 --variant decay --epochs 4 --anchor-stride 24
run divG_e6_u50  --seed 265 --variant decay --epochs 6 --user-fraction 0.5 --user-seed 8
echo "QUEUE3_DONE $(date -u +%H:%M:%S)"
