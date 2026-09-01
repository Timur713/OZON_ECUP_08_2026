#!/bin/bash
# 219 - thirty targets instead of one.
# Every base in the pool is supervised by a single scalar, the log of the 30-day
# sum. That is one number per user per anchor, about 3.5M numbers in total. The
# daily profile of the same window is thirty numbers per user per anchor. This
# is the only remaining idea that gives the network MORE SUPERVISION rather than
# a different view of the same supervision, and more supervision is the one
# thing that has never been tried here.
#
# The scored output stays the scalar head, so the metric and the ensemble
# interface are unchanged; the daily head only shapes the representation.
set -u
cd /home/ubuntu/ecup3
export ECUP_ROOT=/home/ubuntu/ecup3
export ECUP_OUT=/home/ubuntu/ecup3/work/div
PY=./.venv/bin/python
run () {
  local tag="$1"; shift
  if [ -f "work/div/${tag}_final.npy" ]; then echo "SKIP $tag"; return; fi
  echo "=== START $tag $* $(date -u +%H:%M:%S) ==="
  $PY work/train_w409_diverse.py "$tag" "$@" 2>&1 | tail -4
  echo "=== END $tag $(date -u +%H:%M:%S) ==="
}
run divU_daily03_e2 --seed 1001 --variant decay --daily-head 0.3 --epochs 2
run divU_daily10_e2 --seed 1002 --variant decay --daily-head 1.0 --epochs 2
run divU_daily03_e4 --seed 1003 --variant decay --daily-head 0.3 --epochs 4
run divU_daily03_cal --seed 1004 --variant decay --daily-head 0.3 --calendar --epochs 2
run divU_daily10_e4 --seed 1005 --variant decay --daily-head 1.0 --epochs 4
run divU_daily03_shape --seed 1006 --variant decay --daily-head 0.3 --per-user-scale --epochs 4
run divU_daily03_gift --seed 1007 --variant decay --daily-head 0.3 --calendar --valid-channel --anchor-start 43 --epochs 4
echo "QUEUE15_DONE $(date -u +%H:%M:%S)"
