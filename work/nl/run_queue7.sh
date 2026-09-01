#!/bin/bash
# 188 - the lagged view. sw28 kept only the recent 28 days and was worth a
# measured +0.00011 on public. Masking the recent days is the opposite
# restriction along the same axis: the network never sees the period that
# dominates every other base, so whatever it learns is long-run structure and
# its errors are decorrelated by construction rather than by luck.
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
run divL_mask30  --seed 311 --variant decay --mask-recent 30
run divL_mask60  --seed 312 --variant decay --mask-recent 60
run divL_mask90  --seed 313 --variant decay --mask-recent 90
run divL_mask14  --seed 314 --variant decay --mask-recent 14
run divL_mask60_st24 --seed 315 --variant decay --mask-recent 60 --anchor-stride 24
echo "QUEUE7_DONE $(date -u +%H:%M:%S)"
