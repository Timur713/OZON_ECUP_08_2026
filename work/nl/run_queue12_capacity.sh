#!/bin/bash
# 214 - capacity and schedule TOGETHER, which has never been tested.
# FINDINGS 2.7 closed the capacity axis: width 96 -> 192 was worth nothing. But
# that factorial ran at two epochs, where the model already overfits by epoch
# three, so a wider network had no room to use its parameters. This round showed
# that a harder learning problem needs four epochs before it pays. The two
# findings have never been crossed.
#
# Every network in the pool is width 96 with eight dilated blocks, about 313k
# parameters, on 3.5M training rows. That is a small model for this much data.
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
run divR_w192_e4 --seed 701 --variant decay --width 192 --epochs 4
run divR_w192_e6 --seed 702 --variant decay --width 192 --epochs 6
run divR_w384_e4 --seed 703 --variant decay --width 384 --epochs 4
run divR_w256_e6 --seed 704 --variant decay --width 256 --epochs 6
# The same crossed with the confirmed representation.
run divR_shape_w192_e4 --seed 705 --variant decay --per-user-scale --width 192 --epochs 4
run divR_shape_w192_e6 --seed 706 --variant decay --per-user-scale --width 192 --epochs 6
run divR_shape_w384_e4 --seed 707 --variant decay --per-user-scale --width 384 --epochs 4
echo "QUEUE12_DONE $(date -u +%H:%M:%S)"
