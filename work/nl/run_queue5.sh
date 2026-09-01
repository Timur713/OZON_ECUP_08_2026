#!/bin/bash
# 192 - the shape axis is alive; this replaces the old contiguous-block queue.
# divK_shape at 2 epochs scored -0.0000216 and looked like a closed axis. The
# same model at 4 epochs scores +0.0003185, the largest single-base gain of the
# round, and is the strongest model of the night standalone at a calibrated
# 1.667826 against the matched control's 1.667960. Removing level does not
# throw information away the way removing users does; it makes the problem
# harder, so the representation needs a longer schedule before it pays.
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
# Where does the schedule stop paying?
run divM_shape_e6  --seed 321 --variant decay --per-user-scale --epochs 6
run divM_shape_e8  --seed 322 --variant decay --per-user-scale --epochs 8
# Is the e4 result a property of the representation or of one seed?
run divM_shape_e4_s2 --seed 323 --variant decay --per-user-scale --epochs 4
run divM_shape_e4_s3 --seed 324 --variant decay --per-user-scale --epochs 4
# Does the same schedule help the ORDINARY representation, or is the gain
# specific to shape? This is the control that separates the two explanations.
run divM_plain_e4  --seed 325 --variant decay --epochs 4
run divM_plain_e6  --seed 326 --variant decay --epochs 6
# Cross with the other axis that passed.
run divM_shape_e4_st24 --seed 327 --variant decay --per-user-scale --epochs 4 --anchor-stride 24
echo "QUEUE5_DONE $(date -u +%H:%M:%S)"
